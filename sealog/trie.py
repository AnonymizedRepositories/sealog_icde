import re
import heapq
from collections import Counter, defaultdict, deque, OrderedDict
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from tqdm import tqdm



# def seq_distance(seq1, seq2):
#     assert len(seq1) == len(seq2)
#     common_token_count = 0
#     num_params = 0

#     for token1, token2 in zip(seq1, seq2):
#         if token1 == "<*>":
#             num_params += 1
#         if token1 == token2:
#             common_token_count += 1

#     similarity = float(common_token_count) / len(seq1)
#     return similarity, num_params



def merge_consecutive_items(input_list, item):
    merged_list = []
    in_consecutive_sequence = False

    for elem in input_list:
        if elem == item:
            if not in_consecutive_sequence:
                merged_list.append(elem)
                in_consecutive_sequence = True
        else:
            merged_list.append(elem)
            in_consecutive_sequence = False

    return merged_list

def template2regex(template_tokens):
    template = " ".join(template_tokens)
    template = re.sub(r"(<\*>\s?){2,}", "<*>", template)
    regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template)
    regex = regex.replace("\<\*\>", "(.*?)")
    regex = "^" + regex + "$"
    return regex


def seq_distance(logcluster, seq):
    # if match regex:
    if re.findall(logcluster.template_regex, " ".join(seq)):
        return 1
    else:
        template_set, seq_set = set(logcluster.template_tokens), set(seq)
        similarity = len(template_set & seq_set) / len(template_set | seq_set)
        return similarity


class Vocab:
    def __init__(self, stopwords=["<*>"]):
        self.token_counter = Counter()
        self.stopwords = frozenset(set(ENGLISH_STOP_WORDS) | set(stopwords))

    def build(self, sequences):
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk):
        sequence = self.__filter_stopwords(sequence)
        token_count = [(token, self.token_counter[token]) for token in set(sequence)]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: x[1])
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token.lower() not in self.stopwords)
        ]


class TrieNode:
    def __init__(self):
        self.children = OrderedDict()
        self.is_terminal = False
        self.logcluster_list = []

    def __len__(self):
        return len(self.children)


class LogCluster:
    def __init__(self, sequence_id, template_tokens="", cluster_id="") -> None:
        self.sequence_ids = (
            [sequence_id] if not isinstance(sequence_id, list) else sequence_id
        )
        self.template_regex = ""
        self.template_tokens = ""
        self.template_str = ""
        self.cluster_id = cluster_id
        self.expert_score = {"value": 0, "confidence": 0}
        self.update_template(template_tokens)

    def __len__(self):
        return len(self.sequence_ids)

    def update_template(self, sequence, use_lcs=True):
        if not self.template_tokens:
            result_tokens = sequence
        else:
            if use_lcs:  # TODO
                # res = pylcs.lcs_sequence_idx(self.template_tokens, sequence)
                common_tokens = set(self.template_tokens) & set(sequence)
                longger_tokens = max(
                    [self.template_tokens, sequence], key=lambda x: len(x)
                )
                result_tokens = [
                    "<*>" if token not in common_tokens else token
                    for token in longger_tokens
                ]
                # print(result_tokens)
                # if "<*>" in set(result_tokens) and len(set(result_tokens))==1:
                #     embed()
            else:
                result_tokens = []
                i = 0
                for token in self.template_tokens:
                    if token == sequence[i]:
                        result_tokens.append(token)
                    else:
                        result_tokens.append("<*>")
                    i += 1
        
        # result_tokens = merge_consecutive_items(result_tokens, "<*>")
        self.template_tokens = result_tokens
        self.template_regex = template2regex(result_tokens)
        self.template_str = " ".join(result_tokens)


class Trie:
    def __init__(self, depth=None, st=0.5, max_child=5):
        self.root = TrieNode()
        self.depth = depth
        self.st = st
        self.max_child = max_child

    def __find_nearest_logcluster(self, sequence, logcluster_list):
        max_sim = -1
        max_cluster = None
        for logcluster in logcluster_list:
            simi = seq_distance(logcluster, sequence)
            if simi >= max_sim:
                max_sim = simi
                max_cluster = logcluster
        if max_sim >= self.st:
            return max_cluster
        else:
            return None

    def insert(self, sequence, sequence_id, root=None):
        sequence_id = (
            [sequence_id] if not isinstance(sequence_id, list) else sequence_id
        )

        current_node = self.root if root is None else root
        # print("---")
        # print(sequence)

        for i, token in enumerate(sequence):
            if self.depth and i == self.depth:
                break
            # print(token)
            # print(current_node.children.keys())
            # print("Lens of split", len(current_node))
            if token in current_node.children:
                current_node = current_node.children[token]  # step to the next layer
                # print(f"find child node to {token}, step next")
            elif "<*>" in current_node.children:  # cannot match single-word one
                # print("Matching <*>")
                current_node = current_node.children["<*>"]  # step to the next layer
                sequence[i] = "<*>"

            else:  # cannot find exact match and star match
                if len(current_node) < self.max_child:  # within maxchild limit
                    current_node.children[token] = TrieNode()  # create
                    current_node = current_node.children[
                        token
                    ]  # and step to the next layer
                    # print(f"adding child node to {token}")
                elif len(current_node) >= self.max_child:
                    current_node.children["<*>"] = TrieNode()  # create
                    current_node = current_node.children[
                        "<*>"
                    ]  # and step to the next layer
                    sequence[i] = "<*>"
                    # print(f"max child limited, to {token}")

        # if "<*>" in set(sequence) and len(set(sequence))==1:
        #     embed()
        current_node.is_terminal = True

        if len(current_node.logcluster_list) == 0:
            nearest_cluster = None
        else:
            nearest_cluster = self.__find_nearest_logcluster(
                sequence, current_node.logcluster_list
            )

        if nearest_cluster is not None:
            # if match, insert the log
            nearest_cluster.update_template(sequence)
            nearest_cluster.sequence_ids.extend(sequence_id)
            return nearest_cluster
        else:
            # if not match, initiate a new cluster with the log
            new_cluster = LogCluster(sequence_id, template_tokens=sequence)
            current_node.logcluster_list.append(new_cluster)
            return new_cluster

    def search(self, sequence):
        current_node = self.root
        for token in sequence:
            if token not in current_node.children:
                return False
            current_node = current_node.children[token]
        if current_node.is_terminal:
            return current_node
        else:
            return False

    def merge_leaf_nodes(self):
        log_clusters = self.__get_leaf_logclusters()

        # print("num of log_clusters:", len(log_clusters))
        if len(log_clusters) == 1:
            return

        logclusters_sorted = sorted(
            log_clusters, key=lambda x: (x.template_tokens.count("<*>"), len(x.template_tokens)), reverse=True
        )

        # print("---")
        # for item in logclusters_sorted:
        #     item = item.template_tokens
        #     print(item, item.count("<*>"))

        pointer_queue = deque([0])
        matched_idx_set = set()
        merge_results_indice = []
        while len(matched_idx_set) < len(logclusters_sorted):
            pointer_idx = pointer_queue.popleft()
            if pointer_idx in matched_idx_set:
                continue
            matched_idx_set.add(pointer_idx)

            merge_results_indice.append([pointer_idx])

            # next_template = logclusters_sorted[pointer_idx].template_tokens
            template_regex = logclusters_sorted[pointer_idx].template_regex

            # print("regex: ", template_regex)
            for candidate_idx in range(len(logclusters_sorted)):
                if candidate_idx in matched_idx_set:
                    continue
                msg = " ".join(
                    logclusters_sorted[candidate_idx].template_tokens
                ).strip()

                if re.findall(template_regex, msg.strip()):
                    merge_results_indice[-1].append(candidate_idx)
                    matched_idx_set.add(candidate_idx)
                else:
                    pointer_queue.append(candidate_idx)

        # print(merge_results_indice)


        del_clusterids = []
        merged_candidates = []
        update_flag = False
        # print("merge_results_indice: ", merge_results_indice)
        for idx_list in merge_results_indice:
            if len(idx_list) > 1:
                update_flag = True
            # only take the most representative one
            merged_template = logclusters_sorted[idx_list[0]].template_tokens
            merged_sequence_ids = []
            for idx in idx_list:
                merged_sequence_ids.extend(logclusters_sorted[idx].sequence_ids)
                del_clusterids.append(logclusters_sorted[idx].cluster_id)
            merged_candidates.append((merged_template, merged_sequence_ids))

        if update_flag:
            # print("Updating prefix tree.")
            new_root = TrieNode()
            new_logclusters = []
            for merged_template, merged_sequence_ids in merged_candidates:
                # TODO: debug here: ['Setting', 'hostname', '<*>']
                # print(merged_template)
                # if "Setting" in merged_template:
                #     embed()
                new_logcluster = self.insert(
                    merged_template, merged_sequence_ids, root=new_root
                )
                new_logclusters.append(new_logcluster)

                assert new_logcluster.cluster_id == "", "New cluster has cluster_id, check."
                    
            new_logclusters = list({id(obj): obj for obj in new_logclusters}.values())

            self.root = new_root
            logcluster_update_dict = {
                "del_clusterids": del_clusterids,
                "new_logclusters": new_logclusters,
            }
            # print("Delect:", del_clusterids)
            return logcluster_update_dict

    def __get_leaf_nodes(self, node=None):
        if node is None:
            node = self.root

        leaf_nodes = []
        if node.is_terminal:
            leaf_nodes.append(node)

        for child_node in node.children.values():
            leaf_nodes += self.__get_leaf_nodes(child_node)
        return leaf_nodes

    def __get_leaf_logclusters(
        self,
    ):
        leaf_nodes = self.__get_leaf_nodes()
        log_clusters = []
        for leaf_node in leaf_nodes:
            log_clusters.extend(leaf_node.logcluster_list)
        return log_clusters

    def print_trie(self):
        self._print_trie_helper(self.root, "")

    def _print_trie_helper(self, node, prefix):
        if node.is_terminal:
            for logcluster in node.logcluster_list:
                print(logcluster.template_tokens)
                print("# of logs: {}".format(len(logcluster.sequence_ids)))

        for token, child_node in node.children.items():
            self._print_trie_helper(child_node, prefix + " " + token)


class HierTrie:
    def __init__(self, topk=3, depth=None, st=0.5, max_child=5):
        self.trie_per_group = OrderedDict()
        self.vocab = Vocab()
        self.topk = topk
        self.depth = depth
        self.st = st
        self.max_child = max_child

    def add_sequence(self, sequence, sequence_id, hash_str="", update_vocab=True):
        if update_vocab:
            self.__update_vocab(sequence)
        group_id = self.__assign_group(sequence, hash_str)
        if group_id not in self.trie_per_group:
            self.trie_per_group[group_id] = Trie(
                depth=self.depth, st=self.st, max_child=self.max_child
            )
        matched_cluster = self.trie_per_group[group_id].insert(sequence, sequence_id)
        return matched_cluster

    # def search(self, search_sequence):
    #     group_id = self.__assign_group(search_sequence)
    #     search_result = self.trie_per_group[group_id].search(search_sequence)
    #     return search_result

    def __update_vocab(self, sequence):
        self.vocab.update(sequence)

    def __assign_group(self, sequence, hash_str=""):
        topk_tokens = sorted(self.vocab.topk_tokens(sequence, topk=self.topk))
        length = len(sequence)
        # return f"{length}-{topk_tokens}"
        if hash_str:
            return f"{hash_str}-{topk_tokens}"
        else:
            return f"{topk_tokens}"


    def print_hier_trie(self):
        for k, trie in self.trie_per_group.items():
            print("Token:", k)
            trie.print_trie()

    def merge_leaf_nodes(
        self,
    ):
        logcluster_update_dicts = []
        for group_id in self.trie_per_group:
            # print("Groupid: ", group_id)
            logcluster_update_dict = self.trie_per_group[group_id].merge_leaf_nodes()
            if logcluster_update_dict is not None:
                logcluster_update_dicts.append(logcluster_update_dict)
        return logcluster_update_dicts


if __name__ == "__main__":
    # sequences = [
    #     "jABIpVMT from2 root size2 class nrcpts3 msgid jABIpVMT eadmin relay3 localhost",
    #     "jABIpWqD from2 root size2 class nrcpts3 msgid jABIpWqD badmin relay3 localhost",
    #     "jABIpX R from2 root size2 class nrcpts3 msgid jABIpX R aadmin relay3 localhost",
    #     "jABJ V e from2 root size2 class nrcpts3 msgid jABJ V e eadmin relay3 localhost",
    #     "jABJ W A from2 root size2 class nrcpts3 msgid jABJ W A cadmin relay3 localhost",
    #     "jABB L from root size class nrcpts msgid jABB L relay localhost",
    #     "jABB J from root size class nrcpts msgid jABB J relay localhost",
    #     "jA JLV q from root size class nrcpts msgid jA JLV q eadmin relay localhost",
    #     "jACB Qo from root size class nrcpts msgid jACB Qo relay localhost",
    #     "jACB S from root size class nrcpts msgid jACB S relay localhost",
    #     "jADB JX from root size class nrcpts msgid jADB JX relay localhost",
    #     "jADB Zj from root size class nrcpts msgid jADB Zj relay localhost",
    #     "jADBPUuH from root size class nrcpts msgid jADBPUuH relay localhost",
    #     "jADB h t from root size class nrcpts msgid jADB h t relay localhost",]
    #     # "jAN Lrw from root size class nrcpts msgid jAN Lrw cadmin relay localhost",
    #     # "jAN LdwD from root size class nrcpts msgid jAN LdwD aadmin relay localhost",
    #     # "jAN LYxm from root size class nrcpts msgid jAN LYxm badmin relay localhost",
    #     # "jAN LYq from root size class nrcpts msgid jAN LYq dadmin relay localhost",
    #     # "jAN LU n from root size class nrcpts msgid jAN LU n eadmin relay localhost",
    #     # "jAN BocL from root size class nrcpts msgid jAN BocL cadmin relay localhost",
    #     # "jAN BYNc from root size class nrcpts msgid jAN BYNc dadmin relay localhost",
    #     # "jAN fVKi from root size class nrcpts msgid jAN fVKi eadmin relay localhost",
    #     # "jANB Jfu from root size class nrcpts msgid jANB Jfu cn relay localhost",
    #     # "jANA Ubl from root size class nrcpts msgid jANA Ubl eadmin relay localhost",
    #     # "jANALYdL from root size class root size class nrcpts msgid jANALYdL badmin relay localhost",
    #     # "jANALYqx from root size class root size class nrcpts msgid jANALYqx dadmin relay localhost",
    #     # "jANABZfP from root size class root size class nrcpts msgid jANABZfP dadmin relay localhost",
    #     # "jANABYTB from root size class root size class nrcpts msgid jANABYTB badmin relay localhost",
    #     # "jANALdEU from root size class root size class nrcpts msgid jANALdEU aadmin relay localhost",
    #     # "jAN BYwI to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jAN LYxm to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jAN LYq to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jAN LU n to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jAN BocL to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jAN BYNc to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jAN fXSi to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jANALYdL to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jANALYqx to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jANABZfP to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    #     # "jANABYTB to root ctladdr root delay xdelay mailer relay pri relay dsn stat Deferred Connection",
    # ]

    sequences = [
        "dhcpd startup succeeded",
        "httpd startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
        "portmap startup succeeded",
        "irqbalance startup succeeded",
        "snmpd startup succeeded",
        "acpid startup succeeded",
        "xinetd startup succeeded",
        "pbs_mom startup succeeded",
        "crond startup succeeded",
        "gmond startup succeeded",
        "omsad32 startup succeeded",
        "messagebus startup succeeded",
        "haldaemon startup succeeded",
        "dhcpd startup succeeded",
        "httpd startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
        "irqbalance startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
        "gmond startup succeeded",
        "dhcpd startup succeeded",
        "httpd startup succeeded",
        "messagebus startup succeeded",
        "haldaemon startup succeeded",
        "dhcpd startup succeeded",
        "httpd startup succeeded",
        "klogd startup succeeded",
        "syslogd startup succeeded",
        "gmond startup succeeded",
        "portmap startup succeeded",
        "gmetad startup succeeded",
        "klogd startup succeeded",
        "irqbalance startup succeeded",
        "portmap startup succeeded",
        "snmpd startup succeeded",
        "Unable to register device /dev/sda (no Directive -d removable). Exiting.",
        "acpid startup succeeded",
        "crond startup succeeded",
        "gmond startup succeeded",
        "messagebus startup succeeded",
        "omsad32 startup succeeded",
        "pbs_mom startup succeeded",
        "haldaemon startup succeeded",
        "portmap startup succeeded",
        "klogd startup succeeded",
        "syslogd startup succeeded",
        "irqbalance startup succeeded",
        "Unable to register device /dev/sda (no Directive -d removable). Exiting.",
        "acpid startup succeeded",
        "snmpd startup succeeded",
        "gmond startup succeeded",
        "pbs_mom startup succeeded",
        "omsad32 startup succeeded",
        "crond startup succeeded",
        "haldaemon startup succeeded",
        "messagebus startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
        "portmap startup succeeded",
        "irqbalance startup succeeded",
        "Unable to register device /dev/sda (no Directive -d removable). Exiting.",
        "fsck.ext3 -a /dev/sda5",
        "acpid startup succeeded",
        "snmpd startup succeeded",
        "pbs_mom startup succeeded",
        "gmond startup succeeded",
        "crond startup succeeded",
        "omsad32 startup succeeded",
        "haldaemon startup succeeded",
        "messagebus startup succeeded",
        "portmap startup succeeded",
        "23999988k swap on /mnt//dev/sda1. Priority:-1 extents:1",
        "gmetad startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
        "portmap startup succeeded",
        "irqbalance startup succeeded",
        "snmpd startup succeeded",
        "acpid startup succeeded",
        "crond startup succeeded",
        "gmond startup succeeded",
        "pbs_mom startup succeeded",
        "messagebus startup succeeded",
        "omsad32 startup succeeded",
        "haldaemon startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
        "portmap startup succeeded",
        "irqbalance startup succeeded",
        "acpid startup succeeded",
        "snmpd startup succeeded",
        "pbs_mom startup succeeded",
        "gmond startup succeeded",
        "crond startup succeeded",
        "haldaemon startup succeeded",
        "messagebus startup succeeded",
        "omsad32 startup succeeded",
        "portmap startup succeeded",
        "klogd startup succeeded",
        "syslogd startup succeeded",
        "irqbalance startup succeeded",
        "smartd startup succeeded",
        "acpid startup succeeded",
        "snmpd startup succeeded",
        "pbs_mom startup succeeded",
        "omsad32 startup succeeded",
        "gmond startup succeeded",
        "messagebus startup succeeded",
        "crond startup succeeded",
        "haldaemon startup succeeded",
        "portmap startup succeeded",
        "23999988k swap on /mnt//dev/sda1. Priority:-1 extents:1",
        "gmetad startup succeeded",
        "klogd startup succeeded",
        "syslogd startup succeeded",
        "gmond startup succeeded",
        "portmap startup succeeded",
        "23999988k swap on /mnt//dev/sda1. Priority:-1 extents:1",
        "gmetad startup succeeded",
        "portmap startup succeeded",
        "23999988k swap on /mnt//dev/sda1. Priority:-1 extents:1",
        "gmetad startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
        "dhcpd startup succeeded",
        "gmond startup succeeded",
        "portmap startup succeeded",
        "23999988k swap on /mnt//dev/sda1. Priority:-1 extents:1",
        "gmetad startup succeeded",
        "syslogd startup succeeded",
        "klogd startup succeeded",
    ]

    ht = HierTrie(topk=2, depth=4, st=0.4, max_child=1)

    for idx, sequence in enumerate(tqdm(sequences)):
        sequence = sequence.split()
        ht.add_sequence(sequence, idx)

    print("seq len:", len(sequences))
    # ht.print_hier_trie()
    # ht.merge_paths()
    ht.merge_leaf_nodes()
    print("\nPrint trees:")
    ht.print_hier_trie()

    # def find_patterns(seq1, seq2):
    #     patterns = []
    #     i = 0
    #     while i < len(seq1):
    #         j = 0
    #         while j < len(seq2):
    #             if seq1[i] == seq2[j]:
    #                 start_i = i
    #                 start_j = j
    #                 end_i = i
    #                 end_j = j
    #                 while end_i < len(seq1) and end_j < len(seq2) and seq1[end_i] == seq2[end_j]:
    #                     end_i += 1
    #                     end_j += 1
    #                 if end_i - start_i > 1:
    #                     patterns.append(seq1[start_i:end_i])
    #                 i = end_i - 1
    #                 j = end_j - 1
    #             j += 1
    #         i += 1
    #     return patterns

    # def find_common_consecutive_sequences(seq1, seq2):
    #     common_sequences = []
    #     for i in range(len(seq1)):
    #         for j in range(len(seq2)):
    #             if seq1[i] == seq2[j]:
    #                 common_sequence = [seq1[i]]
    #                 k = 1
    #                 while i + k < len(seq1) and j + k < len(seq2) and seq1[i+k] == seq2[j+k]:
    #                     common_sequence.append(seq1[i+k])
    #                     k += 1
    #                 if len(common_sequence) > 1:
    #                     common_sequences.append(common_sequence)
    #     return common_sequences

    # seq1 = ["a", "b", "c", "e", "k", "m", "k", "m"]
    # seq2 = ["a", "b", "c", "x", "k", "m", "o"]
    # print(find_common_consecutive_sequences(seq1, seq2))
