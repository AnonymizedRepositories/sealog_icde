import re
import os
import csv
import json
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm

from itertools import chain
from datetime import datetime
from collections import defaultdict, deque, Counter, OrderedDict
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from scipy.stats import genextreme
from common.utils import load_pkl_file, calculate_metrics
from sklearn.ensemble import IsolationForest
from sealog.trie import LogCluster, HierTrie
from sealog.bayes_model import NaiveBayesAgent


def get_md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def contains_numbers(string):
    return bool(re.search(r"\d", string))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LRUCache:
    def __init__(self, size):
        self.recent = deque(maxlen=size)

    def put(self, value):
        if value in self.recent:
            self.recent.remove(value)
        self.recent.append(value)

    def pop(self, value):
        self.recent.remove(value)

    def get_values(self):
        return set(self.recent)

    def __contains__(self, value):
        return value in self.recent


class ScoreCache:
    def __init__(self) -> None:
        self.cache = {
            "template_str": {},
            "template_regex": {},
        }

    def insert(self, logcluster, score_dict):
        # two level cache
        self.cache["template_str"][logcluster.template_str] = score_dict
        self.cache["template_regex"][logcluster.template_regex] = score_dict

    def get(self, logcluster, level=2):
        template_str = logcluster.template_str
        template_regex = logcluster.template_regex
        # check first level, direct match
        score_dict = self.cache["template_str"].get(template_str, None)

        if score_dict is None and level == 2:
            # check second level, direct match
            for cached_str, score_dict_ in self.cache["template_str"].items():
                if re.findall(template_regex, cached_str):
                    score_dict = score_dict_
        return score_dict

class ClusterStore:
    def __init__(self, lru_size=30) -> None:
        self.cluster_dict = {}
        self.num_cluster = 0
        self.statistic_dist = defaultdict(int)
        self.latest_cluster_set = LRUCache(
            size=lru_size
        )  # error may occur when window_size > lru_size
        self.score_cache = ScoreCache()

    def insert_new_cluster(self, new_cluster):
        assert new_cluster.cluster_id == "", "insert cluster with id={}".format(
            new_cluster.cluster_id
        )
        cluster_id = self.num_cluster
        new_cluster.cluster_id = cluster_id
        self.cluster_dict[cluster_id] = new_cluster
        self.num_cluster += 1
        self.latest_cluster_set.put(cluster_id)
        return cluster_id

    def update_statistics(self):
        for cluster_id, cluster in self.cluster_dict.items():
            self.statistic_dist[cluster_id] = len(cluster)
        return self.statistic_dist

    def clear_statistics(self):
        print("Clearing statistics of cluster store.")
        for cluster_id, cluster in self.cluster_dict.items():
            self.statistic_dist[cluster_id] = 0
        return self.statistic_dist

    def update_clusterid(self, logcluster_update_dicts):
        for logcluster_update_dict in logcluster_update_dicts:
            for new_logcluster in logcluster_update_dict["new_logclusters"]:
                cluster_id = self.insert_new_cluster(new_logcluster)
                self.latest_cluster_set.put(cluster_id)
            for del_key in logcluster_update_dict["del_clusterids"]:
                self.cluster_dict.pop(del_key)
                if del_key in self.latest_cluster_set:
                    self.latest_cluster_set.pop(del_key)

    def __getitem__(self, key):
        return self.cluster_dict[key]

    def __len__(self):
        return len(self.cluster_dict)

    def __contains__(self, cluster_id):
        return cluster_id in self.cluster_dict

    def insert_score(self, logcluster, score_dict):
        self.score_cache.insert(logcluster, score_dict)

    def get_expert_score(self, logcluster, level=2):
        return self.score_cache.get(logcluster, level)

    def get_recent_clusters(
        self,
    ):
        return self.latest_cluster_set.get_values()


splitter = r"[^\w]+"


class ScaleAD:
    def __init__(
        self,
        args,
        indir="./",
        outdir="./result/",
        depth=4,
        st=0.5,
        rex=[],
        hash_heads=[],
        topk=3,
        max_child=2,
        lru_size=10,
        ask_threshold=None,
        expert=None,
        normalize="minmax",
        temperature=0.2,
    ):
        self.args = args
        self.indir = indir
        self.outdir = os.path.join(outdir, self.args.expname)
        self.rex = rex
        self.hash_heads = hash_heads
        self.ask_threshold = ask_threshold
        self.lru_size = lru_size
        self.topk = topk
        self.depth = depth
        self.st = st
        self.max_child = max_child
        self.expert = expert
        self.bayes_agent = NaiveBayesAgent()
        os.makedirs(self.outdir, exist_ok=True)

        self.hier_trie = HierTrie(
            topk=self.topk, depth=self.depth, st=self.st, max_child=self.max_child
        )
        self.cluster_store = ClusterStore(lru_size=self.lru_size)

    def tokenize(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, "<*>", line)
        result = re.split(splitter, line.strip())
        if contains_numbers(result[0]):
            result[0] = "<*>"
        return [token for token in result if token]

    def tree_process(self, line):
        log_id = line["LineId"]
        log_tokens = self.tokenize(line["Content"])
        hash_str = "-".join([line[head] for head in self.hash_heads])
        matched_cluster = self.hier_trie.add_sequence(
            sequence=log_tokens, sequence_id=log_id, hash_str=hash_str
        )
        matched_cluster.label = line["Label"]
        if (
            matched_cluster.cluster_id not in self.cluster_store
        ):  # it is a new logcluster
            self.cluster_store.insert_new_cluster(matched_cluster)
        self.cluster_store.latest_cluster_set.put(matched_cluster.cluster_id)
        return matched_cluster

    def warmup(self, log_file, save_model=True):
        print("Warming up the detection agent.")

        self.log_filename = os.path.splitext(log_file)[0]
        with open(os.path.join(self.indir, log_file), "rb") as fr:
            data_windows = pickle.load(fr)[0:]
            print("{} sessions loaded.".format(len(data_windows)))

        start_time = time.time()
        train_samples = []
        label_samples = []
        template_set = set()
        for window_idx, window in enumerate(tqdm(data_windows[:]), 1):
            hit_clusters = set()
            for idx, line in enumerate(window, 1):
                matched_cluster = self.tree_process(line)
                template_set.add(matched_cluster.template_str)
                hit_clusters.add(matched_cluster.cluster_id)
                train_samples.append(matched_cluster.template_tokens)
                label_samples.append(matched_cluster.label)

        self.bayes_agent.train(train_samples, label_samples)
        self.bayes_agent.evaluate_samples(train_samples, label_samples)
        end_time = time.time()


        if save_model:
            self.__save_prefix_tree()
        print("Warming up done. [Time taken: {!s}]".format(end_time - start_time))

    def update_with_llm_cache(self, llm_cache):
        # load json
        with open(llm_cache, "r") as f:
            llm_cache = json.load(f)
        for idx, (log_line, results) in enumerate(llm_cache.items()):
            update_label = int(results["prediction"] == "anomaly")
            line = {
                "LineId": idx,
                "Content": log_line,
                "Label": update_label
            }
            matched_cluster = self.tree_process(line)
            update_sample = matched_cluster.template_tokens
            # print(update_sample, update_label)
            # get feedback directly from the groundtruth
            self.bayes_agent.update(update_sample, update_label)

    def detect(
        self,
        log_file,
        cached_model_folder=None,
        evaluation=True,
        save_model=False,
        update_bayes=False,
    ):
        self.detection_filename = os.path.splitext(log_file)[0]
        self.cache_level = 2 if "test" in log_file else 1
        self.states = Counter()

        if cached_model_folder:
            self.__load_prefix_tree(cached_model_folder)
            self.cluster_store.latest_cluster_set = LRUCache(size=self.lru_size)

        start_time = time.time()

        with open(os.path.join(self.indir, log_file), "rb") as fr:
            data_windows = pickle.load(fr)[0:]
            print("{} sessions loaded.".format(len(data_windows)))


        deduplication = True
        if deduplication:
            template_set = set()
            deduplicated_lines = []
            for session in data_windows:
                for line in session:
                    if line["EventTemplate"] not in template_set:
                        template_set.add(line["EventTemplate"])
                        deduplicated_lines.append(line)
            data_windows = [[item] for item in deduplicated_lines]
            print("Number of deduplicated lines: {}".format(len(deduplicated_lines)))

            savepath = os.path.join(self.outdir, f"{self.detection_filename}_deduplicated.csv")
            pd.DataFrame(deduplicated_lines).to_csv(savepath, index=False, quoting=csv.QUOTE_ALL)
            print("Deduplicated logs saved to {}".format(savepath))


        template_set = set()
        window_label = []
        window_pred = []
        for window_idx, window in enumerate(tqdm(data_windows[:]), 1):
            pred_tmp = []
            label_tmp = []
            for idx, line in enumerate(window, 1):
                matched_cluster = self.tree_process(line)
                template_set.add(matched_cluster.template_str)
                pred_tmp.append(self.bayes_agent.predict(matched_cluster.template_tokens))
                label_tmp.append(line["Label"])
            window_pred.append(any(pred_tmp))
            window_label.append(any(label_tmp))
                
            if update_bayes:
                update_sample = matched_cluster.template_tokens
                # get feedback directly from the groundtruth
                update_label = matched_cluster.label
                self.bayes_agent.update(update_sample, update_label)

        calculate_metrics(window_label, window_pred)

        end_time = time.time()

        if save_model:
            self.__save_prefix_tree()

        print(self.states)
        print("Parsing & AD done. [Time taken: {!s}]".format(end_time - start_time))

    def save_parsed_results(self, windows):
        print("Saving parsed results...")
        save_path_structured = os.path.join(
            self.outdir, f"{self.log_filename}_structured.csv"
        )
        save_path_templates = os.path.join(
            self.outdir, f"{self.log_filename}_templates.csv"
        )

        df = pd.DataFrame(chain(*[window for window in windows[0:]]))
        logid2template = {}
        for _, logcluster in self.cluster_store.cluster_dict.items():
            logid2template.update(
                {logid: logcluster.template_str for logid in logcluster.sequence_ids}
            )
        df["EventTemplate"] = df["LineId"].map(logid2template)
        df.to_csv(save_path_structured, index=False, quoting=csv.QUOTE_ALL)

        template_counts = pd.DataFrame(
            df["EventTemplate"].value_counts().reset_index()
        ).rename(mapper={"EventTemplate": "Count", "index": "EventTemplate"}, axis=1)
        template_counts.to_csv(save_path_templates, index=False, quoting=csv.QUOTE_ALL)
        print(f"Parsed logs dumpped to {save_path_structured}.")

    def __load_prefix_tree(self, model_folder):
        model_dict = load_pkl_file(model_folder)
        self.hier_trie = model_dict["hier_trie"]
        self.cluster_store = model_dict["cluster_store"]
        print(f"Load cached model: {model_folder}")

    def __save_prefix_tree(self):
        save_path = os.path.join(self.outdir, f"{self.log_filename}_model.pkl")
        with open(save_path, "wb") as fw:
            pickle.dump(
                {"hier_trie": self.hier_trie, "cluster_store": self.cluster_store}, fw
            )
        json_path = os.path.join(self.outdir, "args.json")
        with open(json_path, "w") as f:
            json.dump(vars(self.args), f, indent=4)
        print(f"The trained model is saved to {save_path}.")
