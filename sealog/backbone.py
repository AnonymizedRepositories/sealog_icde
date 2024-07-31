import re
import os
import sys

sys.path.append("../")
import pickle
import json
import time
import random
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from common.gpt_api import LLMInfer
from sealog.embedding_db import EmbeddingDB, ICLModel
from collections import defaultdict
from common.utils import sliding_window, calculate_metrics, extract_json


def parse_llm_result(llm_result_str):
    try:
        llm_result = extract_json(llm_result_str)
    except Exception:
        raise ValueError(f"Cannot extract JSON from {llm_result_str}")
    if "prediction" not in llm_result:
        raise ValueError(f"Missing fields in LLM result: {llm_result}")
    else:
        if llm_result["prediction"] not in ["anomaly", "normal"]:
            raise ValueError("Wrong value in 'prediction' field: {}".format(llm_result["prediction"]))
    return llm_result


class Backbone:
    def __init__(self, args, indir="./", outdir="./result/", regex=[], icl_topk=3):
        self.args = args
        self.indir = indir
        self.outdir = outdir
        self.icl_topk = icl_topk
        self.regex = regex
        self.icl_model = ICLModel(outdir, use_cache=False)
        self.llm_infer = LLMInfer(outdir, use_cache=False)

        self.cache_file = os.path.join(outdir, "llm_cache.json")
        if os.path.exists(self.cache_file) and os.path.getsize(self.cache_file) > 0:
            print("Loading LLM cache file from {}".format(self.cache_file))
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        os.makedirs(self.outdir, exist_ok=True)

    def predict(self, query):
        # Check if the query is in the cache
        if query in self.cache:
            return self.cache[query]

        # the cache should be here!!
        query_start_time = time.time()
        if self.icl_topk > 0:
            examples = self.icl_model.select_samples(query, topk=self.icl_topk)
        else:
            examples = []

        while True:
            # print("start querying")
            llm_result_str = self.llm_infer.ask(examples, query)
            # print("end querying")
            try:
                llm_result = parse_llm_result(llm_result_str)
                break
            except Exception as e:
                print("Error in parsing LLM result: {}".format(e))
        # print("Query time: {:.3f}s".format(time.time() - query_start_time))
        
        # if query in self.error_lines:
        #     print(query)
        #     print()
        #     print(llm_result)
        #     print("-" * 10)
        llm_result["examples"] = examples
        return {query: llm_result}

    def initialize_icl_model(self, train_file):
        self.icl_model.init_candidates(os.path.join(self.indir, train_file))

    def pre_query(self, data_sessions, num_workers=20):
        test_error = False

        if test_error:
            compare_df = pd.read_csv(os.path.join(self.outdir, "llm_gt_compare.csv"))
            self.error_lines = set(compare_df[~compare_df["correct"]]["EventTemplate"])
            query_set = self.error_lines
            print("Number of error queries: {}".format(len(error_lines)))
        else:
            query_set = set()
            for session_idx, session in enumerate(tqdm(data_sessions[:]), 1):
                windows = sliding_window(
                    session, window_size=self.args.window_size, stride=self.args.stride
                )
                for idx, window in enumerate(windows, 1):
                    content_list = [
                        line["EventTemplate"]
                        for line in window
                        if not str(line["EventTemplate"]).strip() == ""
                    ]
                    query = "\n".join(content_list)
                    query_set.add(query)
            print("Number of unique queries: {}".format(len(query_set)))

        if num_workers == 1:
            results = []
            for query in tqdm(list(query_set)):
                results.append(self.predict(query))

        else:
            with mp.Pool(num_workers) as pool:
                results = []
                tasks = [
                    pool.apply_async(self.predict, (query,)) for query in query_set
                ]
                for task in tqdm(tasks, total=len(query_set)):
                    results.append(task.get())

        for dictionary in results:
            self.cache.update(dictionary)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def detect(self, log_file, num_workers=20, use_pre_query=False):
        with open(os.path.join(self.indir, log_file), "rb") as fr:
            data_sessions = pickle.load(fr)[0:]
            print("{} sessions loaded.".format(len(data_sessions)))

        deduplication = False
        if deduplication:
            template_set = set()
            deduplicated_lines = []
            for session in data_sessions:
                for line in session:
                    if line["EventTemplate"] not in template_set:
                        template_set.add(line["EventTemplate"])
                        deduplicated_lines.append(line)
            data_sessions = [[item] for item in deduplicated_lines]
            print("Number of deduplicated lines: {}".format(len(deduplicated_lines)))

        # Pre-query and save results to the cache
        if use_pre_query:
            self.pre_query(data_sessions, num_workers)

        results_save = []
        session_label = []
        session_pred = []
        start_time = time.time()
        for session_idx, session in enumerate(tqdm(data_sessions[:]), 1):
            pred_tmp = []
            label_tmp = []
            windows = sliding_window(
                session, window_size=self.args.window_size, stride=self.args.stride
            )
            for idx, window in enumerate(windows, 1):
                content_list = [
                    line["EventTemplate"]
                    for line in window
                    if not str(line["EventTemplate"]).strip() == ""
                ]
                query = "\n".join(content_list)
                llm_result = self.predict(query=query)
                # print(llm_result)
                pred_tmp.append(int(llm_result["prediction"] == "anomaly"))
                label_tmp.append(any([line["Label"] for line in window]))

                results_save.append(
                    {"prediction": pred_tmp[-1], "label": label_tmp[-1], "query": query, "analysis": llm_result["analysis"]}
                )

            session_pred.append(any(pred_tmp))
            session_label.append(any(label_tmp))

        results_save = pd.DataFrame(results_save)
        results_save["correct"] = results_save["prediction"].astype(int) == results_save["label"].astype(int)
        results_save.to_csv(os.path.join(self.outdir, "llm_gt_compare.csv"), index=False)

        metrics = calculate_metrics(session_label, session_pred)
        metrics["time"] = time.time() - start_time
        return metrics
