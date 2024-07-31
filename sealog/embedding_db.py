import os
import sys
import json

sys.path.append("../")
import pandas as pd
import numpy as np
import faiss
import pickle
from collections import defaultdict
from typing import List, Tuple
from common.gpt_api import get_openai_embedding

from tqdm import tqdm
import re
import random


def preprocess(log, regex_list):
    for regex in regex_list:
        log = re.sub(regex, "", log)
    return log if log != "" else None


class ICLModel:
    def __init__(self, outdir, use_cache=True):
        self.outdir = outdir
        self.cache_file = os.path.join(outdir, "icl_cache.json")
        self.use_cache = use_cache

        if (
            self.use_cache
            and os.path.exists(self.cache_file)
            and os.path.getsize(self.cache_file) > 0
        ):
            print("Loading ICL cache from file.")
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
            print("{} entries loaded.".format(len(self.cache)))
        else:
            print("Initializing empty ICL cache.")
            self.cache = {}

    def init_candidates(self, train_file):
        """
        The fault lib is intialized with the training data LINE BY LINE.
        """

        if os.path.exists(os.path.join(self.outdir, "embedding_db.pkl")):
            self.embedding_db = EmbeddingDB.load(
                os.path.join(self.outdir, "embedding_db.pkl")
            )
            return
        else:
            self.embedding_db = EmbeddingDB(outdir=self.outdir)

        with open(train_file, "rb") as fr:
            data_sessions = pickle.load(fr)[0:]
            print("{} sessions loaded as ICL candidates.".format(len(data_sessions)))

        compat_df = pd.DataFrame(
            [
                {"EventTemplate": line["EventTemplate"], "Label": line["Label"]}
                for window in data_sessions[0:]
                for line in window
            ]
        )
        # compat_df["EventTemplate"] = compat_df["EventTemplate"].map(lambda x: preprocess(x, self.regex)).dropna()
        compat_df.drop_duplicates(subset=["EventTemplate"], inplace=True)
        print(
            "ICL candidates have {} lines of unique log messages.".format(
                len(compat_df)
            )
        )

        candidates = defaultdict(list)
        for line in compat_df.to_dict("records"):
            query = line["EventTemplate"]
            label = line["Label"]
            answer = {
                "prediction": "normal" if not label else "anomaly",
                "anomaly_score": round(random.uniform(0, 0.1), 2)
                if not label
                else round(random.uniform(0.9, 1), 2),
            }
            candidates["query"].append(query)
            candidates["answer"].append(json.dumps(answer))
        self.embedding_db.insert_batch(candidates["query"], candidates["answer"])
        print(
            "{} candidates inserted to embedding DB.".format(len(candidates["query"]))
        )
        self.embedding_db.save()

    def select_samples(self, query, topk):
        # Check if result is in cache
        if self.use_cache and query in self.cache:
            # print("[ICL Samples] revtrieved from cache.")
            candidates = self.cache[query]
        else:
            candidates = self.embedding_db.query(query, topk)
        # Cache result
        if self.use_cache:
            self.cache[query] = candidates
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)

        # Convert to LLM format
        examples = []
        for item in candidates:
            examples.append({"query": item["key"], "answer": item["value"]})
        return examples


class EmbeddingDB:
    def __init__(self, outdir):
        self.outdir = outdir
        self.index = faiss.IndexFlatL2(1536)  # Assuming embeddings are of size 768
        self.id_to_sentence = {}
        self.id_to_value = {}

    def query(self, sentence: str, k: int) -> List[Tuple[str, float]]:
        embedding = get_openai_embedding([sentence])
        distances, indices = self.index.search(embedding, k)
        results = sorted(list(zip(distances[0], indices[0])), key=lambda x: x[0], reverse=True)
        results = [
            {
                "key": self.id_to_sentence[idx],
                "value": self.id_to_value[idx],
            }
            for dist, idx in results
        ]
        return results

    def insert(self, sentence: str, value: str) -> None:
        embedding = get_openai_embedding(sentence)
        if not self.index.is_trained:
            self.index.train(embedding)
        self.index.add(embedding)
        idx = self.index.ntotal - 1
        self.id_to_sentence[idx] = sentence
        self.id_to_value[idx] = value

    def insert_batch(self, sentences: List[str], values: List[str]) -> None:
        embeddings = get_openai_embedding(sentences)
        assert len(embeddings) == len(sentences)
        for embedding in embeddings:
            self.index.add(embedding.reshape(1, -1))
        for idx, (sentence, value) in enumerate(zip(sentences, values)):
            self.id_to_sentence[self.index.ntotal - len(sentences) + idx] = sentence
            self.id_to_value[self.index.ntotal - len(sentences) + idx] = value

    def save(self):
        print("Saving embedding db...")
        with open(os.path.join(self.outdir, "embedding_db.pkl"), "wb") as fw:
            pickle.dump(self, fw)

    @classmethod
    def load(self, filepath):
        print("Loading embedding db from {}...".format(filepath))
        with open(filepath, "rb") as fr:
            return pickle.load(fr)


if __name__ == "__main__":
    # Initialize the EmbeddingDB
    db = EmbeddingDB()

    sentences = [
        "instruction cache parity error corrected",
        "instruction cache parity error corrected",
        # "MidplaneSwitchController performing bit sparing on R-M-L-U- bit ",
        # "generating ",
        # " ddr errors(s) detected and corrected on rank , symbol , bit ",
        # " L EDRAM error(s) (dcr ) detected and corrected",
        # " sym , at , mask ",
        # "total of  ddr error(s) detected and corrected",
        # "ddr: activating redundant bit steering: rank= symbol=",
        # "ddr: excessive soft failures, consider replacing the card",
        # "ciod: Error loading /p/gb/stella/RAPTOR//raptor: invalid or missing program image, No such file or directory",
        # " double-hummer alignment exceptions",
        # " tree receiver  in re-synch state event(s) (dcr ) detected",
        # "ciodb has been restarted.",
        # "idoproxydb has been started: $Name: DRV_ $ Input parameters: -enableflush -loguserinfo .properties BlueGene",
        # "mmcs_db_server has been started: ./mmcs_db_server --useDatabase BGL --dbproperties serverdb.properties --iolog /bgl/",
        # "BlueLight/logs/BGL --reconnect-blocks all",
        # "ciod: LOGIN chdir(/p/gb/draeger/benchmark/datk_) failed: No such file or directory",
        # "ciod: failed to read message prefix on control stream (CioStream socket to :",
        # "ciod: LOGIN chdir(/p/gb/glosli) failed: No such file or directory",
        # " L directory error(s) (dcr ) detected and corrected",
    ]

    # Insert some sentences with associated labels
    # db.insert("This is a test sentence.", "label1")
    # db.insert("Another test sentence.", "label2")
    # db.insert("Yet another test sentence.", "label3")
    # db.insert("This is a completely different sentence.", "label4")

    # # Query the database
    # print(db.query("This is a test sentence.", 2))
    # print(db.query("A completely different sentence.", 2))

    # Insert a batch of sentences with associated labels
    db.insert_batch(sentences, ["label1"] * len(sentences))

    # Query the database
    print(db.query("instruction cache parity error corrected", 2))
