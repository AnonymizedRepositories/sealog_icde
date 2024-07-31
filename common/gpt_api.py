import openai
import os
import re
import time
import json
import string
import numpy as np

openai.api_base = "API_BASE"
openai.api_key = "API_KEY"

instruction = """
Please determine if the given log messages indicate a system run-time anomaly or not. In the following, some similar examples are provided for reference, you should compare the given log messages with them and make your own decision.
"""

def get_openai_key(file_path):
    with open(file_path, "r") as file:
        api_base = file.readline().strip()
        key_str = file.readline().strip()
    return api_base, key_str


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_openai_embedding(text, model="text-embedding-ada-002", chunksize=200):
    # Initialize an empty list to hold the embeddings
    embeddings = []
    # Break the text list into chunks
    for chunk in chunks(text, chunksize):
        while True:  # Infinite loop for retries
            try:
                # Get the embeddings for the chunk
                chunk = [str(line) for line in chunk]
                response = openai.Embedding.create(input=chunk, model=model)["data"]
                # Add the embeddings to our list
                embeddings.extend([item["embedding"] for item in response])
                break  # If successful, break out of retry loop
            except Exception as e:
                print(
                    "Attempt failed with error: {}. Retrying...".format(e)
                )
                time.sleep(3)
    # Convert the list of embeddings to a numpy array and reshape it
    embeddings = np.array(embeddings).reshape(len(text), -1)
    return embeddings



def construct_query(
    instruction,
    examples,
    query,
):
    merged_query = ""
    merged_query += instruction + "\n"
    if examples is not None or len(examples) > 0:
        for idx, item in enumerate(examples):
            merged_query += "Query of example {}:\n".format(idx+1) + "'''\n" + item["query"] + "\n'''\n"
            merged_query += "Answer of example {}:\n".format(idx+1) + "'''\n" + item["answer"] + "\n'''\n" + "\n"
        merged_query += "\n"
    merged_query += \
        '''
The output MUST be in standard JSON format and MUST consist of TWO fields: 'prediction' and 'analysis' .
1. The 'prediction' field. You should choose one of from 'normal' or 'anomaly' according to your analysis, do not use other words. You should make your decision based on the semantics of the given logs without any context. If the log message have content describing system abnormal events (e.g., null pointers, failure), you should choose 'anomaly', otherwise, you should choose 'normal'. For cases that no enough information to make a decision, you should choose 'normal'. 
2. The 'analysis' field. It contains your analysis for the given log messages based on its semantics.

The log messages you should analyse are as follows.
'''

# 

    merged_query += "\nQuery: \n" + "'''\n"+ query + "\n'''\n"
    messages = [
        {"role": "user", "content": merged_query.replace("<*>", "")},
    ]

    return messages


class LLMInfer:
    def __init__(self, outdir, use_cache=True, cache_every_count=1):
        self.use_cache = use_cache
        self.cache_file = os.path.join(outdir, "llm_cache.json")
        self.ask_counter = 0  # Add a counter for ask method
        self.cache_every_count = (
            cache_every_count  # Define the count after which cache should be dumped
        )
        if (
            self.use_cache
            and os.path.exists(self.cache_file)
            and os.path.getsize(self.cache_file) > 0
        ):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def ask(
        self,
        examples,
        query,
        model="gpt-3.5-turbo-0613",
        # model="gpt-4-0613",
        temperature=0.4,
        max_tokens=2048,
    ):
        self.ask_counter += 1  # Increment the counter each time ask is called
        messages = construct_query(instruction, examples, query)
        # Check if result is in cache
        if self.use_cache and query in self.cache:
            # print("[Chatgpt Response] Retrieving result from cache...")
            return self.cache[query]

        retry_times = 0
        while True:
            try:
                answers = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                response = [
                    response["message"]["content"]
                    for response in answers["choices"]
                    if response["finish_reason"] != "length"
                ][0]

                # Cache result
                if self.use_cache:
                    self.cache[query] = response

                # If ask has been called COUNT times, dump the cache
                if self.ask_counter % self.cache_every_count == 0:
                    with open(self.cache_file, "w") as f:
                        json.dump(self.cache, f)
                # print("Not hit!")
                if "gpt-4" in model:
                    time.sleep(1)
                break
            except Exception as e:
                print(e)
                time.sleep(5)
                retry_times += 1

        return response


if __name__ == "__main__":
    # Create an instance of LLMInfer
    llm_infer = LLMInfer(use_cache=True, cache_file="cache.json")

    # Define some test data
    instruction = "Analyse the log messages"
    examples = [
        {"query": "Log 1", "answer": "No anomaly"},
        {"query": "Log 2", "answer": "Anomaly detected"},
    ]
    query = "Log 3"

    # Use the infer_llm method
    response = llm_infer.infer_llm(instruction, examples, query)
    print(response)

    # Use the infer_llm method again with the same query
    # This time, the result should be retrieved from the cache
    response = llm_infer.infer_llm(instruction, examples, query)
    print(response)

    # Delete the llm_infer object
    # This will save the cache to the file 'cache.json'
    del llm_infer
