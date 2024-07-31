import os
import re
import csv
import yaml
import fcntl
import json
import hashlib
import glob
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score

def extract_json(input_string):
    content = re.findall(r'\{.*\}', input_string, re.DOTALL)
    if len(content)>0:
        try:
            return json.loads(content[0])
        except:
            raise ValueError
    else:
        raise ValueError

def sliding_window(lst, window_size=1, stride=1):
    return [lst[i:i+window_size] for i in range(0, len(lst)-window_size+1, stride)]


def calculate_metrics(y_true, y_pred):
    print("Calculating metrics using {} cases".format(len(y_true)))
    precision, recall, f_score = precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)
    print("pc={:.3f}, re={:.3f}, f1={:.3f}".format(precision, recall, f_score))
    # return results as a dict
    return {
        "precision": precision,
        "recall": recall,
        "f1": f_score,
    }
    

def load_pkl_file(folder):
    # Search for .pkl files in the specified folder
    pkl_files = glob.glob(os.path.join(folder, '*.pkl'))

    # Load the first .pkl file found with the rb (read binary) mode
    if len(pkl_files) > 0:
        with open(pkl_files[0], 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"No .pkl file found in folder={folder}")

    return model

    
def load_yaml(filename):
    print(f"Loading config from {filename}.")
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    return data

def generate_args_hashid(args, keys=None):
    # Combine the args and data dictionaries into a single dictionary
    if keys is not None:
        combined_dict = {key: vars(args)[key] for key in keys}
    else:
        combined_dict = vars(args)

    # Serialize the dictionary as a JSON string
    json_str = json.dumps(combined_dict, sort_keys=True)

    # Generate a SHA-256 hash of the JSON string
    hash_object = hashlib.sha256(json_str.encode())

    # Return the first eight characters of the hash as a hexadecimal string
    expname = hash_object.hexdigest()[:8]
    args.expname = expname
    return args


    #  {
    #     "logname": args.logname,
    #     "expname": args.expname,
    #     **result,
    #     "depth": args.depth,
    #     "st": args.st,
    #     "max_child": args.max_child,
    #     "lru_size": args.lru_size,
    #     "ask_threshold": args.ask_threshold,
    #     "topk": args.topk,
    #     "mode": args.mode,
    #     "train_file": args.train_file,
    #     "test_file": args.test_file,
    # }

import fcntl
def write_experiment_results(outfile, args, result):
    # Write the results to a CSV file
    if result is None:
        return
    row_to_save = vars(args)
    row_to_save.update(result)


    with open(outfile, "a", newline="") as csvfile:
        # Acquire a file write lock
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)

        fieldnames = list(row_to_save.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If this is the first row, write the header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write the row for this experiment
        writer.writerow(row_to_save)
        print(f"Experiment record saved to {outfile}.")

        # Release the file write lock
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

    # with open(outfile, "a", newline="") as csvfile:
    #     fieldnames = list(row_to_save.keys())
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     # If this is the first row, write the header
    #     if csvfile.tell() == 0:
    #         writer.writeheader()

    #     # Write the row for this experiment
    #     writer.writerow(row_to_save)
    #     print(f"Experiment record saved to {outfile}.")
