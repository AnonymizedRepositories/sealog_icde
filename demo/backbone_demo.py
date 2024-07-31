#!/usr/bin/env python
import os
import re
import sys

sys.path.append("../")
from common.utils import write_experiment_results, generate_args_hashid
from sealog.backbone import Backbone
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")


# python backbone_demo.py --logname Industry --folder Industry

# for data selection
parser.add_argument("--logname", default="Industry", type=str, help="the name of the log")
parser.add_argument(
    "--folder",
    default="Industry",
    type=str,
    help="the name of the exp",
)
parser.add_argument("--train_file", default="train.pkl", type=str, help="file to train")
parser.add_argument("--test_file", default="test.pkl", type=str, help="file to test")
parser.add_argument("--expname", default="", type=str, help="expname")
parser.add_argument("--window_size", default=1, type=int, help="window size")
parser.add_argument("--stride", default=1, type=int, help="stride size")


args = parser.parse_args()
if not args.expname:
    args = generate_args_hashid(args)

logname = args.logname
indir = f"../proceeded_data/{logname}/{args.folder}"
outdir = f"scalead_result/{args.folder}"  # The output directory of parsing results
regex = {
    "BGL": [r"core\.\d+", r"(\d+\.){3}\d+", r"\b(0x)?[0-9a-fA-F]+\b", r"\d+(?:\.\d+)?"],
    "Thunderbird": [
        r"core\.\d+",
        r"(\d+\.){3}\d+",
        r"\b(0x)?[0-9a-fA-F]+\b",
        r"\d+(?:\.\d+)?",
        r"\/[a-zA-Z0-9_:\-\.\/]+", # filepath    
    ],
    "Industry": []
}

if __name__ == "__main__":
    backbone = Backbone(args, indir, outdir, regex[logname], icl_topk=1)

    backbone.initialize_icl_model(train_file=args.train_file)
    
    backbone.detect(args.test_file, num_workers=20)