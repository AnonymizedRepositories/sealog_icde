#!/usr/bin/env python
import re
import sys

sys.path.append("../")
from common.utils import write_experiment_results, generate_args_hashid
from sealog.agent import ScaleAD
import argparse


# python agent_demo.py --logname Industry --folder Industry
# python agent_demo.py --logname Industry --folder Industry --llm_cache ./scalead_result/Industry/llm_cache.json

parser = argparse.ArgumentParser(description="Process some integers.")
 
parser.add_argument("--logname", default="Industry", type=str, help="the name of the log")
parser.add_argument(
    "--folder",
    default="Industry",
    type=str,
    help="the name of the exp",
)
parser.add_argument("--train_file", default="train.pkl", type=str, help="file to train")
parser.add_argument("--test_file", default="test.pkl", type=str, help="file to test")


parser.add_argument("--load_model", default="", type=str, help="prefix tree to load")
parser.add_argument("--model_save_path", default="", type=str, help="manually define where to save the model.")
parser.add_argument("--test_only", action="store_true", help="whether to test only")


parser.add_argument("--max_child", default=2, type=int, help="the max_child parameter")
parser.add_argument("--topk", default=3, type=int, help="the topk parameter")
parser.add_argument("--expname", default="", type=str, help="expname")
parser.add_argument("--llm_cache", default="", type=str, help="llm_cache")


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
    "Industry": [
    ],
}

hash_heads = {
    "BGL": ["Level"],
    "Thunderbird": ["Level"],
    "Industry": [],
}

if __name__ == "__main__":
    scale_ad = ScaleAD(
        args,
        indir=indir,
        outdir=outdir,
        rex=regex[logname],
        hash_heads=hash_heads[logname],
        max_child=args.max_child,
        topk=args.topk,
    )

    scale_ad.warmup(
                log_file=args.train_file,
                save_model=True,
            )  # thred=0

    
    if args.llm_cache:
        scale_ad.update_with_llm_cache(args.llm_cache)

    scale_ad.detect(
        log_file=args.test_file,
        cached_model_folder=None,
        evaluation=True,
        save_model=False,
        update_bayes=False)

