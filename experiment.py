import json
from mimetypes import init
import os
import argparse
from main import main
from exp_setter import *



parser = argparse.ArgumentParser(description='DG Benchmark in FL setting')
parser.add_argument('--algorithm', default="ERM")
parser.add_argument('--exp_name', help='exp name')
parser.add_argument('--refresh_config', default=True, type=bool) #TODO: implement when available
args = parser.parse_args()


baseline_filename = f"./config/{args.algorithm}/config_baseline.json"
exp_filepath = f"./config/{args.algorithm}/{args.exp_name}/"

if args.exp_name == "test":
    print(f"Test:\n")
    os.system(f"python main.py --config_file {baseline_filename}")

if not os.path.exists(exp_filepath):
    exp_set = eval(args.exp_name)(args.algorithm)
    exp_set.set()


print(f"Experiment {args.exp_name} for {args.algorithm}:\n")
for f in os.listdir(exp_filepath):
    filename = os.path.join(exp_filepath, f)
    if f.endswith('.json') and os.path.isfile(filename):
        print(f"python main.py --config_file {filename}")
        os.system(f"python main.py --config_file {filename}")
