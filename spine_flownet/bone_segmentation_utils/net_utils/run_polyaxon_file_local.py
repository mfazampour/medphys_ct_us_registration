import yaml
from pathlib import Path
import itertools
import subprocess
import time
import os


config_path = Path("modules/mstcn/config/polyaxon_check_code_tcn.yml")


with open(config_path) as file:
    cfg = yaml.full_load(file)

declarations = cfg["declarations"]

# override some for local run
declarations["data_root"] = "/mnt/polyaxon/data1"
declarations["pickle_feature_path"] = "/mnt/polyaxon/outputs1/tobiascz/tecno/groups/436/8956/cholec80_pickle_export"
# declarations["num_validation_chunks"] = 3

arg_list=[]

### IF config in polyaxon file with -c option ###
config_string = None
if "config_file" in declarations.keys():
    config_string = f"-c {declarations['config_file']} "
    del declarations["config_file"]

### Adding all declarations to arg_list ###
for i,x in declarations.items():
    arg_list.append([f"--{i}={x}"])

### HP TUNING PART ###
if "hptuning" in cfg.keys():
    if "matrix" in cfg["hptuning"]:
        matrix = cfg["hptuning"]["matrix"]
        for i,x in matrix.items():
            current_list=[]
            if not "--" in str(x["values"][0]):
                for a in x["values"]:
                    current_list.append(f"--{i}={a}")
            else:
                current_list = x["values"]
            arg_list.append(current_list)

all_combinations = list(itertools.product(*arg_list))
all_combinations = [" ".join(x) for x in all_combinations]

if config_string is not None:
    all_combinations = [config_string + x for x in all_combinations]

delay = 5
print(f"Polyaxon file: {config_path}\nnumber of experiments to run: {len(all_combinations)}\nStarting in {delay}s ...")
time.sleep(delay)

for run_config_string in all_combinations:
    run_list = run_config_string.split( )
    run_list.insert(0,"/home/tobiascz/miniconda3/envs/torch/bin/python")
    run_list.insert(1,"train.py")
    print(run_list)
    subprocess.call(run_list)
