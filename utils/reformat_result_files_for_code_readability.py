import json
import os

import argparse

parser = argparse.ArgumentParser()

# add code for multiple files


parser.add_argument("-i", '--input_file', type=str, required=True)


args = parser.parse_args()

def reformat_data(data):
    target_keys_for_reformat = ["wrong_code", "correct_code", "prediction"]
    for di, d in enumerate(data['result']):
        for k,v in d.items():
            if k in target_keys_for_reformat:
                data['result'][di][k] = v.strip("\n").split("\n")
    return data

def write_data(data, path):
    with open(path.replace(".json", "_reformated.json"), 'w') as f:
        json.dump(data, f, indent=4)


if args.input_file is os.path.isfile(args.input_file):

    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    formated_data = reformat_data(data)

    write_data(formated_data, args.input_file)
else:
    input_files = [f for f in os.listdir(args.input_file) if f.endswith(".json")]
    for fn in input_files:
        with open(os.path.join(args.input_file, fn), 'r') as f:
            data = json.load(f)
        
        formated_data = reformat_data(data)
        write_data(formated_data, os.path.join(args.input_file, fn))

