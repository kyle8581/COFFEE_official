import sys
import io
import subprocess
import os
import time
import math
import json
from tqdm.auto import tqdm
from datasets import load_dataset
import copy
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--dict_dir", type=str, default="train_problem_success_test_case.json")
    parser.add_argument("--try_num", type=int, default=3)
    parser.add_argument("--n", type=int, default=6)
    # parser.add_argument("--min_context_len", type=int, default=0)
    return parser.parse_args()


def run_given_code(code, test_input, timeout):
    command = ["python", "-c", code]
    try:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate(input=test_input.encode(), timeout=timeout)
    except:
        p.kill()
        return -1
    while p.poll() is None:
        # Process hasn't exited yet, let's wait some
        time.sleep(0.5)
    if p.returncode != 0:
        return -1
    get_output_text = output.decode()

    return get_output_text


def get_timeout_value(dict_data):
    sorted_items = sorted(dict_data.items(), key=lambda x: x[0])
    return math.ceil(float(sorted_items[2][1].split("초")[0].strip()))


def main(args):
    make_final_dict = {}
    with open(args.dict_dir, "r") as f:
        problem_dict = json.load(f)
    for i in range(args.try_num):
        folder_dir = os.path.join(args.json_dir, f"{i+1}_sample")
        for j in tqdm(range(args.n)):
            get_file_dir = os.path.join(folder_dir, f"seed_try_{j}.json")
            with open(get_file_dir, "r") as f:
                load_json = json.load(f)
            for data in load_json:
                problem_id = data["problem_id"]
                wrong_code = data["wrong_code"]
                predicted_code = data["prediction"]
                if wrong_code not in make_final_dict:
                    make_final_dict[wrong_code] = []
                feedback = data["feedback"]
                timeout = get_timeout_value(data["metadata"])
                if problem_id in problem_dict:
                    test_cases = problem_dict[problem_id]["success_test_case"]
                    for test_case in test_cases:
                        test_input, test_output = test_case
                        get_output = run_given_code(predicted_code, test_input, timeout)
                        if get_output == test_output:
                            value = (feedback, True)
                            print("correct!")
                            make_final_dict[wrong_code].append(value)
                        else:
                            value = (feedback, False)
                            make_final_dict[wrong_code].append(value)
    with open("final_dict.json", "w") as f:
        json.dump(make_final_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
