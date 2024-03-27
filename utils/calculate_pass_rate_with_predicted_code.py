import json
import os
import argparse
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_case_file",
    type=str,
    default=None,
)
parser.add_argument(
    "--inference_file",
    type=str,
    default=None,
)


def run_code(code, input_data):
    process = subprocess.Popen(
        ["python", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Join inputs with newline and encode
    # input_data = input_data.replace("\r\n", "\n")
    input_data = input_data.encode()

    try:
        stdout_data, stderr_data = process.communicate(input=input_data, timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        print("The subprocess exceeded the time limit and was killed.")
        return None

    if process.returncode != 0:
        # There was an error
        print(code)
        print(input_data)
        print(Exception(f"Error executing code:\n{stderr_data.decode('utf-8')}"))
        return None

    return stdout_data.decode("utf-8")


def compare_output(predictions, references):
    # Handle None values and perform string operations
    predictions = [p.replace("\r\n", "\n").strip("\n") if p is not None else None for p in predictions]
    references = [r.replace("\r\n", "\n").strip("\n") if r is not None else None for r in references]

    return [p == r for p, r in zip(predictions, references)]


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.test_case_file, "r") as f:
        test_cases = json.load(f)

    with open(args.inference_file, "r") as f:
        inference_cases = json.load(f)

    avg_scores = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for di, d in enumerate(tqdm(inference_cases)):
            cur_problem_id = d["problem_id"]
            if cur_problem_id not in test_cases:
                continue
            cur_testcase = test_cases[cur_problem_id]["success_test_case"]
            cur_testcase_inputs = [t[0] for t in cur_testcase]
            cur_testcase_outputs = [t[1] for t in cur_testcase]
            cur_refined_code = d["prediction"]

            # Execute test cases concurrently for the current code
            cur_outputs = list(
                executor.map(run_code, [cur_refined_code] * len(cur_testcase_inputs), cur_testcase_inputs)
            )

            cur_testcase_pass_results = compare_output(cur_outputs, cur_testcase_outputs)
            inference_cases[di]["testcase_pass_results"] = cur_testcase_pass_results
            if len(cur_testcase_pass_results) != 0:
                avg_scores.append(sum(cur_testcase_pass_results) / len(cur_testcase_pass_results))
            else:
                avg_scores.append(0)
            print(sum(avg_scores) / len(avg_scores))

    with open(args.inference_file.replace(".json", "_with_testcase_results.json"), "w") as f:
        json.dump(inference_cases, f, indent=4)
