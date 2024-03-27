import argparse
import asyncio
import json
import os
import random
import glob
import yaml
import vllm
import copy
import re
from transformers.trainer_utils import set_seed

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datasets import load_dataset, concatenate_datasets
import openai
from evaluate import load

import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# openai.api_key = "EMPTY"
# openai.api_base = "http://localhost:8000/v1"
set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_model_name", type=str, default=None)
    parser.add_argument("--generation_model_name", type=str, default=None)
    parser.add_argument("--feedback_model_port", type=int, default=None)
    parser.add_argument("--generation_model_port", type=int, default=None)
    parser.add_argument("--data_name", type=str, default="bigcode/humanevalpack")
    parser.add_argument("--prompt", type=str, default="vllm/feedback_inference_test.yaml")
    parser.add_argument("--prompt_key", type=str, default="base")
    parser.add_argument("--feedback_or_code", type=str, default="feedback", choices=["feedback", "code"])
    parser.add_argument("--use_feedback", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--do_iterate", default="Yes", choices=["Yes", "No"])
    parser.add_argument("--iterate_num", type=int, default=5, help="number of iteration")
    parser.add_argument("--seed_json", type=str, default="No", help="use saved seed json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num_sample", type=int, default=3, help="number of samples to generate")
    parser.add_argument("--n", type=int, default=5, help="number of samples to generate in feedback")

    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--limit", type=int)
    # parser.add_argument("--min_context_len", type=int, default=0)
    return parser.parse_args()


def load_data(data_name, split=None):
    data = load_dataset(data_name)
    print("=========== dataset statistics ===========")
    print(len(data[split]))
    print("==========================================")
    if args.limit:
        return [data[split][i] for i in range(args.limit)]
    return data[split]


def load_json(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def save_json(save_dir, results):
    with open(save_dir, "w", encoding="UTF-8") as f:
        json.dump(results, f, indent=4)


def load_prompt(prompt_path):
    with open(args.prompt, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)[args.prompt_key]
    return prompt


def split_function_header_and_docstring(s):
    # pattern = re.compile(r'\"\"\"(.*?)\"\"\"', re.DOTALL)
    pattern = re.compile(r"(\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\')", re.DOTALL)
    match = pattern.findall(s)
    if match:
        # docstring = match.group(-1)
        docstring = match[-1][0]
        code_without_docstring = s.replace(docstring, "").replace('"' * 6, "").strip()
        docstring = docstring.replace('"', "")
    else:
        raise ValueError
    return code_without_docstring, docstring


def prepare_model_input(prompt, code_data):
    description = code_data["prompt"]
    function_header, docstring = split_function_header_and_docstring(description)
    problem = docstring.split(">>>")[0]

    wrong_code = function_header + code_data["buggy_solution"]
    template_dict = {"function_header": function_header, "description": problem, "wrong_code": wrong_code}
    model_input = prompt.format(**template_dict)
    return model_input, problem, function_header


def load_and_prepare_data(args):
    dataset = load_data(args.data_name, args.split)
    prompt = load_prompt(args.prompt)
    all_model_inputs = []
    print("### load and prepare data")
    for data in tqdm(dataset):
        model_input, problem, function_header = prepare_model_input(prompt, data)
        if args.use_feedback == "Yes":
            new_model_input = model_input
        else:
            new_model_input = f"{model_input}\nCorrect code:\n{function_header}"
        data["header"] = function_header
        all_model_inputs.append([new_model_input, data])
    return all_model_inputs


async def async_generate(llm, model_input, idx, save_dir, feedback_flag):
    if feedback_flag:
        while True:
            try:
                response = await llm.agenerate(prompts=[model_input[0]])  # if you need it
                # print("Completion result:", completion)
                break
            except Exception as e:
                print(f"Exception occurred: {e}")
                # response = None
                # return None
        result_list = []
        for response_text in response.generations[0]:
            result = {
                "prediction": response_text.text,
                "description": model_input[1]["prompt"],
                "wrong_code": model_input[1]["buggy_solution"],
                "correct_code": model_input[1]["canonical_solution"],
                "model_input": model_input[0],
                **model_input[1],
            }
            result_list.append(result)
        return result_list
    else:
        while True:
            try:
                response = await llm.agenerate(prompts=[model_input[0]])  # if you need it
                # print("Completion result:", completion)
                break
            except Exception as e:
                print(f"Exception occurred: {e}")
                # response = None
                # return None

        result = {
            "prediction": response.generations[0][0].text,
            "description": model_input[1]["prompt"],
            "wrong_code": model_input[1]["buggy_solution"],
            "correct_code": model_input[1]["canonical_solution"],
            "model_input": model_input[0],
            **model_input[1],
        }

        return result


async def generate_concurrently(all_model_input, start_idx, stop, args, feedback_flag):
    if feedback_flag:  ## if feedback model
        llm = OpenAI(
            model_name=args.feedback_model_name,
            openai_api_base=f"http://localhost:{args.feedback_model_port}/v1",
            openai_api_key="EMPTY",
            max_tokens=128,
            top_p=0.95,
            temperature=0.5,
            frequency_penalty=0.4,
            stop=stop,
            n=args.n,
        )
    else:
        llm = OpenAI(
            model_name=args.generation_model_name,
            openai_api_base=f"http://localhost:{args.generation_model_port}/v1",
            openai_api_key="EMPTY",
            max_tokens=1024,
            top_p=0.95,
            temperature=0.1,
            frequency_penalty=0.0,
            stop=stop,
        )
    tasks = [
        async_generate(llm, model_input, i + start_idx, args.save_dir, feedback_flag)
        for i, model_input in enumerate(all_model_input)
    ]
    return await tqdm_asyncio.gather(*tasks)


def evaluation(prediction_result, args):
    def reconsruct_preds(preds):
        reconstruct_preds = []
        for data in preds:
            header = data["header"]
            prediction = data["prediction"]
            if "Correct code:" in prediction:
                prediction = prediction.split("Correct code:")[1]
            reconstruct_preds.append([header + prediction])
        return reconstruct_preds

    reconstruct_prediction = reconsruct_preds(prediction_result)
    humaneval_fix = load_dataset("bigcode/humanevalpack")["test"]
    code_metric = load("Muennighoff/code_eval_octopack")
    language = "python"
    timeout = 1
    if args.limit:
        references = [humaneval_fix[i]["test"] for i in range(args.limit)]
    else:
        references = [d["test"] for d in humaneval_fix]

    results, logs = code_metric.compute(
        references=references,
        predictions=reconstruct_prediction,
        language=language,
        timeout=timeout,
        num_workers=4,
    )
    assert len(reconstruct_prediction) == len(logs)
    return_dict = dict()
    return_dict["pass@1"] = results["pass@1"]
    current_prediction_list = list()
    for i in range(len(logs)):
        copy_prediction = copy.deepcopy(prediction_result[i])
        copy_prediction["logs"] = logs[i][0][1]

        current_prediction_list.append(copy_prediction)
    return_dict["result"] = current_prediction_list
    return return_dict


def evaluation_wrongonly(previous_evaluate_result, prediction_result, args):
    def reconsruct_preds(preds):
        reconstruct_preds = []
        for data in preds:
            header = data["header"]
            prediction = data["prediction"]
            if "Correct code:" in prediction:
                prediction = prediction.split("Correct code:")[1]
            reconstruct_preds.append([header + prediction])
        return reconstruct_preds

    reconstruct_prediction = reconsruct_preds(prediction_result)
    humaneval_fix = load_dataset("bigcode/humanevalpack")["test"]
    code_metric = load("Muennighoff/code_eval_octopack")
    language = "python"
    timeout = 1
    if args.limit:
        references = [humaneval_fix[i]["test"] for i in range(args.limit)]
    else:
        references = [d["test"] for d in humaneval_fix]

    results, logs = code_metric.compute(
        references=references,
        predictions=reconstruct_prediction,
        language=language,
        timeout=timeout,
        num_workers=4,
    )
    assert len(reconstruct_prediction) == len(logs)
    return_dict = dict()
    current_prediction_list = list()
    for i in range(len(logs)):
        copy_prediction = copy.deepcopy(prediction_result[i])
        copy_prediction["logs"] = logs[i][0][1]

        current_prediction_list.append(copy_prediction)
    previous_result_list = previous_evaluate_result["result"]
    assert len(previous_result_list) == len(current_prediction_list)

    sum_list = list()
    for i in range(len(previous_result_list)):
        previous_passed = previous_result_list[i]["logs"]["passed"]
        if previous_passed:
            sum_list.append(previous_result_list[i])
        else:
            sum_list.append(current_prediction_list[i])
    counter = 0
    for i in range(len(sum_list)):
        passed = sum_list[i]["logs"]["passed"]
        if passed:
            counter += 1
    pass_1 = float(counter / len(sum_list))

    return_dict["pass@1"] = pass_1
    return_dict["result"] = sum_list
    return return_dict


def evaluation_public_wrongonly(previous_evaluate_result, prediction_result, args):
    def reconsruct_preds(preds):  ##reshape prediction to input for evaluation
        reconstruct_preds = []
        for data in preds:
            header = data["header"]
            prediction = data["prediction"]
            if "Correct code:" in prediction:
                prediction = prediction.split("Correct code:")[1]
            reconstruct_preds.append([header + prediction])
        return reconstruct_preds

    def reconstruct_prev_preds(prev_evaluate_result):  ##reshape previous prediction to input for evaluation
        reconstruct_prev = []
        prev_list = prev_evaluate_result["result"]
        for data in prev_list:
            header = data["header"]
            prediction = data["prediction"]
            if "Correct code:" in prediction:
                prediction = prediction.split("Correct code:")[1]
            reconstruct_prev.append([header + prediction])
        return reconstruct_prev

    reconstruct_prediction = reconsruct_preds(prediction_result)
    reconstruct_previous_prediction = reconstruct_prev_preds(previous_evaluate_result)
    ## load previous prediction & current prediction with function header
    humaneval_fix = load_dataset("bigcode/humanevalpack")["test"]
    code_metric = load("Muennighoff/code_eval_octopack")
    language = "python"
    timeout = 1

    public_test_case = [d["example_test"] for d in humaneval_fix]  ## public test case
    results_on_public, log_on_public = code_metric.compute(  ## evaluate current prediction on public test case
        references=public_test_case,
        predictions=reconstruct_prediction,
        language=language,
        timeout=timeout,
        num_workers=4,
    )
    current_prediction_list = list()
    for i in range(len(log_on_public)):
        copy_prediction = copy.deepcopy(prediction_result[i])
        copy_prediction["logs"] = log_on_public[i][0][1]

        current_prediction_list.append(
            copy_prediction
        )  ## add evaluation result of public test case for current prediction

    prev_result_on_public, prev_log_on_public = code_metric.compute(
        references=public_test_case,
        predictions=reconstruct_previous_prediction,
        language=language,
        timeout=timeout,
        num_workers=4,
    )
    assert len(log_on_public) == len(prev_log_on_public)
    prev_prediction_list = list()
    previous_result_list = previous_evaluate_result["result"]
    for i in range(len(prev_log_on_public)):
        copy_prediction = copy.deepcopy(previous_result_list[i])
        copy_prediction["logs"] = prev_log_on_public[i][0][1]
        assert (
            copy_prediction["logs"] is not previous_result_list[i]["logs"]
        )  ## make sure log is overwritten with new log
        prev_prediction_list.append(
            copy_prediction
        )  ## add evaluation result of public test case for previous prediction

    assert len(prev_prediction_list) == len(current_prediction_list)
    sum_list = list()
    for i in range(
        len(prev_prediction_list)
    ):  ## sum previous prediction and current prediction about public test case
        previous_public_case_passed = prev_prediction_list[i]["logs"]["passed"]
        if previous_public_case_passed:
            sum_list.append(prev_prediction_list[i])
        else:
            sum_list.append(current_prediction_list[i])

    if args.limit:
        references = [humaneval_fix[i]["test"] for i in range(args.limit)]
    else:
        references = [d["test"] for d in humaneval_fix]
    reconstruct_sum_list = reconsruct_preds(sum_list)
    ##reshape prev prediction + current prediction to input for evaluation
    sum_result, sum_logs = code_metric.compute(  ## evaluate for real test case.
        references=public_test_case,
        predictions=reconstruct_sum_list,
        language=language,
        timeout=timeout,
        num_workers=4,
    )
    sum_passed_list = []
    for i in range(len(sum_logs)):
        if sum_logs[i][0][1]["passed"]:
            sum_passed_list.append(i)

    final_results, final_logs = code_metric.compute(  ## evaluate for real test case.
        references=references,
        predictions=reconstruct_sum_list,
        language=language,
        timeout=timeout,
        num_workers=4,
    )

    hidden_passed_list = []
    for i in range(len(final_logs)):
        if final_logs[i][0][1]["passed"]:
            hidden_passed_list.append(i)

    return_dict = dict()
    return_dict["pass@1"] = final_results["pass@1"]
    return_dict["public_passed"] = sum_passed_list
    return_dict["hidden_passed"] = hidden_passed_list
    return_dict["result"] = sum_list

    return return_dict


def prepare_correction_with_none(all_model_inputs):
    all_correction_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        cur_feedback = "Feedback: None"
        cur_function_header = model_input_dict[1]["header"]
        new_model_input = f"{model_input}{cur_feedback}\nCorrect code:\n{cur_function_header}"
        model_input_dict[0] = new_model_input
        all_correction_model_inputs.append(model_input_dict)
    return all_correction_model_inputs


def prepare_correction_input(all_model_inputs, all_feedback_results):
    all_correction_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        cur_feedback = all_feedback_results[i]["prediction"]
        model_input_dict[1]["feedback"] = cur_feedback.split("\n")
        cur_function_header = model_input_dict[1]["header"]
        new_model_input = f"{model_input}{cur_feedback}\nCorrect code:\n{cur_function_header}"
        model_input_dict[0] = new_model_input
        all_correction_model_inputs.append(model_input_dict)

    return all_correction_model_inputs


def prepare_iteration_input(all_model_inputs, all_predict_results, args):
    all_iteration_model_inputs = []
    prompt = load_prompt(args.prompt_key)
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        cur_prediction = all_predict_results[i]["prediction"]
        if "Correct code:" in cur_prediction:
            cur_prediction = cur_prediction.split("Correct code:")[1]
        data = model_input_dict[1]
        data["buggy_solution"] = cur_prediction
        model_input, _, function_header = prepare_model_input(prompt, data)
        if args.use_feedback == "Yes":
            new_model_input = model_input
        else:
            new_model_input = f"{model_input}\nCorrect code:\n{function_header}"
        all_iteration_model_inputs.append([new_model_input, data])
    return all_iteration_model_inputs


def prepare_correction_with_none(all_model_inputs):
    all_correction_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        cur_feedback = "Feedback: None"
        cur_function_header = model_input_dict[1]["header"]
        new_model_input = f"{model_input}{cur_feedback}\nCorrect code:\n{cur_function_header}"
        model_input_dict[0] = new_model_input
        all_correction_model_inputs.append(model_input_dict)
    return all_correction_model_inputs


def reformat_data(data):
    target_keys_for_reformat = ["wrong_code", "correct_code", "prediction"]
    copy_data = copy.deepcopy(data)
    for di, d in enumerate(copy_data["result"]):
        for k, v in d.items():
            if k in target_keys_for_reformat:
                if type(v) is not str:
                    continue
                copy_data["result"][di][k] = v.strip("\n").split("\n")
    return copy_data


async def main(args):
    all_model_inputs = load_and_prepare_data(args)
    for i in range(args.num_sample):
        save_path = os.path.join(args.save_dir, f"{i+1}_sample")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ## NOTE: Output file of the Selector should be 'selected_feedback.json'
        selected_feedback_list = load_json(os.path.join(save_path, "selected_feedback.json"))
        # Correction
        all_correction_input = prepare_correction_input(all_model_inputs, selected_feedback_list)
        # generate code
        all_results = await generate_concurrently(all_correction_input, 0, None, args, False)
        eval_results = evaluation(all_results, args)
        with open(os.path.join(save_path, f"seed_try.json"), "w", encoding="UTF-8") as f:
            json.dump(eval_results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
    print("Done!")
