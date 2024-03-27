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

import sys

from async_function import *
from evaluate_function import *

import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# openai.api_key = "EMPTY"
# openai.api_base = "http://localhost:8000/v1"
set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_model_name", type=str, default=None)
    parser.add_argument("--generation_model_name", type=str, required=True)
    parser.add_argument("--feedback_model_port", type=int, default=None)
    parser.add_argument("--generation_model_port", type=int, required=True)
    parser.add_argument("--data_name", type=str, default="bigcode/humanevalpack")
    parser.add_argument("--prompt", type=str, default="vllm/feedback_inference_test.yaml")
    parser.add_argument("--prompt_key", type=str, default="base")
    parser.add_argument("--select_key", type=str, default="chatgpt_select")
    parser.add_argument("--iterative_strategy", default="wrongonly", choices=["wrongonly", "public_wrongonly"])
    parser.add_argument("--use_feedback", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--do_iterate", default="Yes", choices=["Yes", "No"])
    parser.add_argument("--iterate_num", type=int, default=5, help="number of iteration")
    parser.add_argument("--seed_json", type=str, default="No", help="use saved seed json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num_candidates", type=int, default=5, help="number of candidates to generate")
    parser.add_argument("--num_sample", type=int, default=3, help="number of samples to generate")

    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--limit", type=int)
    # parser.add_argument("--min_context_len", type=int, default=0)
    return parser.parse_args()


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


def load_data(data_name, split=None):
    data = load_dataset(data_name)
    print("=========== dataset statistics ===========")
    print(len(data[split]))
    print("==========================================")
    if args.limit:
        return [data[split][i] for i in range(args.limit)]
    return data[split]


def load_prompt(prompt_key):
    with open(args.prompt, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)[prompt_key]
    return prompt


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
    prompt = load_prompt(args.prompt_key)
    all_model_inputs = []
    print("### load and prepare data")
    for data in tqdm(dataset):
        model_input, problem, function_header = prepare_model_input(prompt, data)
        if args.use_feedback == "Yes":
            new_model_input = model_input
        else:
            new_model_input = f"{model_input}\nCorrect code:\n{function_header}"
        data["header"] = function_header
        data["problem"] = problem
        all_model_inputs.append([new_model_input, data])
    return all_model_inputs


def select_feedback(all_model_inputs, all_feedback_results, args):
    prompt = load_prompt(args.select_key)
    make_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        data = all_model_inputs[i][1]
        input_string = ""
        get_feedback_list = all_feedback_results[i]["feedback_list"]
        data["feedback_list"] = get_feedback_list
        for index, feedback in enumerate(get_feedback_list):
            input_string += f"{index}. {feedback}\n"
        wrong_code = data["header"] + data["buggy_solution"]
        template_dict = {"wrong_code": wrong_code, "description": data["problem"], "feedback": input_string}
        model_input = prompt.format(**template_dict)
        make_model_inputs.append([model_input, data])

    return make_model_inputs


def prepare_correction_input(all_model_inputs, all_feedback_results):
    all_correction_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        data = all_model_inputs[i][1]
        cur_feedback = all_feedback_results[i]["prediction"]
        data["selected feedback"] = cur_feedback
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
        data["buggy_solution"] = cur_prediction  ## use prediction for next wrong code
        model_input, _, function_header = prepare_model_input(prompt, data)
        if args.use_feedback == "Yes":
            new_model_input = model_input
        else:
            new_model_input = f"{model_input}\nCorrect code:\n{function_header}"
        all_iteration_model_inputs.append([new_model_input, data])
    return all_iteration_model_inputs


async def main(args):
    all_model_inputs = load_and_prepare_data(args)
    for i in range(args.num_sample):
        save_path = os.path.join(args.save_dir, f"{i+1}_sample")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ## use seed json or not
        if args.seed_json == "No":
            print("seed json is not defined. generate seed json")
        else:
            print(f"seed json is defined. use {args.seed_json}")

        if args.use_feedback == "Yes":
            #### if use feedback but feedback model is not defined ####
            if args.feedback_model_name is None:
                args.feedback_model_name = args.generation_model_name
            if args.feedback_model_port is None:
                args.feedback_model_port = args.generation_model_port

            ## generate feedback
            print("generate feedback")
            feedback_list = await generate_feedback(all_model_inputs, 0, None, args, True)

            ## select ready & sfeedback
            print("select feedback")
            selected_model_inputs = select_feedback(all_model_inputs, feedback_list, args)
            all_feedback_results = await generate_select(selected_model_inputs, 0, None, args, None)

            ## generate code
            print("generate code")
            all_correction_input = prepare_correction_input(all_model_inputs, all_feedback_results)
            all_results = await generate_code(all_correction_input, 0, None, args, False)
        else:
            ## only generate code
            all_results = await generate_code(all_model_inputs, 0, None, args, False)

        ## use seed json or not
        if args.seed_json == "No":
            eval_results = evaluation(all_results, args)
            wrongonly_eval = copy.deepcopy(eval_results)
            publiconly_eval = copy.deepcopy(eval_results)
            reformated_eval_results = reformat_data(eval_results)
            with open(os.path.join(save_path, "seed_try.json"), "w", encoding="UTF-8") as f:
                json.dump(reformated_eval_results, f, indent=4)
        else:
            with open(args.seed_json, "r", encoding="UTF-8") as f:
                eval_results = json.load(f)
            wrongonly_eval = copy.deepcopy(eval_results)
            publiconly_eval = copy.deepcopy(eval_results)

        ## if iterate
        if args.do_iterate == "Yes":
            for i in range(args.iterate_num):
                ## ready for iteration
                iterative_model_inputs = prepare_iteration_input(all_model_inputs, all_results, args)
                if args.use_feedback == "Yes":
                    ## generate feedback
                    iterative_feedback_list = await generate_feedback(iterative_model_inputs, 0, None, args, True)

                    ## select ready & feedback
                    iterative_selected_model_inputs = select_feedback(
                        iterative_model_inputs, iterative_feedback_list, args
                    )
                    iterative_feedback_results = await generate_select(
                        iterative_selected_model_inputs, 0, None, args, None
                    )

                    iterative_correction_inputs = prepare_correction_input(
                        iterative_model_inputs, iterative_feedback_results
                    )
                    all_results = await generate_code(iterative_correction_inputs, 0, None, args, False)
                else:
                    all_results = await generate_code(iterative_model_inputs, 0, None, args, False)
                eval_results = evaluation(all_results, args)
                reformatted_eval_results = reformat_data(eval_results)

                with open(os.path.join(save_path, f"{i+1}_fixall.json"), "w", encoding="UTF-8") as f:
                    json.dump(reformatted_eval_results, f, indent=4)
                wrongonly_eval = evaluation_wrongonly(wrongonly_eval, all_results, args)
                reformatted_wrong_only_results = reformat_data(wrongonly_eval)
                with open(os.path.join(save_path, f"{i+1}_wrongonly.json"), "w", encoding="UTF-8") as f:
                    json.dump(reformatted_wrong_only_results, f, indent=4)
                publiconly_eval = evaluation_public_wrongonly(publiconly_eval, all_results, args)
                reformatted_public_only_eval = reformat_data(publiconly_eval)
                with open(os.path.join(save_path, f"{i+1}_public_wrongonly.json"), "w", encoding="UTF-8") as f:
                    json.dump(reformatted_public_only_eval, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
    print("Done!")
