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
    parser.add_argument("--generation_model_name", type=str, required=True)
    parser.add_argument("--feedback_model_port", type=int, default=None)
    parser.add_argument("--generation_model_port", type=int, required=True)
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="vllm/feedback_inference_test.yaml")
    parser.add_argument("--prompt_key", type=str, default="base")
    parser.add_argument("--use_feedback", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval", "test"])
    parser.add_argument("--num_try", type=int, default=3, help="number of samples to generate")
    parser.add_argument("--n", type=int, default=5, help="number of samples to generate in feedback")
    parser.add_argument("--test_sample", type=int, default=750, help="number of samples to generate in test")

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
    model_input = prompt.format(**code_data)
    return model_input


def load_and_prepare_data(args):
    dataset = load_data(args.data_name, args.split)
    prompt = load_prompt(args.prompt)
    all_model_inputs = []
    print("### load and prepare data")
    for data in tqdm(dataset):
        model_input = prepare_model_input(prompt, data)
        if args.use_feedback == "Yes":
            new_model_input = model_input
        else:
            new_model_input = f"{model_input}\nCorrect code:\n"
        all_model_inputs.append([new_model_input, data])
    return all_model_inputs


def sample_indices(all_model_inputs, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_inputs)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices


def sample_list(all_model_inputs, num_sample):
    sampled_indicies = []
    if num_sample is None:
        print("num sample is None. sampling all data")
        return all_model_inputs
    else:
        sampled_indices = sample_indices(all_model_inputs, num_sample)
        sampled_list = []
        for i in sampled_indices:
            copy_data = copy.deepcopy(all_model_inputs[i])
            sampled_list.append(copy_data)
        return sampled_list


def filter_data(all_model_inputs, num_sample):
    sampled_list = sample_list(all_model_inputs, num_sample)
    return sampled_list


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


def prepare_correction_with_none(all_model_inputs):
    all_correction_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        cur_feedback = "Feedback: None"
        model_input_dict[1]["feedback"] = cur_feedback
        new_model_input = f"{model_input}{cur_feedback}\nCorrect code:\n"
        model_input_dict[0] = new_model_input
        all_correction_model_inputs.append(model_input_dict)
    return all_correction_model_inputs


def prepare_correction_input(all_model_inputs, all_feedback_results):
    all_correction_model_inputs = []
    for i in range(len(all_model_inputs)):
        model_input_dict = copy.deepcopy(all_model_inputs[i])
        model_input = model_input_dict[0]
        cur_feedback = all_feedback_results[i]["prediction"]
        model_input_dict[1]["feedback"] = cur_feedback
        new_model_input = f"{model_input}{cur_feedback}\nCorrect code:\n"
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
    for i in range(args.num_try):
        save_path = os.path.join(args.save_dir, f"{i+1}_sample")
        temp_path = os.path.join(args.save_dir, f"temp_{i+1}_sample")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        print("generate all feedback")
        all_results = [[] for _ in range(args.n)]
        temporary_index = 0
        for start_idx in tqdm(range(0, len(all_model_inputs), 300)):
            temporary_index += 1
            if len(all_model_inputs) < 300:
                cur_model_inputs = all_model_inputs
            else:
                if len(all_model_inputs) - start_idx < 300:
                    cur_model_inputs = all_model_inputs[start_idx:]
                else:
                    cur_model_inputs = all_model_inputs[start_idx : start_idx + 300]
            cur_feedback_result_list = await generate_concurrently(
                cur_model_inputs, 0, None, args, True
            )  ## generate feedback
            cur_feedback_list = []
            for i in range(args.n):
                temp_list = []
                for feedback_result in cur_feedback_result_list:
                    temp_list.append(feedback_result[i])
                cur_feedback_list.append(temp_list)
            for i in range(args.n):
                cur_correction_input = prepare_correction_input(cur_model_inputs, cur_feedback_list[i])  # Correction
                current_result = await generate_concurrently(
                    cur_correction_input, 0, None, args, False
                )  # generate code
                all_results[i].extend(current_result)
                with open(
                    os.path.join(temp_path, f"temp_seed_try_{i}_{temporary_index}_index.json"), "w", encoding="UTF-8"
                ) as f:
                    json.dump(current_result, f, indent=4)
        for i in range(args.n):
            with open(os.path.join(save_path, f"seed_try_{i}.json"), "w", encoding="UTF-8") as f:
                json.dump(all_results[i], f, indent=4)
        none_result = []
        temporary_index = 0
        for start_idx in tqdm(range(0, len(all_model_inputs), 300)):
            temporary_index += 1
            if len(all_model_inputs) < 300:
                cur_model_inputs = all_model_inputs
            else:
                if len(all_model_inputs) - start_idx < 300:
                    cur_model_inputs = all_model_inputs[start_idx:]
                else:
                    cur_model_inputs = all_model_inputs[start_idx : start_idx + 300]
            cur_correction_input = prepare_correction_with_none(cur_model_inputs)
            current_result = await generate_concurrently(cur_correction_input, 0, None, args, False)
            none_result.extend(current_result)
            with open(
                os.path.join(temp_path, f"temp_seed_try_none_{temporary_index}_index.json"), "w", encoding="UTF-8"
            ) as f:
                json.dump(current_result, f, indent=4)
        with open(os.path.join(save_path, "seed_try_none.json"), "w", encoding="UTF-8") as f:
            json.dump(none_result, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
    print("Done!")
