import argparse
import asyncio
import json
import os
import glob
import yaml
import re
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datasets import load_dataset, concatenate_datasets
import tiktoken
import copy
import time
import math
import openai


lock = asyncio.Lock()


MAPPING_LEVEL_TO_INT = {
    "bronze": 1,
    "silver": 6,
    "gold": 11,
    "platinum": 16,
    "diamond": 21,
}
MODEL_MAX_LENGTH = 4050
TARGET_MAX_LENGTH = 500

collected_list = []

TOTAL_COST = 0  # making this a global variable, be aware this may lead to issues in concurrent scenarios


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4", "llama2"],
    )
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="./prompt/feedback_annotation.yaml")
    parser.add_argument("--prompt_key", type=str, default="revised_short_feedback_annotation")
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=[None, "bronze", "silver", "gold", "platinum", "diamond"],
    )
    parser.add_argument(
        "--level_data_path",
        type=str,
        default="problem_num/problem_num_by_level.json",
        help="If --level is not None, it must be a path of problem_num_by_level.json",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Use this argument to specify the directory where you want to save the results.",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=None,
        help="If you want to test your code by sampling a small number of data, you can set this argument.",
    )
    parser.add_argument("--annotation_start_index", type=int, default=None, help="for splitting annotation")
    parser.add_argument("--annotation_end_index", type=int, default=None, help="for splitting annotation")
    # parser.add_argument("--min_context_len", type=int, default=0)
    return parser.parse_args()


def load_data(data_name, split=None):
    data = load_dataset(data_name)
    print("=========== dataset statistics ===========")
    print("total: ", len(data[split]))
    # print("sample: ", len(data["sample"]))
    print("==========================================")
    ## you must specify split!!
    return data[split]


def load_json(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def load_prompt(args):
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
    template_dict = {"description": problem, "wrong_code": wrong_code}
    model_input = prompt.format(**template_dict)
    return model_input


def count_total_data(args):
    dataset = load_data(args.data_name, args.split)
    prompt = load_prompt(args)
    all_model_inputs = []
    for data in tqdm(dataset):
        model_input = prepare_model_input(prompt, data)
        all_model_inputs.append([model_input, data])
    num_sample = count_sample(all_model_inputs)
    print(f"total available dataset: {num_sample}")


def load_and_prepare_data(args):
    dataset = load_data(args.data_name, args.split)
    prompt = load_prompt(args)
    all_model_inputs = []
    print("### load and prepare data")
    for data in tqdm(dataset):
        ## level args가 주어졌을 때 level에 포함되어 있는 문제가 아닐 경우 필터링
        if args.level:
            mapped_level = MAPPING_LEVEL_TO_INT[args.level]
            level_data = load_json(args.level_data_path)
            ## NOTE: sum([[...], [...], ...], []) -> 2중 리스트 flatten
            if data["problem_id"] not in sum(
                [nums for i, nums in level_data.items() if int(i) >= mapped_level and int(i) < mapped_level + 5],
                [],
            ):
                continue
        ##

        model_input = prepare_model_input(prompt, data)
        all_model_inputs.append([model_input, data])
    print(f"loaded {args.level} data length: {len(all_model_inputs)}")
    return all_model_inputs


def count_sample(all_model_inputs):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    counter = 0
    for i in tqdm(range(len(all_model_inputs))):
        ##checking source length
        get_tokenized_length = len(tokenizer.encode(all_model_inputs[i][0]))
        if get_tokenized_length <= MODEL_MAX_LENGTH - TARGET_MAX_LENGTH:
            counter += 1
    return counter


def sample_list(all_model_inputs, num_sample):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    sampled_indicies = []
    if num_sample is None:
        print("num sample is None. sampling all data")
        # for i in range(len(all_model_inputs)):
        #     ##checking source length
        #     get_tokenized_length = len(tokenizer.encode(all_model_inputs[i][0]))
        #     if get_tokenized_length <= MODEL_MAX_LENGTH - TARGET_MAX_LENGTH:
        #         sampled_indicies.append(i)
        # sampled_list = [all_model_inputs[i] for i in sampled_indicies]
        # return sampled_list
        return all_model_inputs
    else:
        counter = 0
        for i in range(len(all_model_inputs)):
            ##checking source length
            get_tokenized_length = len(tokenizer.encode(all_model_inputs[i][0]))
            if get_tokenized_length <= MODEL_MAX_LENGTH - TARGET_MAX_LENGTH:
                sampled_indicies.append(i)
                counter += 1
            if counter == num_sample:
                break
        sampled_list = [all_model_inputs[i] for i in sampled_indicies]
        return sampled_list


def filter_data(all_model_inputs, num_sample):
    sampled_list = sample_list(all_model_inputs, num_sample)
    return sampled_list


async def async_generate(llm, model_input, idx, save_dir):
    global TOTAL_COST
    global collected_list
    # system_message = SystemMessage(content=model_input[0])
    human_message = HumanMessage(content=model_input[0])  # if you need it
    while True:
        try:
            response = await llm.agenerate([[human_message]])

            ## llama2
            # response = llm.generate(model_input[0])
            # print(response)

            # token_used = response["usage"]["completion_tokens"]
            token_used = response.llm_output["token_usage"]["total_tokens"]
            TOTAL_COST += token_used / 1000 * 0.002  # gpt-3.5-turbo
            # TOTAL_COST += token_used / 1000 * 0.06 # gpt-4
            print(idx, TOTAL_COST)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None
            # return None
    async with lock:
        copy_data = copy.deepcopy(model_input[1])
        if "feedback" in copy_data.keys():
            del copy_data["feedback"]
        copy_data["feedback"] = "".join([r.text for r in response.generations[0]])
        save_data = dict()
        for key, value in copy_data.items():
            ##decoding 과정 추가
            if type(value) is str:
                value.encode("utf-8").decode("utf-8")
                copy_value = copy.deepcopy(value)
                save_data[key] = copy_value
            elif type(value) is dict:
                make_dict = dict()
                for inner_key, inner_value in value.items():
                    inner_value.encode("utf-8").decode("utf-8")
                    copy_value = copy.deepcopy(inner_value)
                    make_dict[inner_key] = copy_value
                save_data[key] = make_dict
            else:
                save_data[key] = value
        ## 원본 데이터와 매핑
        collected_list.append(save_data)

    return copy_data


async def generate_concurrently(all_model_input, start_idx, args):
    ## 'gpt-3.5-turbo' or 'gpt-4'
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=TARGET_MAX_LENGTH,
        max_retries=100,
    )

    ## llama2
    # llm = LLM(model=args.model_name)

    tasks = [
        async_generate(llm, model_input, i + start_idx, args.save_dir) for i, model_input in enumerate(all_model_input)
    ]
    # tasks = list(filter(lambda x: x is not None, tasks))
    return await tqdm_asyncio.gather(*tasks)


async def main(args):
    # count_total_data(args)
    all_model_inputs = load_and_prepare_data(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    all_results = []
    if args.annotation_start_index is not None and args.annotation_end_index is not None:
        if args.annotation_end_index > len(all_model_inputs):
            all_model_inputs = all_model_inputs[args.annotation_start_index :]
        else:
            all_model_inputs = all_model_inputs[args.annotation_start_index : args.annotation_end_index]
    print(f"total number of samples to be annotated : {len(all_model_inputs)}")
    if len(all_model_inputs) > 300:
        for start_idx in tqdm(range(0, len(all_model_inputs), 300)):
            cur_model_inputs = all_model_inputs[start_idx : start_idx + 300]
            all_results.extend(await generate_concurrently(cur_model_inputs, start_idx, args))
    else:
        all_results = await generate_concurrently(all_model_inputs, 0, args)
    if args.num_sample is not None:
        save_path = os.path.join(args.save_dir, str(args.num_sample))
    else:
        save_path = os.path.join(args.save_dir, args.split)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # with open(os.path.join(save_path, "all_results.json"), "w", encoding="UTF-8") as f:
    #     json.dump(all_results, f, indent=4, ensure_ascii=False)
    save_dict = {"data": collected_list}
    with open(os.path.join(save_path, "feedback.json"), "w", encoding="UTF-8") as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False)
    text_write = open(os.path.join(save_path, "feedback.txt"), "w", encoding="UTF-8")
    text_write.write(f"cost:   {TOTAL_COST}\n")
    text_write.write(f"length: {len(collected_list)}\n")
    text_write.close()


if __name__ == "__main__":
    args = parse_args()
    start = time.time()
    asyncio.run(main(args))
    end = time.time()
    print(f"finished in {end - start:.5f} sec")
