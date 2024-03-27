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
from copy import deepcopy
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
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", help="train, eval")
    parser.add_argument("--prompt", type=str, default="dpo/construct_data/feedback_prompt.yaml")
    parser.add_argument("--prompt_key", type=str, default="feedback_only")
    parser.add_argument("--prompt_label_key", type=str, default="feedback_only_answer")
    parser.add_argument("--inference_model_name", type=str, default=None)
    parser.add_argument("--inference_model_port", type=int, default=8001)
    parser.add_argument("--save_dir", type=str, required=True, help="It should be a NEW DIRECTORY. Please do not use an existing")
    parser.add_argument("--num_sample", type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    # parser.add_argument("--min_context_len", type=int, default=0)
    args = parser.parse_args()
    if args.num_sample:
        args.save_dir = args.save_dir + f"_sample{args.num_sample}"
        
    return args


def load_data(data_name, split=None):
    data = load_dataset(data_name)
    print("=========== dataset statistics ===========")
    print(f"{split} data size : {len(data[split])}")
    print("==========================================")
    return data[split]


def load_json(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def load_prompt(args, key):
    with open(args.prompt, "r", encoding="UTF-8") as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)[key]
    return prompt


def prepare_model_input(prompt:str, ans_prompt:str, args):
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''
    if args.input_path:
        data = load_json(args.input_path)
    else:
        data = load_data(args.dataset_name, split=args.split)

    all_model_data = []
    for idx, d in tqdm(enumerate(data)):
        input_temp = {
            "id": idx,
            **d,
        }
        ## TODO : change this code to prepare the model input from your own data ##
        input_temp['model_input'] = prompt.format(**d)
        input_temp['label'] = ans_prompt.format(**d)
        all_model_data.append(input_temp)
        
    print("The number of data : ", len(all_model_data))
    return all_model_data


def load_and_prepare_data(args):
    prompt = load_prompt(args, args.prompt_key)
    answer_prompt = load_prompt(args, args.prompt_label_key)
    print("Preparing model inputs...")
    all_model_data = prepare_model_input(prompt, answer_prompt, args)
    return all_model_data

def sample_indices(all_model_inputs, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_inputs)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices

def filter_data(all_model_data, num_sample):
    if num_sample:
        sampled_indices = sample_indices(all_model_data, num_sample)
        all_model_data = [all_model_data[i] for i in sampled_indices]
    return all_model_data

async def async_generate(llm, model_input, idx, save_dir):
    while True:
        try:
            response = await llm.agenerate(prompts=[model_input['model_input']])  # if you need it
            # print("Completion result:", completion)
            break
        except Exception as e:
            pass
            # print(f"Exception occurred: {e}")
            # response = None
            # return None

    result = {
        **model_input,
        "prediction": response.generations[0][0].text,
    }

    return result


async def generate_concurrently(all_model_data, start_idx, stop, args):
    llm = OpenAI(
        model_name=args.inference_model_name,
        openai_api_base=f"http://localhost:{args.inference_model_port}/v1",
        openai_api_key="EMPTY",
        max_tokens=1024,
        top_p=0.95,
        temperature=0.5,
        frequency_penalty=0.4,
        stop=stop,
    )
    tasks = [
        async_generate(llm, model_input, i + start_idx, args.save_dir) for i, model_input in enumerate(all_model_data)
    ]
    return await tqdm_asyncio.gather(*tasks)


async def main(args):
    all_model_data = load_and_prepare_data(args)
    all_model_data = filter_data(all_model_data, args.num_sample)

    # Check if the save_dir exists
    if os.path.exists(args.save_dir):
        print("The save_dir already exists. Please change the save_dir.")

    # os.makedirs(args.save_dir, exist_ok=True)
    # all_results = await generate_concurrently(all_model_data, 0, None, args)

    ## version 2
    all_results = []
    for start_idx in tqdm(range(0, len(all_model_data), 100)):
        cur_model_data = all_model_data[start_idx:start_idx + 100]
        all_results.extend(await generate_concurrently(cur_model_data, start_idx, None, args))
    
    total_result_path = args.save_dir + "_total_results.json"
    with open(os.path.join(total_result_path), "w", encoding='UTF-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
