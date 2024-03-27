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
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat, OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datasets import load_dataset, concatenate_datasets
import openai
from evaluate import load
import copy
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        default=None,
    )
    parser.add_argument("--n", type=int, default=3)
    return parser.parse_args()


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def find_max_indices_in_chunks(all_feedback, feedback_score, chunk_size=6):
    return_list = []
    for get_feedback_index, i in enumerate(range(0, len(feedback_score), chunk_size)):
        chunk = feedback_score[i : i + chunk_size]
        max_index = max(enumerate(chunk), key=lambda x: x[1])[0]
        if max_index != chunk_size - 1:
            return_list.append(all_feedback[get_feedback_index][max_index])
        elif max_index == chunk_size - 1:
            copy_feedback = copy.deepcopy(all_feedback[get_feedback_index][0])
            copy_feedback["prediction"] = "Feedback: None"
            return_list.append(copy_feedback)
    return return_list


def main(args):
    config = read_yaml(args.config_file_path)
    model_last_name = config["GENERATE_MODEL_NAME_OR_PATH"].split("/")[-1]
    first_model_name = config["FEEDBACK_MODEL_NAME_OR_PATH"].split("/")[-1]
    second_model_name = model_last_name
    file_dir = f"./vllm/classification_inference/results/{first_model_name}-{second_model_name}-multifeedback"
    for j in tqdm(range(args.n)):
        i = j + 1
        all_feedback_dir = os.path.join(file_dir, f"{i}_sample", "all_feedback.json")
        predict_result_dir = os.path.join(file_dir, f"{i}_sample", "result_new.json")
        with open(all_feedback_dir, "r") as f:
            all_feedback = json.load(f)
        with open(predict_result_dir, "r") as f:
            predict_result = json.load(f)
        # copy_feedback = copy.deepcopy(all_feedback)
        # random_result = find_random_indices_in_chunks(copy_feedback, predict_result["prediction"])
        # random_save_dir = os.path.join(file_dir, f"{i}_sample", "random_feedback.json")
        # with open(random_save_dir, "w") as f:
        # json.dump(random_result, f, indent=4)
        save_result = find_max_indices_in_chunks(all_feedback, predict_result["prediction"])
        save_dir = os.path.join(file_dir, f"{i}_sample", "selected_feedback.json")
        with open(save_dir, "w") as f:
            json.dump(save_result, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
