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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_dir",
        type=str,
        default=None,
    )
    return parser.parse_args()


def get_dir_and_filename(file_dir):
    file_name = file_dir.split("/")[-1]
    dir_name = "/".join(file_dir.split("/")[:-1])
    return dir_name, file_name


def write_text(data_list):
    write_text = ""
    for i, data in enumerate(data_list):
        write_text += "#" * 25 + f"index {i}" + "#" * 25 + "\n"
        passed = data["logs"]["passed"]
        if passed:
            write_text += "\npassed!\n\n"
        else:
            write_text += "\nfailed!\n\n"
        write_text += data["docstring"] + "\n\n"

        write_text += "*" * 10 + "wrong code" + "*" * 10 + "\n"
        write_text += data["header"] + "\n\n"
        write_text += data["wrong_code"] + "\n\n"
        write_text += "*" * 10 + "feedback" + "*" * 10 + "\n"
        for feedback in data["feedback"]:
            write_text += feedback + "\n"
        write_text += "*" * 10 + "prediction" + "*" * 10 + "\n"
        write_text += data["header"] + "\n\n"
        write_text += data["prediction"] + "\n\n"
        write_text += "*" * 10 + "correct code" + "*" * 10 + "\n"
        write_text += data["header"] + "\n\n"
        write_text += data["correct_code"] + "\n\n"
        write_text += "\n\n\n"
    return write_text


def transform_result(args):
    dir_name, file_name = get_dir_and_filename(args.json_dir)
    with open(args.json_dir, "r") as f:
        load_json = json.load(f)["result"]
    get_text = write_text(load_json)
    save_file_name = file_name.split(".")[0] + ".txt"
    with open(os.path.join(dir_name, save_file_name), "w") as f:
        f.write(get_text)


if __name__ == "__main__":
    args = parse_args()
    transform_result(args)
