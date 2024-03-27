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
from apps import compute_metrics_apps

import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# bigcode/humanevalpack, mbpp, codeparrot/apps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int)
    return parser.parse_args()


def evaluation_humanevalsynth(prediction_result):
    def reconsruct_preds(preds):
        reconstruct_preds = []
        for data in preds:
            prediction = data[0]
            reconstruct_preds.append([prediction])
        return reconstruct_preds

    reconstruct_prediction = reconsruct_preds(prediction_result)
    labels = load_dataset("bigcode/humanevalpack")["test"]["test"]
    code_metric = load("Muennighoff/code_eval_octopack")
    language = "python"
    timeout = 1
    references = labels
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
    return_list = list()
    for i in range(len(logs)):
        copy_prediction = copy.deepcopy(prediction_result[i])
        log_dict = {}
        log_dict["logs"] = logs[i][0][1]
        return_list.append(log_dict)
    return_dict["result"] = return_list
    return return_dict


def evaluate_apps(prediction_result, index, level):
    result_list = []
    results, errors = compute_metrics_apps(prediction_result[index], index, level=level)
    return_dict = dict()
    return_dict["pass@1"] = results["strict_accuracy"]
    return_dict["errors"] = errors
    result_list.append(return_dict)
    return return_dict


def evaluate_mbpp(prediction_result):
    def reconsruct_preds(preds, reference):
        reconstruct_preds = []
        for i, data in enumerate(preds):
            prediction = data[0]
            make_prediction_list = [prediction]
            prediction_list = [copy.deepcopy(make_prediction_list) for _ in range(len(reference[i]))]
            reconstruct_preds.append(prediction_list)
        return reconstruct_preds

    labels = load_dataset("mbpp")["test"]

    references = [labels[i]["test_list"] for i in range(len(labels))]
    reconstruct_prediction = reconsruct_preds(prediction_result, references)

    code_metric = load("code_eval")

    return_dict = dict()
    result_dict = dict()
    assert len(reconstruct_prediction) == len(references)
    print("evaluation start")
    pass_1_list = list()
    for i in tqdm(range(len(references))):
        results = code_metric.compute(references=references[i], predictions=reconstruct_prediction[i], k=[1])
        make_key = f"index-{i}"
        result_dict[make_key] = results
        pass_1_list.append(results[0]["pass@1"])
    pass_1 = sum(pass_1_list) / len(pass_1_list)
    return_dict["pass@1"] = pass_1
    current_prediction_list = list()
    for i in range(len(prediction_result)):
        copy_prediction = copy.deepcopy(prediction_result[i])
        current_prediction_list.append(copy_prediction)
    return_dict["result"] = current_prediction_list
    return_dict["evaluate"] = result_dict
    return return_dict


if __name__ == "__main__":
    # main()
    args = parse_args()
    chatgpt_humaneval = (
        "./result_evaluate/generation/chatgpt/chatgpt_humaneval.json"
    )
    with open(chatgpt_humaneval, "r") as f:
        chatgpt_humaneval_generation = json.load(f)
    chatgpt_humaneval_result = evaluation_humanevalsynth(chatgpt_humaneval_generation)
    save_dir = "./baseline/code_edit/result/chatgpt_humaneval_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"seed_try.json"), "w") as f:
        json.dump(chatgpt_humaneval_result, f, indent=4)
    ###
    octocoder_humaneval = (
        "./result_evaluate/generation/octocoder/octocoder_humaneval.json"
    )
    with open(octocoder_humaneval, "r") as f:
        octocoder_humaneval_generation = json.load(f)
    octocoder_humaneval_result = evaluation_humanevalsynth(octocoder_humaneval_generation)
    save_dir = "./baseline/code_edit/result/octocoder_humaneval_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"seed_try.json"), "w") as f:
        json.dump(octocoder_humaneval_result, f, indent=4)

    # octocoder_mbpp = "./result_evaluate/generation/octocoder/octocoder_mbpp.json"
    # with open(octocoder_mbpp, "r") as f:
    #     octocoder_mbpp_generation = json.load(f)
    # octocoder_mbpp_result = evaluate_mbpp(octocoder_mbpp_generation)
    # save_dir = "./baseline/code_edit/result/octocoder_mbpp_result"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(os.path.join(save_dir, f"seed_try.json"), "w") as f:
    #     json.dump(octocoder_mbpp_result, f, indent=4)
    # ###
    # chatgpt_mbpp = "./result_evaluate/generation/chatgpt/chatgpt_mbpp.json"
    # with open(chatgpt_mbpp, "r") as f:
    #     chatgpt_mbpp_generation = json.load(f)
    # chatgpt_mbpp_result = evaluate_mbpp(chatgpt_mbpp_generation)
    # save_dir = "./baseline/code_edit/result/chatgpt_mbpp_result"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(os.path.join(save_dir, f"seed_try.json"), "w") as f:
    #     json.dump(chatgpt_mbpp_result, f, indent=4)
