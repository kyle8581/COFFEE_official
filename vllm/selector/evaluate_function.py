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