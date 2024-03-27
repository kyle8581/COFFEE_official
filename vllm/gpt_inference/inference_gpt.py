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
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
    )
    parser.add_argument("--data_name", type=str, default="bigcode/humanevalpack")
    parser.add_argument("--prompt", type=str, default="vllm/feedback_inference_test.yaml")
    parser.add_argument("--prompt_key", type=str, default="base")
    parser.add_argument(
        "--gpt_feedback_result",
        type=str,
        default=None,
    )
    parser.add_argument("--iterative_strategy", default="wrongonly", choices=["wrongonly", "public_wrongonly"])
    parser.add_argument("--use_feedback", type=str, default="Yes", choices=["Yes", "No"])
    parser.add_argument("--do_iterate", action="store_true", help="do iteration")
    parser.add_argument("--iterate_num", type=int, default=5, help="number of iteration")
    parser.add_argument("--seed_json", default=None, help="use saved seed json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num_sample", type=int, default=3, help="number of samples to generate")
    parser.add_argument("--port", type=int)

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
    match = pattern.search(s)
    if match:
        docstring = match.group(1)
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


def load_from_gpt_inference(json_file):
    with open(json_file, "r") as f:
        load_json = json.load(f)["result"]
    make_dict = dict()
    for data in load_json:
        task_id = data["task_id"]
        raw_feedback = data["feedback"]

        feedback = "\n".join(raw_feedback)

        copy_data = copy.deepcopy(data)
        if "Feedback for Refining the Code:" in feedback:
            feedback = feedback.split("Feedback for Refining the Code:")[1].strip()
        copy_data["feedback"] = feedback
        make_dict[task_id] = copy_data
    return make_dict


# def load_and_prepare_data(args):
#     dataset = load_data(args.data_name, args.split)
#     prompt = load_prompt(args.prompt)
#     all_model_inputs = []
#     with open(args.gpt_feedback_result, "r") as f:
#         gpt4_feedback = json.load(f)["result"]
#     print("### load and prepare data")
#     for index, data in tqdm(enumerate(dataset)):
#         model_input, problem, function_header = prepare_model_input(prompt, data)
#         get_feedback = "".join(gpt4_feedback[index]["feedback"])
#         data["gpt-feedback"] = get_feedback
#         new_model_input = f"{model_input}\nFeedback:{get_feedback}\nCorrect code:\n{function_header}"
#         data["header"] = function_header
#         all_model_inputs.append([new_model_input, data])
#     return all_model_inputs


def load_and_prepare_data(args):
    dataset = load_data(args.data_name, args.split)
    prompt = load_prompt(args.prompt)
    all_model_inputs = []
    feedback_dict = load_from_gpt_inference(args.gpt_feedback_result)
    print("### load and prepare data")
    for index, data in tqdm(enumerate(dataset)):
        model_input, problem, function_header = prepare_model_input(prompt, data)
        get_feedback = feedback_dict[data["task_id"]]
        assert data["task_id"] == get_feedback["task_id"]
        extract_feedback = get_feedback["feedback"]
        data["feedback"] = extract_feedback
        new_model_input = f"{model_input}\nFeedback:{extract_feedback}\nCorrect code:\n{function_header}"
        data["header"] = function_header
        all_model_inputs.append([new_model_input, data])
    return all_model_inputs


async def async_generate(llm, model_input, idx, save_dir):
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


async def generate_concurrently(all_model_input, start_idx, stop, args):
    llm = OpenAI(
        model_name=args.model_name,
        openai_api_base=f"http://localhost:{args.port}/v1",
        openai_api_key="EMPTY",
        max_tokens=512,
        top_p=0.95,
        temperature=0.5,
        frequency_penalty=0.4,
        stop=stop,
    )
    tasks = [
        async_generate(llm, model_input, i + start_idx, args.save_dir) for i, model_input in enumerate(all_model_input)
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


async def main(args):
    all_model_inputs = load_and_prepare_data(args)
    for i in range(args.num_sample):
        print(f"### {i+1} sample")
        save_path = os.path.join(args.save_dir, f"{i+1}_sample")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        all_results = await generate_concurrently(all_model_inputs, 0, None, args)

        eval_results = evaluation(all_results, args)
        wrongonly_eval = copy.deepcopy(eval_results)
        publiconly_eval = copy.deepcopy(eval_results)
        with open(os.path.join(save_path, "seed_try.json"), "w", encoding="UTF-8") as f:
            json.dump(eval_results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
    print("Done!")
