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

async def async_select(llm, model_input, idx, save_dir):
    human_message = HumanMessage(content=model_input[0])
    while True:
        try:
            response = await llm.agenerate([[human_message]])  # if you need it
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

async def generate_select(all_model_input, start_idx, stop, args, feedback_flag):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=1024,
        top_p=0.95,
        temperature=0.1,
        frequency_penalty=0.0,
        stop=stop,
    )
    tasks = [
        async_select(llm, model_input, i + start_idx, args.save_dir)
        for i, model_input in enumerate(all_model_input)
    ]
    return await tqdm_asyncio.gather(*tasks)


async def asnyc_feedback(llm, model_input, idx, save_dir):
    while True:
        try:
            response = await llm.agenerate(prompts=[model_input[0]])  # if you need it
            # print("Completion result:", completion)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            # response = None
            # return None
    feedback_list = []
    for i in response.generations[0]:
        feedback_list.append(i.text)
    result = {
        "feedback_list": feedback_list,
        "description": model_input[1]["prompt"],
        "wrong_code": model_input[1]["buggy_solution"],
        "correct_code": model_input[1]["canonical_solution"],
        "model_input": model_input[0],
        **model_input[1],
    }

    return result

async def generate_feedback(all_model_input, start_idx, stop, args, feedback_flag):
    assert feedback_flag == True
    llm = OpenAI(
        model_name=args.feedback_model_name,
        openai_api_base=f"http://localhost:{args.feedback_model_port}/v1",
        openai_api_key="EMPTY",
        max_tokens=128,
        top_p=0.95,
        temperature=0.5,
        frequency_penalty=0.4,
        stop=stop,
        n=args.num_candidates,
    )
    tasks = [
        asnyc_feedback(llm, model_input, i + start_idx, args.save_dir)
        for i, model_input in enumerate(all_model_input)
    ]
    return await tqdm_asyncio.gather(*tasks)



async def async_code(llm, model_input, idx, save_dir):
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


async def generate_code(all_model_input, start_idx, stop, args, feedback_flag):
    assert feedback_flag == False
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
        async_code(llm, model_input, i + start_idx, args.save_dir)
        for i, model_input in enumerate(all_model_input)
    ]
    return await tqdm_asyncio.gather(*tasks)