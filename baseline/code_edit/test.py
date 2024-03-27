import json
from datasets import load_dataset
from mbpp import MBPPeval
import os
from evaluate import load

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


with open("./result_evaluate/generation/chatgpt/chatgpt_mbpp.json", "r") as f:
    mbpp_generation = json.load(f)

mbpp_dataset = load_dataset("mbpp")["test"]


# print("evaluation start")
# results = code_metric.compute(references=references[i], predictions=reconstruct_prediction[i], k=[1])
results = MBPPeval._compute(None, predictions=[mbpp_dataset[0]], references=mbpp_generation[0], k=[1])
print(results)
