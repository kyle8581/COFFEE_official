import json
import argparse
import random
import os
from datasets import Dataset, load_dataset, DatasetDict
import pandas as pd
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None
        type=str,
    )
    parser.add_argument("--hub_name", default=None)
    args = parser.parse_args()
    return args


def make_list(train_dataset, eval_dataset):
    instruction = "Considering the given incorrect code and description, determine whether the feedback is valid or not. Problem Description:{description}\n Wrong Code:{wrong_code}\n Feedback:{feedback}\n"
    make_train_list = []
    make_eval_list = []
    for data in train_dataset:
        description = data["description"]
        wrong_code = data["wrong_code"]
        valuable_feedback = data["valuabe_feedback"]
        invaluabe_feedback = data["invaluabe_feedback"]
        valuable_text = instruction.format(description=description, wrong_code=wrong_code, feedback=valuable_feedback)
        invaluable_text = instruction.format(
            description=description, wrong_code=wrong_code, feedback=invaluabe_feedback
        )
        pair_dict = {"chosen": valuable_text, "reject": invaluable_text}
        make_train_list.append(pair_dict)

    for data in eval_dataset:
        description = data["description"]
        wrong_code = data["wrong_code"]
        valuable_feedback = data["valuabe_feedback"]
        invaluabe_feedback = data["invaluabe_feedback"]
        valuable_text = instruction.format(description=description, wrong_code=wrong_code, feedback=valuable_feedback)
        invaluable_text = instruction.format(
            description=description, wrong_code=wrong_code, feedback=invaluabe_feedback
        )
        pair_dict = {"chosen": valuable_text, "reject": invaluable_text}
        make_eval_list.append(pair_dict)
    return make_train_list, make_eval_list


def main(args):
    with open(os.path.join(args.data_dir, "train.json"), "r") as f:
        train_data = json.load(f)["data"]
    with open(os.path.join(args.data_dir, "eval.json"), "r") as f:
        eval_data = json.load(f)["data"]
    train_list, eval_list = make_list(train_data, eval_data)
    print(f"train_list: {len(train_list)}")
    print(f"eval_list: {len(eval_list)}")
    processed_train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_list))
    processed_eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_list))

    dataset_annotated = DatasetDict({"train": processed_train_dataset, "eval": processed_eval_dataset})
    if args.hub_name is not None:
        dataset_annotated.push_to_hub(args.hub_name)
    train_save_dict = dict()
    eval_save_dict = dict()
    train_save_dict["data"] = train_list
    eval_save_dict["data"] = eval_list
    with open(os.path.join(args.data_dir, "dpo_train.json"), "w") as f:
        json.dump(train_save_dict, f)
    with open(os.path.join(args.data_dir, "dpo_eval.json"), "w") as f:
        json.dump(eval_save_dict, f)


if __name__ == "__main__":
    args = get_args()
    main(args)
