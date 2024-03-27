from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument(
        "--peft_model_path",
        type=str,
    )
    parser.add_argument("--save_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    if args.save_path is None:
        save_path = "/".join(args.peft_model_path.split("/")[:-1])
    else:
        save_path = args.save_path + "-merged"
        os.makedirs(save_path, exist_ok=True)
    save_dir = save_path
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_path, f"info.txt"), "w") as f:
        f.write(f"peft model path: {args.peft_model_path}\n")
        f.write(f"Model saved to {args.base_model_name_or_path}")
        f.close()
    print(f"Model saved to {args.base_model_name_or_path}-merged")


if __name__ == "__main__":
    main()
