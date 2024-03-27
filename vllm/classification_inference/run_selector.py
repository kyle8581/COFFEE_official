import yaml
import subprocess
import sys
import shutil
import os
import getpass
from datetime import datetime, timedelta


def current_kst():
    """Get the current time in KST (Korea Standard Time)."""
    UTC_OFFSET = 9
    utc_time = datetime.utcnow()
    kst_time = utc_time + timedelta(hours=UTC_OFFSET)
    return kst_time.strftime("%Y-%m-%d %H:%M:%S KST")


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference_with_yaml_config.py <path_to_yaml_config> <index>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    index = sys.argv[2]
    config = read_yaml(config_file_path)

    # Derive the save directory from the model name or path
    model_last_name = config["GENERATE_MODEL_NAME_OR_PATH"].split("/")[-1]
    use_two_servers = config["USE_TWO_SERVERS"]
    if use_two_servers == "yes":
        first_model_name = config["FEEDBACK_MODEL_NAME_OR_PATH"].split("/")[-1]
        second_model_name = model_last_name
        save_dir = f"vllm/classification_inference/results/{first_model_name}-{second_model_name}-multifeedback"
    else:
        save_dir = f"vllm/results/{model_last_name}-single-model"

    output_dir = os.path.join(save_dir, f"{index}_sample")
    feedback_dir = os.path.join(save_dir, f"{index}_sample", "all_feedback.json")

    # Check if two servers are required
    command = ["vllm/classification_inference/eval_4bit.sh", feedback_dir, output_dir]

    for i, c in enumerate(command):
        print(i, c)

    subprocess.run(command)
