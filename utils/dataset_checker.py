
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--range", type=str, default="0,100")
parser.add_argument("--dataset", type=str, default=None)
args = parser.parse_args()

dataset = load_dataset(args.dataset)
trainset = dataset['train']

import json

import textwrap

def pretty_print_code(code):
    """
    Print the code with wrapped text and handle newline characters.
    """
    code = code.replace("\\n", "\n")  # Convert escaped newlines to actual newlines
    wrapped_code = '\n'.join(textwrap.fill(line, width=100) for line in code.splitlines())
    print(code)

def print_in_color(text, color_code):
    """Prints the given text in the specified color."""
    RESET = '\033[0m'
    print(f"{color_code}{text}{RESET}")

RED = '\033[91m'
GREEN = '\033[92m'


start, end = list(map(int, args.range.split(",")))
i = start
while start <= i < end:
    data = trainset[i]
    
    # Remove the metadata field
    if "metadata" in data:
        del data["metadata"]

    # Print the "correct_code" and "wrong_code" fields
    print_in_color("\n=== Wrong Code ===", GREEN)
    pretty_print_code(data["wrong_code"])

    print_in_color("=== Correct Code ===", GREEN)
    pretty_print_code(data["correct_code"])

    print_in_color("=== Feedback ===", GREEN)
    pretty_print_code(data['feedback'])
    print_in_color(f"Current Index: {i}", RED)
    print_in_color(f"Problem Num: {data['problem_id']}", RED)
    cmd = input("Enter 'n' for next, 'p' for previous, or a specific index (0-99): ")

    if cmd == 'n':
        if i < 99:  # So it doesn't exceed the range
            i += 1
    elif cmd == 'p':
        if i > 0:  # So it doesn't go negative
            i -= 1
    else:
        try:
            new_index = int(cmd)
            if 0 <= new_index < 100:
                i = new_index
            else:
                print("Index out of range. Showing current data.")
        except ValueError:
            print("Invalid input. Showing current data.")
