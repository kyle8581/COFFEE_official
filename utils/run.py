import sys
import io
import subprocess
import os
import time


def run_given_code(code, test_input):
    with open("temp.py", "w") as f:
        f.write(code)
    command = ["python", "temp.py"]
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.communicate(input=test_input.encode())[0]
    while p.poll() is None:
        # Process hasn't exited yet, let's wait some
        time.sleep(0.5)
    if p.returncode == 1:
        print("error!")
        return False
    else:
        assert p.returncode == 0
    get_output_text = output.decode()
    if os.path.isfile("temp.py"):
        os.remove("temp.py")
    return get_output_text
