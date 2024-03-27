import json
import sys
import faulthandler
import platform

# used for debugging to time steps
from datetime import datetime

# to run the solution files we're using a timing based approach
import signal

import numpy as np

# for capturing the stdout
from io import StringIO

# used for testing the code that reads from input
from unittest.mock import patch, mock_open

from pyext import RuntimeModule

from enum import Enum


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)
timeout = 4  # seconds


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def run_test(sample, test=None, debug=False):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    # reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError:
        in_outs = None
    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        return in_outs, [None]
    elif test is not None:
        results = []
        errors = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 0 compilation error = {e}")
                results.append(-2)
                errors.append(f"type 0 compilation error = {e}")
                return results, errors
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 1 compilation error = {e}")
                results.append(-2)
                errors.append(f"type 1 compilation error = {e}")
                return results, errors
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")
            results.append(-2)
            errors.append(f"unable to get function error = {e}")
            return results, errors

        for index, inputs in enumerate(in_outs["inputs"]):
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
            except:
                True

            if debug:
                print(
                    f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}"
                )
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp_result = output == in_outs["outputs"][index]
                    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                    except:
                        True
                    results.append(tmp_result)

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    if debug:
                        print(f"Standard input runtime error or time limit exceeded error = {e}")
                    results.append(-1)
                    errors.append(f"Standard input runtime error or time limit exceeded error = {e}")
                    return results, errors
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(
                        f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs["outputs"][index], list):
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                        results.append(-1)
                        errors.append(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                    signal.alarm(0)

                if not passed:
                    nl = "\n"
                    if debug:
                        if not isinstance(inputs, list):
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                        else:
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                    if not isinstance(inputs, list):
                        errors.append(
                            f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        errors.append(
                            f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    return results, errors
                if passed and debug:
                    print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

                if custom_compare_(output, in_outs["outputs"][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check1 exception = {e}")
                    errors.append(f"Failed check1 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [
                            x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                        ]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                    in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check2 exception = {e}")
                    errors.append(f"Failed check2 exception = {e}")
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check3 exception = {e}")
                    errors.append(f"Failed check3 exception = {e}")
                    pass

                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs["outputs"][index]]
                    tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                    )
                except Exception as e:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                        tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                        )
                except Exception as e:
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = output == in_outs["outputs"][index]
                except Exception as e:
                    if debug:
                        print(f"Failed check4 exception = {e}")
                    errors.append(f"Failed check4 exception = {e}")
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = set(frozenset(s) for s in output) == set(
                        frozenset(s) for s in in_outs["outputs"][index]
                    )
                except Exception as e:
                    if debug:
                        print(f"Failed check5 exception = {e}")
                    errors.append(f"Failed check5 exception = {e}")

                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (
                        set(frozenset(round(float(t), 3) for t in s) for s in output)
                        == set(frozenset(round(float(t), 3) for t in s) for s in in_outs["outputs"][index])
                    )
                except Exception as e:
                    if debug:
                        print(f"Failed check6 exception = {e}")
                    errors.append(f"Failed check6 exception = {e}")

                if tmp_result == True and debug:
                    print("PASSED")

                results.append(tmp_result)

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

    return results, errors


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


import itertools
import json
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset

DATASET = "codeparrot/apps"
TIMEOUT = 10


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(sample, generation, debug, result, error):
        temp_result, temp_error = run_test(sample, test=generation, debug=debug)
        result.append(temp_result)
        error.append(temp_error)

    manager = multiprocessing.Manager()
    result = manager.list()
    error = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result, error))
    p.start()
    p.join(timeout=timeout + 1)
    error = list(error)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
        return [-1], error

    return result[0], error


def evaluate_generations(generations: list, index: int, level: str = "all", debug: bool = False):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.
    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"
    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset
    apps_eval = load_dataset(DATASET, split="test", difficulties=[level])
    results = {}
    # code generations for problem (index)
    problem_generations = generations
    # get corresponding samples from APPS dataset
    sample = apps_eval[index]
    res = []
    # loop over the generations
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        curr_error = [None]
        try:
            curr_res, curr_error = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {index}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases")
        except Exception as e:
            curr_error = [f"Compilation failed, test framework exception = {repr(e)}{e}\n"]
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
        results[index] = res
    return results, curr_error


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def get_results(results: Dict[int, list], count_errors: bool = False, k_list: list = [1, 10, 100]):
    """
    Given the results evaluated against the testcases we output some statistics.
    For single generations:
    >>> example_results = {0: [[-2]], 1: [[False,False]], 2: [[True,True]], 3: [[False,True,False,True]], 4: [[-1,-1]]}
    >>> get_results(example_results, count_errors=True)
    Computing accuracy metrics...
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of problems evaluated = 5
    Average Accuracy : 0.3
    Strict Accuracy : 0.2
    {'avg_accuracy': 0.3, 'strict_accuracy': 0.2, 'pass_at_k': None}
    For multiple generations:
    >>> example_results = {0: [[-2], [True, True, True]], 1: [[-1,-1, -1], [True, False, True]]}
    >>> get_results(example_results, k_list=[1, 2])
    Computing pass@k metric for multiple generations...
    {'pass@1': 0.25, 'pass@2': 0.5}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 0.25, 'pass@2': 0.5}}
    """

    metrics = {"avg_accuracy": None, "strict_accuracy": None, "pass_at_k": None}

    if len(results) == 1:
        # for single generations we compute average accuracy and stric accuracy: original APPS metrics
        # print("Computing accuracy metrics...")
        res = []
        per_prob_res = []
        all_correct = []
        for index in results:
            problem_results = np.asarray(results[index])
            res.extend(problem_results)
            per_prob_res.append(np.mean(problem_results > 0))
            all_correct.append(np.all(problem_results > 0))
        # we count campilation and runtime errors once per pronlem
        compile_errors = len([e for e in res if -2 in e])
        runtime_errors = len([e for e in res if -1 in e])
        total_testcases = len(res)
        # if count_errors:
        #     print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}")
        #     print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
        #     print(f"number of problems evaluated = {total_testcases}")

        # print(f"Average Accuracy : {np.mean(per_prob_res)}")
        # print(f"Strict Accuracy : {np.mean(all_correct)}")
        metrics["avg_accuracy"] = np.mean(per_prob_res)
        metrics["strict_accuracy"] = np.mean(all_correct)

    else:
        # for multiple generations we use pass@k metric used in the HumanEval benchmark
        # we use strict accuracy, a generation is valid if it has to pass all the tests
        print("Computing pass@k metric for multiple generations...")
        # total is list with nb generations per task (task=index)
        # correct is number of generations that passed all tests per task
        total = []
        correct = []
        for index in results:
            all_correct = []
            for generation in results[index]:
                gen = np.array(generation)
                all_correct.append(np.all(gen > 0))
            total.append(len(all_correct))
            correct.append(sum(all_correct))
        total = np.array(total)
        correct = np.array(correct)
        ks = k_list
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
        print(pass_at_k)
        metrics["pass_at_k"] = pass_at_k
    return metrics


def compute_metrics_apps(generations, index, level="all", k_list=[1, 10, 100], count_errors=True, debug=False):
    """Return metrics for the given generations.
    Args:
        generations: list of code generations for each problem (each generation is a list of generations)
        k_list: list of k values to compute pass@k when using multiple generations
        count_errors: whether to count compilation and runtime errors when using single generations
        level: difficulty level in APPS dataset that was used for the given generations (from: "all", "introductory", "interview", "competition")
    Returns:
        metrics: dict of metrics
    Examples:
    >>> import json
    >>> # lists of solutions to the two first APPS problems (note not all solutions pass all tests)
    >>> solution_sample1 = json.load(open("test_examples/solutions_problem_1.json", "r"))
    >>> solution_sample2 = json.load(open("test_examples/solutions_problem_2.json", "r"))
    >>> single_solutions = [solution_sample1[:1], solution_sample2[:1]]
    >>> compute_metrics(single_solutions, level="all")
    Computing accuracy metrics...
    number of compile errors = 0 avg = 0.0
    number of runtime errors = 0 avg = 0.0
    number of problems evaluated = 2
    Average Accuracy : 1.0
    Strict Accuracy : 1.0
    {'avg_accuracy': 1.0, 'strict_accuracy': 1.0, 'pass_at_k': None}
    >>> multiple_solutions = [solution_sample1[:3], solution_sample2[:3]]
    >>> compute_metrics(multiple_solutions, level="all", k_list=[1, 2, 3])
    Computing pass@k metric for multiple generations...
    {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}}
    """
    results, error = evaluate_generations(generations, index=index, level=level, debug=debug)
    metrics = get_results(results, count_errors=count_errors, k_list=k_list)
    return metrics, error
