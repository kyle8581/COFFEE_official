import json
import argparse
import random
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None
        type=str,
    )
    args = parser.parse_args()
    return args


def get_correct_score(result_list):
    correct_counter = 0
    if result_list == [] or result_list == None:
        return 0
    for result in result_list:
        if result:
            correct_counter += 1
    ratio = correct_counter / len(result_list)
    return ratio


def check_list(lst):
    # 길이가 5가 아닌 경우 True 반환
    if len(lst) != 6:
        return True
    # 모든 element가 0인 경우 False 반환
    if all(x["ratio"] == 0 for x in lst):
        return False
    get_first_ratio = lst[0]["ratio"]
    if all(x["ratio"] == get_first_ratio for x in lst):
        return False
    if all(x["ratio"] == 1 for x in lst):
        return False
    # 그 외의 경우 True 반환
    return True


def make_data(args):
    dict_by_index = {}
    for i in range(6):
        with open(os.path.join(args.data_dir, f"seed_try_{i}_with_testcase_results.json"), "r") as f:
            load_json = json.load(f)
        for data in load_json:
            if "testcase_pass_results" not in data:
                continue
            ratio = get_correct_score(data["testcase_pass_results"])
            wrong_code = data["wrong_code"]
            feedback = data["feedback"]
            description = data["description"]
            index = data["index"]
            make_dict = {
                "ratio": ratio,
                "wrong_code": wrong_code,
                "feedback": feedback,
                "description": description,
                "index": index,
            }
            if index not in dict_by_index:
                dict_by_index[index] = []
            else:
                assert dict_by_index[index][0]["wrong_code"] == wrong_code
                assert dict_by_index[index][0]["description"] == description
            dict_by_index[index].append(make_dict)

    counter = 0
    make_final_list = []
    none_counter = 0
    for key, values in dict_by_index.items():
        check_valid = check_list(values)
        if check_valid:
            ratio_and_feedback = []
            ratio_list = []
            for value in values:
                ratio_and_feedback.append((value["ratio"], value["feedback"]))
                ratio_list.append(value["ratio"])
            wrong_code = values[0]["wrong_code"]
            description = values[0]["description"]
            index = values[0]["index"]
            for i in range(1, 6):
                assert wrong_code == values[i]["wrong_code"]
                assert description == values[i]["description"]
                assert index == values[i]["index"]
            sorted_ratio_and_feedback = sorted(ratio_and_feedback, key=lambda x: x[0], reverse=True)
            most_valuable_feedback = sorted_ratio_and_feedback[0][1]
            most_invaluabe_feedback = sorted_ratio_and_feedback[-1][1]
            if "None" in most_valuable_feedback:
                none_counter += 1
            assert sorted_ratio_and_feedback[0][0] > sorted_ratio_and_feedback[-1][0]
            make_dict = {
                "wrong_code": wrong_code,
                "description": description,
                "index": index,
                "valuabe_feedback": most_valuable_feedback,
                "invaluabe_feedback": most_invaluabe_feedback,
            }
            make_final_list.append(make_dict)

    print(f"total length: {len(make_final_list)}")
    print(f"none counter: {none_counter}")

    eval_length = int(len(make_final_list) * 0.1)
    eval_list = random.sample(make_final_list, eval_length)
    train_list = [x for x in make_final_list if x not in eval_list]

    train_save_dict = {}
    eval_save_dict = {}
    train_save_dict["data"] = train_list
    eval_save_dict["data"] = eval_list
    with open(os.path.join(args.data_dir, "train.json"), "w") as f:
        json.dump(train_save_dict, f, indent=4)

    with open(os.path.join(args.data_dir, "eval.json"), "w") as f:
        json.dump(eval_save_dict, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    make_data(args)
