import csv
import json
import os
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_location", type=str, required=True)
    parser.add_argument("--save_location", type=str, default=None)
    parser.add_argument("--max_index", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=5)
    return parser.parse_args()


def main(args):
    folder_location = args.folder_location
    file_indexs = [i for i in range(1, args.max_index + 1)]
    fixall_list = []
    public_wrongonly_list = []
    wrongonly_list = []
    seedtry_list = []
    file_types = ["fixall", "public_wrongonly", "wrongonly"]
    for i in file_indexs:
        folder_name = f"{i}_sample"
        local_fixall = []
        local_public_wrongonly = []
        local_wrongonly = []
        local_seedtry = []
        for j in range(1, args.iteration + 1):
            for file_type in file_types:
                file_name = f"{j}_{file_type}.json"
                file_location = os.path.join(folder_location, folder_name, file_name)
                with open(file_location, "r") as f:
                    load_pass = json.load(f)["pass@1"] * 100
                if file_type == "fixall":
                    local_fixall.append(load_pass)
                elif file_type == "public_wrongonly":
                    local_public_wrongonly.append(load_pass)
                elif file_type == "wrongonly":
                    local_wrongonly.append(load_pass)
        with open(os.path.join(folder_location, folder_name, "seed_try.json"), "r") as f:
            local_seedtry.append(json.load(f)["pass@1"] * 100)
        fixall_list.append(local_fixall)
        public_wrongonly_list.append(local_public_wrongonly)
        wrongonly_list.append(local_wrongonly)
        seedtry_list.append(local_seedtry)

    fixall_list = np.array(fixall_list)
    public_wrongonly_list = np.array(public_wrongonly_list)
    wrongonly_list = np.array(wrongonly_list)
    seedtry_list = np.array(seedtry_list)
    average_fixall = np.average(fixall_list, axis=0)
    average_wrongonly = np.average(wrongonly_list, axis=0)
    average_public_wrongonly = np.average(public_wrongonly_list, axis=0)
    average_seedtry = np.average(seedtry_list, axis=0)

    if args.save_location is None:
        save_location = folder_location
    else:
        save_location = args.save_location
    f = open(os.path.join(save_location, "result.csv"), "w", newline="\n")
    wr = csv.writer(f)
    wr.writerow(average_fixall)
    wr.writerow(average_wrongonly)
    wr.writerow(average_public_wrongonly)
    wr.writerow(average_seedtry)
    f.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
