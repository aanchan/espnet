#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import sys
from io import open

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge source and target data.json files into one json file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-spk-list", type=str, help="list of source speakers with path to JSON files  - "
                                                         "<SPK_NAME> <SPK_JSON_FILE_PATH> - one spk per line")
    parser.add_argument(
        "--trg-spk-list",
        type=str,
        default=None,
        help="list of target speakers with path to JSON files - <SPK_NAME> <SPK_JSON_FILE_PATH> - one spk per line")

    parser.add_argument(
        "--num_utts", default=-1, type=int, help="Number of utterances (take from head)"
    )
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument(
        "--out",
        "-O",
        type=str,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    src_spks = {}
    with open(args.src_spk_list, "r") as f:
        for line in f:
            spk, spk_json = line.strip().split(' ')
            src_spks[spk] = spk_json

    trg_spks = {}
    with open(args.trg_spk_list, "r") as f:
        for line in f:
            spk, spk_json = line.strip().split(' ')
            trg_spks[spk] = spk_json
    data = {"utts": {}}
    count = 1 
    for trg_spk, trg_json_path in trg_spks.items():

        with open(trg_json_path, "rb") as trg_f:
            trg_json = json.load(trg_f)["utts"]

            for src_spk, src_json_path in src_spks.items():
                with open(src_json_path, "rb") as src_f:

                    src_json = json.load(src_f)["utts"]

                    # get source and target speaker
                    _ = list(src_json.keys())[0].split("_")
                    srcspk = _[0]

                    _ = list(trg_json.keys())[0].split("_")
                    trgspk = _[0]

                    

                    # (dirty) loop through input only because in/out should have same files
                    for k, v in src_json.items():
                        _ = k.split("_")
                        number = "_".join(_[1:])
                        number_key = f'{count:04}'
                        print(number_key)
                        entry = {"input": src_json[srcspk + "_" + number]["input"]}

                        
                        entry["output"] = trg_json[trgspk + "_" + number]["input"]
                        entry["output"][0]["name"] = "target1"

                        data["utts"][number_key] = entry
                        count += 1
                        

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, "w", encoding="utf-8")

    json.dump(
        data,
        out,
        indent=4,
        ensure_ascii=False,
        separators=(",", ": "),
    )
