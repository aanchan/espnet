#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from io import open
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='Make json file for autoencoder-style pretraining.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-json', type=str,
                        help='Json file for input')
    parser.add_argument('--output-json', type=str,
                        help='Json file for output')
    parser.add_argument('--spk-label', type=str,
                        help='Json file for output')
    parser.add_argument('--input-feat', type=str,
                        choices=['input','output'], default='input',
                        help='Feature to use from the input')
    parser.add_argument('--output-feat', type=str,
                        choices=['input','output'], default='input',
                        help='Feature to use from the output')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with open(args.input_json, 'rb') as f:
        input_json = json.load(f)['utts']
    with open(args.output_json, 'rb') as f:
        output_json = json.load(f)['utts']
    if args.spk_label:
        with open(args.spk_label, 'rb') as f:
            spk_label = json.load(f)['utts']

    data = {"utts": {}}
    # (dirty) loop through input only because in/out should have same files
    for k, v in input_json.items():

        entry = {"input": input_json[k][args.input_feat],
                 "output": output_json[k][args.output_feat],
                 }
        entry["input"][0]['name'] = 'input1'
        entry["output"][0]['name'] = 'target1'
        # delete speaker embedding in output
        if args.output_feat == 'input':
            while len(entry["output"]) > 1:
                entry["output"].pop() 
        entry["utt2spk"] = input_json[k]["utt2spk"]
        if args.spk_label:
            entry["spk_label"] = int(spk_label[k]["spk_label"])
        data["utts"][k] = entry

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')

    json.dump(data, out,
              indent=4, ensure_ascii=False,
              separators=(',', ': '),
              )
