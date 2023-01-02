#!/usr/bin/env bash
set -e

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

download_dir=$1
pretrained_model=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <pretrained_model>"
    echo ""
    echo "Available pretrained models:"
    echo "    - tts1 (JSUT TTS + Speaker Emb)"
    exit 1
fi



case "${pretrained_model}" in
    "phn_train_no_dev_pytorch_train_pytorch_transformer+spkemb")             share_url="https://drive.google.com/open?id=1m2KfwEClR4RWXthXCRwwhTjuCqV3nQ_N" ;;
    *) echo "No such pretrained model: ${pretrained_model}"; exit 1 ;;
esac

dir=${download_dir}/${pretrained_model}

mkdir -p ${dir}
if [ ! -e ${dir}/.complete ]; then
    download_from_google_drive.sh ${share_url} ${dir} ".tar.gz"
    touch ${dir}/.complete
fi
echo "Successfully finished download of pretrained model."