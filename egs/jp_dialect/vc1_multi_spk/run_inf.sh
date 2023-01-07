#!/usr/bin/env bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=6
stop_stage=6
ngpu=3       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=2       # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# Input transcription type: char or phn
# Example
#  char: ミズヲマレーシアカラカワナクテワナラナイノデス。
#  phn: m i z u o m a r e e sh i a k a r a k a w a n a k U t e w a n a r a n a i n o d e s U
# NOTE: original transcription is provided by 漢字仮名交じり文. We convert the input to
# kana or phoneme using OpenJTalk's NLP frontend at the data prep. stage.
trans_type="phn"  # char or phn

# config files
train_config=conf/train_pytorch_transformer.tts_pt.single.smk_emb.yaml
decode_config=conf/decode.yaml

# decoding related
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
model=                      # VC Model checkpoint for decoding. If not specified, automatically set to the latest checkpoint 
voc=PWG                     # vocoder used (GL or PWG)
griffin_lim_iters=64        # The number of iterations of Griffin-Lim

# pretrained model related
pretrained_model=phn_train_no_dev_pytorch_ept_train_pytorch_transformer_spk_emb          # available pretrained models: m_ailabs.judy.vtn_tts_pt

# dataset configuration
db_root=downloads/jp_dialect
srcspk=HCK02                   # available speakers: "slt" "clb" "bdl" "rms"
trgspk=TK04

num_train_utts=-1           # -1: use all 932 utts
norm_name=pt_norm                 # used to specify normalized data.
                            # Ex: `pt_norm` for normalization with pretrained model, `self` for self-normalization

# exp tag
tag="multi_spk"  # tag for managing experiments.

#UL
#. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



source_spks=(
    "HCK01" "HCK02" "HCK03" "HCK04" "HCK05" "HSY01" "HSY02" \
    "HSY03" "HSY04" "HSY05" "KOU01" "KOU02" "KOU03" "KOU04" \
    "KOU05" "OS01" "OS02" "OS03" "OS04" "OS05"
    )

target_spks=(
    "TK01" "TK02" "TK03" "TK04" "TK05"
    )

spk_list=( "${source_spks[@]}" "${target_spks[@]}" )

pair=${srcspk}_${trgspk}
pair_dev_set=${pair}_dev
pair_eval_set=${pair}_eval
pair_dt_dir=${dumpdir}/${pair_dev_set}_${norm_name}; mkdir -p ${pair_dt_dir}
pair_ev_dir=${dumpdir}/${pair_eval_set}_${norm_name}; mkdir -p ${pair_ev_dir}

echo "comment starts here"

#TODO - delete these commented out lines, once reference need is complete
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    #local/data_download.sh ${db_root} ${srcspk}
    #local/data_download.sh ${db_root} ${trgspk}

    # download pretrained model for training
    #if [ -n "${pretrained_model}" ]; then
    #    local/pretrained_model_download.sh ${db_root} ${pretrained_model}
    #fi

    # download pretrained PWG
    if [ ${voc} == "PWG" ]; then
        local/pretrained_model_download.sh ${db_root} pwg_${trgspk}
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    for spk in ${spk_list[*]}; do
        local/data_prep_task1.sh ${db_root} data/${spk} ${spk} ${trans_type}
        utils/data/resample_data_dir.sh ${fs} data/${spk} # Downsample to fs from 24k
        utils/fix_data_dir.sh data/${spk}
        utils/validate_data_dir.sh --no-feats data/${spk}
    done
fi

if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional on each frame
    fbankdir=fbank
    for spk in  ${spk_list[*]}; do
        echo "Generating fbanks features for ${spk}..."

        spk_train_set=${spk}_train
        spk_dev_set=${spk}_dev
        spk_eval_set=${spk}_eval

        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${spk} \
            exp/make_fbank/${spk}_${norm_name} \
            ${fbankdir}

        # make train/dev/eval set
        utils/subset_data_dir.sh --last data/${spk} 20 data/${spk}_tmp
        utils/subset_data_dir.sh --last data/${spk}_tmp 10 data/${spk_eval_set}
        utils/subset_data_dir.sh --first data/${spk}_tmp 10 data/${spk_dev_set}
        n=$(( $(wc -l < data/${spk}/wav.scp) - 20 ))
        utils/subset_data_dir.sh --first data/${spk} ${n} data/${spk_train_set}
        rm -rf data/${spk}_tmp

    # If not using pretrained models statistics, calculate in a speaker-dependent way.
        if [ -n "${pretrained_model}" ]; then
            spk_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
        else
            compute-cmvn-stats scp:data/${spk_train_set}/feats.scp data/${spk_train_set}/cmvn.ark
            spk_cmvn=data/${spk_train_set}/cmvn.ark
        fi

        # dump features
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${spk_train_set}/feats.scp ${spk_cmvn} \
            exp/dump_feats/${spk_train_set}_${norm_name} ${dumpdir}/${spk_train_set}_${norm_name}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${spk_dev_set}/feats.scp ${spk_cmvn} \
            exp/dump_feats/${spk_dev_set}_${norm_name} ${dumpdir}/${spk_dev_set}_${norm_name}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${spk_eval_set}/feats.scp ${spk_cmvn} \
            exp/dump_feats/${spk_eval_set}_${norm_name} ${dumpdir}/${spk_eval_set}_${norm_name}
  done
fi

train_dir=${dumpdir}/train_dir && mkdir -p ${train_dir}
dev_dir=${dumpdir}/dev_dir && mkdir -p ${dev_dir}
eval_dir=${dumpdir}/eval_dir && mkdir -p ${eval_dir}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Dictionary and Json Data Preparation"

    # make dummy dict
    dict="data/dummy_dict/X.txt"
    mkdir -p ${dict%/*}
    echo "<unk> 1" > ${dict}

    # make json labels
    for spk in  ${spk_list[*]}; do
        spk_train_set=${spk}_train
        spk_dev_set=${spk}_dev
        spk_eval_set=${spk}_eval
        data2json.sh --feat data/${spk_train_set}/feats.scp \
             data/${spk_train_set} ${dict} > ${dumpdir}/${spk_train_set}_${norm_name}/data.json
	        
	data2json.sh --feat data/${spk_dev_set}/feats.scp \
             data/${spk_dev_set} ${dict} > ${dumpdir}/${spk_dev_set}_${norm_name}/data.json
	
	data2json.sh --feat data/${spk_eval_set}/feats.scp \
             data/${spk_eval_set} ${dict} > ${dumpdir}/${spk_eval_set}_${norm_name}/data.json
    done

    [ -e data/tgt_spk_list_train ] && rm data/tgt_spk_list_train
    [ -e data/tgt_spk_list_dev ] && rm data/tgt_spk_list_dev
    [ -e data/tgt_spk_list_eval ] && rm data/tgt_spk_list_eval
    
    touch data/tgt_spk_list_train
    touch data/tgt_spk_list_dev
    touch data/tgt_spk_list_ev

    for spk in ${target_spks[*]};do
	spk_train_set=${spk}_train
	spk_dev_set=${spk}_dev
	spk_eval_set=${spk}_eval
	echo "${spk}" "${dumpdir}/${spk_train_set}_${norm_name}/data.json" >> data/tgt_spk_list_train
	echo "${spk}" "${dumpdir}/${spk_dev_set}_${norm_name}/data.json" >> data/tgt_spk_list_dev
	echo "${spk}" "${dumpdir}/${spk_eval_set}_${norm_name}/data.json" >> data/tgt_spk_list_eval
    done

    [ -e data/src_spk_list_train ] && rm data/src_spk_list_train
    [ -e data/src_spk_list_dev ] && rm data/src_spk_list_dev
    [ -e data/src_spk_list_eval ] && rm data/src_spk_list_eval
    
    
    touch data/src_spk_list_train
    touch data/src_spk_list_dev
    touch data/src_spk_list_eval
    for spk in ${source_spks[*]}; do
	spk_train_set=${spk}_train
	spk_dev_set=${spk}_dev
	spk_eval_set=${spk}_eval
	echo "${spk}" "${dumpdir}/${spk_train_set}_${norm_name}/data.json" >> data/src_spk_list_train
	echo "${spk}" "${dumpdir}/${spk_dev_set}_${norm_name}/data.json" >> data/src_spk_list_dev
	echo "${spk}" "${dumpdir}/${spk_eval_set}_${norm_name}/data.json" >> data/src_spk_list_eval
    done
fi




if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for spk in  ${spk_list[*]}; do
        spk_train_set=${spk}_train
        spk_dev_set=${spk}_dev
        spk_eval_set=${spk}_eval
        #
        for name in ${spk_train_set} ${spk_dev_set} ${spk_eval_set};do \
            utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
            utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
            steps/make_mfcc.sh \
                --write-utt2num-frames true \
                --mfcc-config conf/mfcc.conf \
                --nj ${nj} --cmd "$train_cmd" \
                data/${name}_mfcc_16k exp/make_mfcc ${mfccdir}
            utils/fix_data_dir.sh data/${name}_mfcc_16k
            sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
                data/${name}_mfcc_16k exp/make_vad ${vaddir}
            utils/fix_data_dir.sh data/${name}_mfcc_16k
        done

        # Check pretrained model existence
        nnet_dir=exp/xvector_nnet_1a
        if [ ! -e ${nnet_dir} ]; then
            echo "X-vector model does not exist. Download pre-trained model."
            wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
            tar xvf 0008_sitw_v2_1a.tar.gz
            mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
            rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
        fi
        # Extract x-vector
        for name in ${spk_train_set} ${spk_dev_set} ${spk_eval_set}; do
            sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
                ${nnet_dir} data/${name}_mfcc_16k \
                ${nnet_dir}/xvectors_${name}
        done


        local/update_json.sh ${dumpdir}/${spk_train_set}_${norm_name}/data.json ${nnet_dir}/xvectors_${spk_train_set}/xvector.scp
        local/update_json.sh ${dumpdir}/${spk_dev_set}_${norm_name}/data.json ${nnet_dir}/xvectors_${spk_dev_set}/xvector.scp
        local/update_json.sh ${dumpdir}/${spk_eval_set}_${norm_name}/data.json ${nnet_dir}/xvectors_${spk_eval_set}/xvector.scp

    done
    # Update json
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Pair Json Data Preparation"

    # make pair json
     make_pair_json_mult_spk.py \
            --src-spk-list data/src_spk_list_train \
            --trg-spk-list data/tgt_spk_list_train \
            -O ${train_dir}/data.json

    make_pair_json_mult_spk.py \
            --src-spk-list data/src_spk_list_dev \
            --trg-spk-list data/tgt_spk_list_dev \
            -O ${dev_dir}/data.json

    make_pair_json_mult_spk.py \
        --src-spk-list data/src_spk_list_eval \
        --trg-spk-list data/tgt_spk_list_eval \
        -O ${eval_dir}/data.json
fi

if [[ -z ${train_config} ]]; then
    echo "Please specify --train_config."
    exit 1
fi

# If pretrained model specified, add pretrained model info in config
if [ -n "${pretrained_model}" ]; then
    pretrained_model_path=$(find ${db_root}/${pretrained_model} -name "snapshot*" | head -n 1)
    train_config="$(change_yaml.py \
        -a enc-init="${pretrained_model_path}" \
        -a dec-init="${pretrained_model_path}" \
        -o "conf/$(basename "${train_config}" .yaml).${tag}.yaml" "${train_config}")"
fi
if [ -z ${tag} ]; then
    expname=multi_spk_${backend}_$(basename ${train_config%.*})
else
    expname=multi_spk_${backend}_${tag}
fi
expdir=exp/${expname}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: VC model training"

    mkdir -p ${expdir}

    tr_json=${train_dir}/data.json
    dt_json=${dev_dir}/data.json

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi


if [ -z "${model}" ]; then
    model="$(find "${expdir}" -name "snapshot*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
    model=$(basename ${model})
    echo "Model found : ${model}"
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    src_spk_dev_set=${srcspk}_dev
    src_spk_eval_set=${srcspk}_eval
    trg_spk_dev_set=${trgspk}_dev
    trg_spk_eval_set=${trgspk}_eval
    make_pair_json.py \
        --src-json ${dumpdir}/${src_spk_dev_set}_${norm_name}/data.json \
        --trg-json ${dumpdir}/${trg_spk_dev_set}_${norm_name}/data.json \
        -O ${pair_dt_dir}/data.json

    make_pair_json.py \
        --src-json ${dumpdir}/${src_spk_eval_set}_${norm_name}/data.json \
        --trg-json ${dumpdir}/${trg_spk_eval_set}_${norm_name}/data.json \
        -O ${pair_ev_dir}/data.json


    echo "stage 6: Decoding and synthesis"

    echo "Decoding..."
    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}_${norm_name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Synthesis..."

    pids=() # initialize pids
    for name in ${pair_dev_set} ${pair_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}

        # Normalization
        # If not using pretrained models statistics, use statistics of target speaker
        if [ -n "${pretrained_model}" ]; then
            trg_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
        else
            trg_cmvn=data/${trg_train_set}/cmvn.ark
        fi
        apply-cmvn --norm-vars=true --reverse=true ${trg_cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

        # GL
        if [ ${voc} = "GL" ]; then
            echo "Using Griffin-Lim phase recovery."
            convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --iters ${griffin_lim_iters} \
                ${outdir}_denorm/${name} \
                ${outdir}_denorm/${name}/log \
                ${outdir}_denorm/${name}/wav
        # PWG
        elif [ ${voc} = "PWG" ]; then
            echo "Using Parallel WaveGAN vocoder."

            # check existence
            voc_expdir=${db_root}/pwg_${trgspk}
            if [ ! -d ${voc_expdir} ]; then
                echo "${voc_expdir} does not exist. Please download the pretrained model."
                exit 1
            fi

            # variable settings
            voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
            voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
            voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
            wav_dir=${outdir}_denorm/${name}/pwg_wav
            hdf5_norm_dir=${outdir}_denorm/${name}/hdf5_norm
            [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
            [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

            # normalize and dump them
            echo "Normalizing..."
            ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
                parallel-wavegan-normalize \
                    --skip-wav-copy \
                    --config "${voc_conf}" \
                    --stats "${voc_stats}" \
                    --feats-scp "${outdir}_denorm/${name}/feats.scp" \
                    --dumpdir ${hdf5_norm_dir} \
                    --verbose "${verbose}"
            echo "successfully finished normalization."

            # decoding
            echo "Decoding start. See the progress via ${wav_dir}/decode.log."
            ${cuda_cmd} --gpu 1 "${wav_dir}/decode.log" \
                parallel-wavegan-decode \
                    --dumpdir ${hdf5_norm_dir} \
                    --checkpoint "${voc_checkpoint}" \
                    --outdir ${wav_dir} \
                    --verbose "${verbose}"

            # renaming
            rename -f "s/_gen//g" ${wav_dir}/*.wav

            echo "successfully finished decoding."
        else
            echo "Vocoder type not supported. Only GL and PWG are available."
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ];then
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Objective Evaluation"

    for name in ${pair_dev_set} ${pair_eval_set}; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --db_root ${db_root} \
            --vocoder ${voc} \
            ${outdir} ${name}
    done
fi
fi
