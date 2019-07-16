#!/usr/bin/env bash

set -eux

export FLAGS_sync_nccl_allreduce=1
# export CUDA_VISIBLE_DEVICES=2

#export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#export MODEL_PATH="/home/yue/Data/ERNIE_stable-1.0.1"
export MODEL_PATH="/home/yue/Projects/baidu/LARK/ERNIE/checkpoints/baiduSKE_0715/step_8000"
export TASK_DATA_PATH="/home/liyue/Data/ernie_processed/0715"

python -u run_classifier.py \
                   --use_cuda false \
                   --do_train false \
                   --do_val true \
                   --do_test true \
                   --batch_size 16 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --dev_set ${TASK_DATA_PATH}/dev0.tsv \
                   --test_set ${TASK_DATA_PATH}/dev.tsv \
                   --vocab_path config/vocab.txt \
                   --checkpoints ./checkpoints/baiduSKE_0715 \
                   --max_seq_len 256 \
                   --ernie_config_path config/ernie_config.json \
                   --num_labels 49 \
                   --random_seed 1

