#!/usr/bin/env bash

set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=7

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TASK_DATA_PATH="/home/liyue/Data/ernie_processed/0724/relation"
export CHECKPOINT_PATH="/home/liyue/ERNIE/checkpoints/baiduSKE_0724/relation/step_263242"

python predict_classifier.py \
             --use_cuda true \
             --do_train false \
             --do_val false \
             --do_test true \
             --batch_size 32 \
             --init_checkpoint ${CHECKPOINT_PATH} \
             --test_set ${TASK_DATA_PATH}/dev_ner.tsv \
             --vocab_path config/vocab.txt \
             --max_seq_len 256 \
             --ernie_config_path config/ernie_config.json \
             --num_labels 50 \
             --label_map_config ${TASK_DATA_PATH}/label_map_relation.json \
             --random_seed 1
