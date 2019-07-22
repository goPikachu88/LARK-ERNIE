#!/usr/bin/env bash

set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=4
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CHECKPOINT_PATH="/home/liyue/ERNIE/checkpoints/baiduSKE_ner/step_50000"
export TASK_DATA_PATH="/home/liyue/Data/ernie_processed"

python -u run_sequence_labeling.py \
                   --use_cuda true \
                   --do_train false \
                   --do_val true \
                   --do_test true \
                   --batch_size 16 \
                   --init_checkpoint ${CHECKPOINT_PATH} \
                   --num_labels 65 \
                   --label_map_config ${TASK_DATA_PATH}/label_map_all_entities.json \
                   --dev_set ${TASK_DATA_PATH}/ner/dev0.tsv \
                   --test_set ${TASK_DATA_PATH}/ner/dev.tsv \
                   --vocab_path config/vocab.txt \
                   --ernie_config_path config/ernie_config.json \
                   --max_seq_len 256 \
                   --random_seed 1