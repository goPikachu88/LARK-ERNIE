#!/usr/bin/env bash

set -eux

export INPUT_DATA_PATH="/home/yue/Data/ernie_processed"
export OUTPUT_DATA_PATH="/home/yue/Data/ernie_processed/ner56"

#python -u preprocess.py \
#                   --data ${INPUT_DATA_PATH}/train_postag.json \
#                   --label_map ${INPUT_DATA_PATH}/ner56/label_map_entities_56.json \
#                   --output ${OUTPUT_DATA_PATH}/train.tsv
#
#python -u preprocess.py \
#                   --data ${INPUT_DATA_PATH}/dev_data_postag.json \
#                   --label_map ${INPUT_DATA_PATH}/ner56/label_map_entities_56.json \
#                   --output ${OUTPUT_DATA_PATH}/dev.tsv
#
#python -u preprocess.py \
#                   --data ${INPUT_DATA_PATH}/dev0_postag.json \
#                   --label_map ${INPUT_DATA_PATH}/ner56/label_map_entities_56.json \
#                   --output ${OUTPUT_DATA_PATH}/dev0.tsv


python -u preprocess.py \
                   --data ${INPUT_DATA_PATH}/train_postag.json \
                   --output ${OUTPUT_DATA_PATH}/train.tsv

python -u preprocess.py \
                   --data ${INPUT_DATA_PATH}/dev_data_postag.json \
                   --output ${OUTPUT_DATA_PATH}/dev.tsv

python -u preprocess.py \
                   --data ${INPUT_DATA_PATH}/dev0_postag.json \
                   --output ${OUTPUT_DATA_PATH}/dev0.tsv
