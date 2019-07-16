set -eux

export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=2

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export MODEL_PATH="/home/liyue/Data/ERNIE_stable-1.0.1"
export TASK_DATA_PATH="/home/liyue/Data/ernie_processed/0715"

python -u run_classifier.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 32 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/dev0.tsv \
                   --test_set ${TASK_DATA_PATH}/dev.tsv \
                   --vocab_path config/vocab.txt \
                   --checkpoints ./checkpoints/baiduSKE_0715 \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 5 \
                   --max_seq_len 256 \
                   --ernie_config_path config/ernie_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 49 \
                   --random_seed 1