#!/bin/bash
# Copyright 2022 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===== Normally fixed parameters
dataset_name=$1           # e.g. "wikipedia-20200501.en,bookcorpus"
dataset_path=$2           # e.g. data_generated/wikipedia-20200501.en,bookcorpus
                          #   = where all `train/test_shard_xxx.hdf5` lies
effective_batch_size=$3   # e.g. 4096
warmup_proportion=$4      # e.g. 0.06 (a float between [0, 1]
num_step=$5               # e.g. 1000000
                          #   = Number of steps
micro_batch_size=$6       # e.g. 16 (around 9G GPU memory)
                          #   = the batch will be devided into multiple synced
                          #     micro batches so that we can afford large
                          #     batches with even limited GPU memory.
data_loader_type=$7       # e.g. dist

# ===== Normally changing parameters
prefix=$8                 # e.g. "inverse-time-decay"
init_lr=$9                # e.g. 2e-5
lr_curve_conf_file=${10}  # e.g. conf/lr-scheduler_elastic-step-decay.conf
                          #   = Path to learning rate curve conf file
optimizer_conf_file=${11} # e.g. conf/optimizer_vanilla-sgd.conf
                          #   = Path to optimizer conf file
num_device=${12}

log_dir="log/pretrain/${dataset_name}"
tmp_dir="tmp/pretrain/${dataset_name}"
output_dir="saved_models/pretrain/${dataset_name}/${prefix}"

mkdir -p ${log_dir} ${tmp_dir} ${output_dir}

# Skips experiments if it has been run before
if [ -f ${tmp_dir}/${prefix}.mark ]; then
  exit
fi

# If the experiments was killed by signals like Ctrl+C, remove its skip mark
trap "rm -f ${tmp_dir}/${prefix}.mark" SIGINT
touch "${tmp_dir}/${prefix}.mark"

deepspeed --num_gpus=${num_device} \
  run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --intermediate_size 4096 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr ${init_lr} \
  --train_batch_size ${effective_batch_size} \
  --train_micro_batch_size_per_gpu ${micro_batch_size} \
  --lr_schedule step \
  --curve customized \
  --curve_conf_file ${lr_curve_conf_file} \
  --warmup_proportion ${warmup_proportion} \
  --gradient_clipping 0.0 \
  --optimizer_type customized \
  --optimizer_conf_file ${optimizer_conf_file} \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps ${num_step} \
  --total_training_time 24000.0 \
  --early_exit_time_marker 24000.0 \
  --dataset_path ${dataset_path} \
  --output_dir ${output_dir} \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --job_name pretraining_experiment \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch ${micro_batch_size} \
  --deepspeed \
  --data_loader_type ${data_loader_type} \
  --do_validation \
  --seed 42 \
  --fp16 \
  > ${log_dir}/${prefix}.log \
  2> ${log_dir}/${prefix}.err
