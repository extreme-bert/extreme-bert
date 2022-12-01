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
#
# Assumption: Must be run under the project directory

function run_hyperparam_search() {
  local pretrain_dataset_name=$1
  local task=$2
  local model_set_path=$3
  local num_gpu=$4

  for model_path in ${model_set_path}; do
    # For every pretrained model, we use finetune search to find the best
    # finetune results under standard BERT finetune settings
    local model_name=$(basename ${model_path})

    # Trick of 24h-bert: save finetuned models on MNLI to finetune MRPC, RTE,
    # STSB. Regarding how to select the intermediate finetuning hyperparameter,
    # no completed matched details found in 24h-bert code/paper:
    #   https://arxiv.org/pdf/2104.07705.pdf
    #   https://github.com/IntelLabs/academic-budget-bert
    #
    # or the mentioned STILT code/paper in the previous paper:
    #   https://arxiv.org/pdf/1811.01088.pdf
    #   https://github.com/zphang/bert_on_stilts
    #
    # so I use the closet setting to the STILT:
    #
    #                         STILT paper             24h-bert finetune search
    #   warmup_proportion     0.1                    >0.06
    #   train_eopchs          >3                      {3, 5}
    #   weight_decay          >0.1                    0.1
    #   batch_size            16 or 24 or 32         >32
    #   init_lr               2e-5                    {>5e-5, 8e-5}
    #   lr_schedule           strange warmup_linear  >warmup + linear
    #   optimizer             handcraft AdamW        >AdamW
    #
    if [ "${task_name}" = "mnli" ]; then
      save_finetune_checkpoint_at_end=True
    else
      save_finetune_checkpoint_at_end=False
    fi
    if [ "${task_name}" = "rte" -o "${task_name}" = "mrpc" -o "${task_name}" = "stsb" ]; then
      mnli_setting="epoch-3_batch-size-32_init-lr-5e-5"
      model_path="output/finetune/${pretrain_dataset_name}/mnli/${model_name}/${mnli_setting}"

      # Need this file exists to run, otherwise waits
      prerequisite_file="tmp/finetune/${pretrain_dataset_name}/mnli/${model_name}/${mnli_setting}.completed.mark"
    else
      prerequisite_file=""     # Default: no need for prerequisite file
    fi

    # Search! (With default linear decay lr scheduler)
    if [ "${task_name}" = "mnli" -o "${task_name}" = "qqp" -o "${task_name}" = "qnli" ]; then
      for num_epoch in 3 5; do
        for batch_size in 32; do
          for init_lr in 5e-5 8e-5; do

            prefix="epoch-${num_epoch}_batch-size-${batch_size}_init-lr-${init_lr}"
            ./run_glue.sh \
              ${pretrain_dataset_name} \
              ${task} \
              ${prefix} \
              ${num_epoch} \
              ${batch_size} \
              ${init_lr} \
              ${model_path} \
              ${model_name} \
              ${save_finetune_checkpoint_at_end} \
              "${prerequisite_file}" \
              ${num_gpu}
          done
        done
      done

    else
      # Standard finetune setting in 24h BERT paper (table 7, last page)
      for num_epoch in 3 5 10; do
        for batch_size in 16 32; do
          for init_lr in 1e-5 3e-5 5e-5 8e-5; do

            prefix="epoch-${num_epoch}_batch-size-${batch_size}_init-lr-${init_lr}"
            ./run_glue.sh \
              ${pretrain_dataset_name} \
              ${task} \
              ${prefix} \
              ${num_epoch} \
              ${batch_size} \
              ${init_lr} \
              ${model_path} \
              ${model_name} \
              ${save_finetune_checkpoint_at_end} \
              "${prerequisite_file}" \
              ${num_gpu}
          done
        done
      done
    fi
  done
}

#===============================================================================
# Reference: https://www.tensorflow.org/datasets/catalog/glue
# Reference: https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification
#===============================================================================
function main() {
  local model_path
  local num_epoch
  local batch_size
  local task_name
  local num_sample

  local pretrain_dataset_name="$1"
  local num_gpu=$2
  local prefix=$3       # A string that specifies unique pretrain setting
  local model_set_path="saved_models/pretrain/${pretrain_dataset_name}/${prefix}"

  # By default, searched hyperparameters are skipped.
  # But with --remove-old-record, we will remove those records and do the
  # grid search from scratch. Think carefully when you use this option!
  if [ "$1" = "--remove-old-record" ]; then
    rm tmp/finetune/${pretrain_dataset_name}/*/*/*.mark
  fi

  # Number of samples in each task:
  #   wnli:   635
  #   rte:    2490
  #   mrpc:   3668
  #   stsb:   5749
  #   cola:   8551
  #   sst2:   67349
  #   qnli:   104743
  #   qqp:    363846
  #   mnli:   392702
  for task_name in mnli qqp qnli sst2 cola stsb mrpc rte wnli; do
  # for task_name in wnli; do
    run_hyperparam_search \
      ${pretrain_dataset_name} \
      ${task_name} \
      ${model_set_path} \
      ${num_gpu}
  done

}

main "$@"
