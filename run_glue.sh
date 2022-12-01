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

pretrain_dataset_name=$1    # e.g. wikipedia-20220401.en,bookcorpus
task_name=$2        # e.g. cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli
prefix=$3           # Any string, e.g. epoch-2_batch-size-16_init-lr-5e-5
num_epoch=$4        # e.g. 3
batch_size=$5       # e.g. 16
init_lr=$6          # Initial learning rate, e.g. 5e-5
model_path=$7       # e.g. pretrain/saved_models/xxx/elastic-step-decay_step-based-num-iter-1000_optimizer-adamw_batch-size-16_init-lr-1.0_interval-shrink-rate-1.4142_cr-k-3
model_name=$8       # e.g. elastic-step-decay_step-based-num-iter-1000_optimizer-adamw_batch-size-16_init-lr-1.0_interval-shrink-rate-1.4142_cr-k-3
                    # In our case, it is the basename of ${model_path} and
                    # means the pretrained model's lr scheduler setting
                    # during pretraining
saved_finetune_checkpoint_at_end=$9     # "True" or "False"
prerequisite_file=${10}       # None or a file path, the script will not invoke
                              # `run_glue.py` until this file starts to exist
num_gpu=${11}

if [ ${num_gpu} -ne 1 -a ${num_gpu} -ne 2 -a ${num_gpu} -ne 4 -a ${num_gpu} -ne 8 -a ${num_gpu} -ne 16 ]; then
  echo "$(date): [ERROR] process $$: unsupported number of gpus ${num_gpu}. Valid option: [1, 2, 4, 8, 16]"
  exit 1
fi

per_device_train_batch_size=$(python -c "print(${batch_size} // ${num_gpu})")
per_device_eval_batch_size=16             # Default value in huggingface is 8

shared_dir_suffix="finetune/${pretrain_dataset_name}/${task_name}/${model_name}"
log_dir="log/${shared_dir_suffix}"
tmp_dir="tmp/${shared_dir_suffix}"
output_dir="output/${shared_dir_suffix}/${prefix}"
output_test_dir="output_test/${shared_dir_suffix}"

mkdir -p ${log_dir} ${tmp_dir} ${output_dir} ${output_test_dir}

ongoing_mark_file=${tmp_dir}/${prefix}.ongoing.mark
completed_mark_file=${tmp_dir}/${prefix}.completed.mark

# Skips experiments if it is being executed by another process right now or
# completed before
if [ -f ${ongoing_mark_file} -o -f ${completed_mark_file} ]; then
  exit
fi
trap "rm -rf ${ongoing_mark_file} ${completed_mark_file} ${output_dir}; exit" SIGINT SIGTERM SIGKILL
touch "${ongoing_mark_file}"

echo "$(date): process: $$, task: ${task_name}, num_epoch: ${num_epoch}, batch_size: ${batch_size}, init_lr: ${init_lr} start..."

# Waits until the prerequisite file exists
wait_time=0
check_frequency=30       # unit of measure: seconds
if [ "${prerequisite_file}" != "" ]; then
  while true; do
    if [ -f ${prerequisite_file} ]; then
      if [ ${wait_time} -gt 0 ]; then
        echo "$(date): process $$: prerequisite file completed, start training."
      fi
      break
    fi
    if [ ${wait_time} -eq 0 ]; then
      echo "$(date): process $$: prerequisite file hasn't completed yet, waiting."
    fi

    sleep ${check_frequency}
    wait_time=$(( wait_time + check_frequency ))

    if [ ${wait_time} -eq 21600 ]; then
      echo "$(date): [WARNING] process $$ have waited for at least 6 hours, normally it should take at most 3-4 hours for the corresponding intermediate training to complete. Please check if the intermediate training result for "${prerequisite_file}" is normal, or if the running status is okay."
    fi
  done
fi

# Finds the saved model binary for finetuning
#
#   1) "pretraining_experiment-" = "{job_name}-{current_run_id}" is the default
#      job name "pretraining_experiment" with an empty current run id (default).
#
#   2) "epoch1000000_step-*" = "epoch-{num_epoch}_step-{actual_num_step}" is
#      the model saved at epoch 1000000 (which is the default number of epochs,
#      also can be used to specify the last saved model) with the number of
#      steps actually runs during training.
#
#      Since under some scenarios, "actual_num_step" != "num_step" due to
#      dynamic features of the trainer, e.g. ignoring steps with gradient
#      overflows, which are caused by the dynamic loss scaling feature of mixed
#      precision training. For example, when we have num_step = 1000,
#      actual_num_step can be 1016 due to gradient overflows in first 16 steps.
#
#      But that doesn't matter, since we will only have one such directory, we
#      just choose the first one, which will be the only one.

model_bin_parent_dir="${model_path}/pretraining_experiment-"
model_bin_subdir=$(ls ${model_bin_parent_dir} | grep epoch1000000_step | head -1)
model_bin_path=${model_bin_parent_dir}/${model_bin_subdir}

python run_glue.py \
  --model_name_or_path ${model_bin_path} \
  --task_name ${task_name} \
  --max_seq_length 128 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size ${per_device_eval_batch_size} \
  --learning_rate ${init_lr} \
  --weight_decay 0.1 \
  --logging_strategy steps \
  --evaluation_strategy no \
  --save_strategy no \
  --max_grad_norm 1.0 \
  --num_train_epochs ${num_epoch} \
  --lr_scheduler_type polynomial \
  --warmup_ratio 0.06 \
  --disable_tqdm True \
  --finetune_checkpoint_at_end ${save_finetune_checkpoint_at_end} \
  > ${log_dir}/${prefix}.log \
  2> ${log_dir}/${prefix}.err

# Copies the prediction results to ${output_test_dir}
cp ${output_dir}/predict_results_${task_name}.txt ${output_test_dir}/${prefix}.txt
if [ "${task_name}" = "mnli" ]; then
  cp ${output_dir}/predict_results_${task_name}-mm.txt ${output_test_dir}/${prefix}-mm.txt
fi

# We need to save MNLI models for RTE, MRPC, STSB finetuning
# This is one of the tricks used in 24h-bert
if [ "${task_name}" != "mnli" ]; then
  rm -rf ${output_dir}
fi

touch ${completed_mark_file}

echo "$(date): process: $$, task: ${task_name}, num_epoch: ${num_epoch}, batch_size: ${batch_size}, init_lr: ${init_lr} done"
