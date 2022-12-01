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
# Summarizes validation results to a summary file. The output file is sorted
# by the first column in descending order.

function collect_validation_result() {
  # ===== Reported results
  # MRPC  F1/Accuracy 88.85/84.07 2:21
  # CoLA  Matthews corr 56.53 3:17
  # SST-2 Accuracy  92.32 26:06
  # STS-B Pearson/Spearman corr.  88.64/88.48 2:13
  # QQP Accuracy/F1 90.71/87.49 2:22:26
  # MNLI  Matched acc./Mismatched acc.  83.91/84.10 2:35:23
  # QNLI  Accuracy  90.66 40:57
  # RTE Accuracy  65.70 57
  # WNLI  Accuracy  56.34 24

  local task_name=$1
  local log_dir=$2
  local result_file=$3

  if [ "${task_name}" = "mrpc" ]; then
    echo "eval_f1 eval_accuracy eval_loss finetune_file" > ${result_file}
  elif [ "${task_name}" = "cola" ]; then
    echo "eval_matthews_correlation eval_loss finetune_file" > ${result_file}
  elif [ "${task_name}" = "stsb" ]; then
    echo "eval_pearson eval_spearmanr eval_loss finetune_file" > ${result_file}
  elif [ "${task_name}" = "qqp" ]; then
    echo "eval_f1 eval_accuracy eval_loss finetune_file" > ${result_file}
  elif [ "${task_name}" = "mnli" ]; then
    echo "eval_matched_accuracy eval_loss finetune_file" > ${result_file}
  else
    echo "eval_accuracy eval_loss finetune_file" > ${result_file}
  fi

  for file_err_path in ${log_dir}/*.err; do
    local file_prefix=$(echo ${file_err_path} | sed 's/\.err//g')
    cat ${file_prefix}.log ${file_prefix}.err > ${file_prefix}.log-err
    local file_path=${file_prefix}.log-err

    # Collects relevant statisitcs (can be missed if not available)
    local eval_loss=$(cat ${file_path} | grep -F "***** eval metrics *****" -A 11 | grep "eval_loss  " | awk '{ print $NF }' | sed -e "s/\r//")
    local eval_f1=$(cat ${file_path} | grep -F "***** eval metrics *****" -A 11 | grep "eval_f1  " | awk '{ print $NF }' | sed -e "s/\r//")
    local eval_accuracy=$(cat ${file_path} | grep -F "***** eval metrics *****" -A 11 | grep "eval_accuracy  " | awk '{ print $NF }' | sed -e "s/\r//")
    local eval_matthews_correlation=$(cat ${file_path} | grep -F "***** eval metrics *****" -A 11 | grep "eval_matthews_correlation " | awk '{ print $NF }' | sed -e "s/\r//")
    local eval_pearson=$(cat ${file_path} | grep -F "***** eval metrics *****" -A 11 | grep "eval_pearson  " | awk '{ print $NF }' | sed -e "s/\r//")
    local eval_spearmanr=$(cat ${file_path} | grep -F "***** eval metrics *****" -A 11 | grep "eval_spearmanr  " | awk '{ print $NF }' | sed -e "s/\r//")
    local eval_matched_accuracy=$(cat ${file_path} | grep -F "***** eval metrics *****" -A 11 | grep "eval_accuracy  " | awk '{ print $NF }' | sed -e "s/\r//")

    local file=$(basename ${file_path})

    if [ "${task_name}" = "mrpc" ]; then
      echo "${eval_f1} ${eval_accuracy} ${eval_loss} ${file}"
    elif [ "${task_name}" = "cola" ]; then
      echo "${eval_matthews_correlation} ${eval_loss} ${file}"
    elif [ "${task_name}" = "stsb" ]; then
      echo "${eval_pearson} ${eval_spearmanr} ${eval_loss} ${file}"
    elif [ "${task_name}" = "qqp" ]; then
      echo "${eval_f1} ${eval_accuracy} ${eval_loss} ${file}"
    elif [ "${task_name}" = "mnli" ]; then
      echo "${eval_matched_accuracy} ${eval_loss} ${file}"
    else
      echo "${eval_accuracy} ${eval_loss} ${file}"
    fi
  done | sort -rn >> ${result_file}
}

function main() {
  local task_name

  # ===== Test setting
  # local pretrain_dataset_name="bookcorpus-evenly-article-partition"

  # ===== Standard finetune setting
  local pretrain_dataset_name=$1
  local prefix=$2

  local model_set_path="saved_models/pretrain/${pretrain_dataset_name}"

  for model_path in ${model_set_path}/${prefix}; do
    for task_name in wnli rte mrpc stsb cola sst2 qnli qqp mnli;do
    # for task_name in wnli;do
      local model_name=$(basename ${model_path})
      local log_dir="log/finetune/${pretrain_dataset_name}/${task_name}/${model_name}"
      local result_file="${log_dir}/summary.log"
      echo "${task_name}: summarize results to '${result_file}'..." >&2
      collect_validation_result ${task_name} ${log_dir} ${result_file}
    done
  done
}

main "$@"
