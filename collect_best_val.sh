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

function main() {
  local task_name

  # ===== Test setting
  # local pretrain_dataset_name="bookcorpus-evenly-article-partition"

  # ===== Standard finetune setting
  local pretrain_dataset_name=$1

  local model_set_path="saved_models/pretrain/${pretrain_dataset_name}"

  local epoch_pattern="epoch-.*"

  for task_name in wnli rte mrpc stsb cola sst2 qnli qqp mnli; do
  # for task_name in mrpc; do

    # Collects best validation results
    for model_path in ${model_set_path}/*; do
      local model_name=$(basename ${model_path})
      local input_file="log/finetune/${pretrain_dataset_name}/${task_name}/${model_name}/summary.log"
      local output_file="log/finetune/${pretrain_dataset_name}/${task_name}/${model_name}/best-val.log"

      echo "${task_name}: collect best results to '${output_file}'..." >&2

      # Heading bar: the meaning of metrics
      echo "$(head -1 ${input_file} | tr -d '\n') pretrain_setting" \
        | sed 's/ /    /g' > ${output_file}

      # Best validation results
      cat ${input_file} | grep "${epoch_pattern}" | head -1 >> ${output_file}
    done

    # Displays heading + metrics meanings
    echo "===== ${task_name}"
    for model_path in ${model_set_path}/*; do
      local model_name=$(basename ${model_path})
      local output_file="log/finetune/${pretrain_dataset_name}/${task_name}/${model_name}/best-val.log"
      cat ${output_file} | head -1
      break         # only display metric meanings once
    done

    # Displays best validation results
    for model_path in ${model_set_path}/*; do
      local model_name=$(basename ${model_path})
      local output_file="log/finetune/${pretrain_dataset_name}/${task_name}/${model_name}/best-val.log"
      echo "$(cat ${output_file} | head -2 | tail -1) ${model_name}"
    done
  done
}

main "$@"
