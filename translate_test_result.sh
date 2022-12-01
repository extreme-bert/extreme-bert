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
# Translate prediction results to submit format.

function translate() {
  local task=$1
  local input_file=$2
  local output_file=$3

  if [ "${task}" = "mrpc" ]; then
    cat ${input_file} \
        | sed 's/not_equivalent/0/' \
        | sed 's/equivalent/1/' \
        > ${output_file}
  elif [ "${task}" = "cola" ]; then
    cat ${input_file} \
        | sed 's/unacceptable/0/' \
        | sed 's/acceptable/1/' \
        > ${output_file}
  elif [ "${task}" = "sst2" ]; then
    cat ${input_file} \
        | sed 's/negative/0/' \
        | sed 's/positive/1/' \
        > ${output_file}
  elif [ "${task}" = "stsb" ]; then
    # Truncates value to no larger than 5, as required by the GLUE server
    cat ${input_file} \
      | awk 'NR > 1 { value=$2; if (value > 5) { value = 5 } print $1"\t"value } NR == 1 { print $0 }' \
      > ${output_file}
  elif [ "${task}" = "qqp" ]; then
    cat ${input_file} \
        | sed 's/not_duplicate/0/' \
        | sed 's/duplicate/1/' \
        > ${output_file}
  elif [ "${task}" = "mnli" ]; then
    cp ${input_file} ${output_file}
  elif [ "${task}" = "qnli" ]; then
    cp ${input_file} ${output_file}
  elif [ "${task}" = "rte" ]; then
    cp ${input_file} ${output_file}
  elif [ "${task}" = "wnli" ]; then
    cat ${input_file} \
        | sed 's/not_entailment/0/' \
        | sed 's/entailment/1/' \
        > ${output_file}
  fi
}


function main() {
  # ===== Test setting
  # local pretrain_dataset_name="bookcorpus-evenly-article-partition"

  # ===== Standard finetune setting
  local pretrain_dataset_name=$1
  local prefix=$2

  local model_set_path="saved_models/pretrain/${pretrain_dataset_name}"

  # Untranslated outputed prediction files
  local input_dir="output_test/finetune/${pretrain_dataset_name}"
  # Translated
  local shared_output_dir="output_test_translated/finetune/${pretrain_dataset_name}"

  # Translates all test files to the given format and zip them
  for model_path in ${model_set_path}/${prefix}; do
    local model_name=$(basename ${model_path})

    local output_dir=${shared_output_dir}/${model_name}
    mkdir -p ${output_dir}

    # A standard AX.tsv, since we don't do this task
    cp conf/glue_example/AX.tsv ${output_dir}/

    # Gets best validation result, based on that to select prediction file
    for task in wnli rte mrpc stsb cola sst2 qnli qqp mnli; do
    # for task in wnli; do
    # for task in wnli rte mrpc stsb cola sst2 qnli; do
      # Chooses the file which corresponds to the best validation result
      #
      # > Example of a 'best-val.log' (excluding the lines of """)
      # """
      # eval_f1    eval_accuracy    train_loss    finetune_file    pretrain_setting
      # 0.8097 0.7108 0.3694 epoch-4_batch-size-16_init-lr-5e-5.log
      # """
      #
      # This should outputs 'epoch-4_batch-size-16_init-lr-5e-5' for the example
      local log_dir="log/finetune/${pretrain_dataset_name}/${task}/${model_name}"
      local prefix_for_best_val=$(cat ${log_dir}/best-val.log \
          | tail -1 \
          | awk '{ print $NF }' \
          | sed 's/\.log-err$//')

      # Translates the .txt prediction of huggingface to .tsv for GLUE server
      local input_prefix="${input_dir}/${task}/${model_name}/${prefix_for_best_val}"
      if [ "${task}" != "mnli" ]; then
        # Locates input file
        local input_file="${input_prefix}.txt"
        if [ ! -f ${input_file} ]; then
          echo "Error: no prediction result for pretrained model '${model_name}' on task '${task}' for pretrained dataset '${pretrain_dataset_name}'" 1>&2
          continue
        fi

        # Decides correct naming for output file
        local task_formatted=$(printf ${task} \
            | sed 's/wnli/WNLI/' \
            | sed 's/rte/RTE/' \
            | sed 's/mrpc/MRPC/' \
            | sed 's/stsb/STS-B/' \
            | sed 's/cola/CoLA/' \
            | sed 's/sst2/SST-2/' \
            | sed 's/qnli/QNLI/' \
            | sed 's/qqp/QQP/')
        local output_file="${output_dir}/${task_formatted}.tsv"
        translate ${task} ${input_file} ${output_file}

      else
        # Special treatment for task "mnli" since it has two prediction files
        local input_matched_file="${input_prefix}.txt"
        local input_mismatched_file="${input_prefix}-mm.txt"

        if [ ! -f ${input_matched_file} -o ! -f ${input_mismatched_file} ]; then
          if [ ! -f ${input_matched_file} ]; then
            echo "Error: no matched prediction result for pretrained model '${model_name}' on task '${task}' for pretrained dataset '${pretrain_dataset_name}'" 1>&2
          fi
          if [ ! -f ${input_mismatched_file} ]; then
            echo "Error: no mismatched prediction result for pretrained model '${model_name}' on task '${task}' for pretrained dataset '${pretrain_dataset_name}'" 1>&2
          fi
          continue
        fi

        local output_matched_file="${output_dir}/MNLI-m.tsv"
        local output_mismatched_file="${output_dir}/MNLI-mm.tsv"

        translate ${task} ${input_matched_file} ${output_matched_file}
        translate ${task} ${input_mismatched_file} ${output_mismatched_file}
      fi
    done

    # Finally, compresses the file
    cd ${shared_output_dir}
    zip ${model_name}.zip ${model_name}/* > /dev/null
    cd - > /dev/null

    local output_zip="${output_dir}.zip"
    echo "Output translated results to '${output_zip}'"
  done
}

main "$@"
