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

function parse_args() {
  local help_message=""
  help_message+="Options:\n"
  help_message+="  --debug\n"
  help_message+="    Default: 0 (disabled). Run in debug mode, which only requires on single GPU and much less GPU memory\n"
  help_message+="  --remove-old-record\n"
  help_message+="    Default: 0 (disabled). (Be careful!) Force re-run of all cached experiments by removing their cache marks\n"
  help_message+="  --baseline-only\n"
  help_message+="    Default: 0 (disabled). Only run baseline: linear decay\n"
  help_message+="  --optimizer OPTIMIZER\n"
  help_message+="    Default: \"adamw\". Other options: \"vanilla-sgd\", \"sgd-momentum\"\n"
  help_message+="  --init-lr-search-list \"STR\"\n"
  help_message+="    Default: \"\" (disabled customized lr list and use default setting). Example: \"1e2 1e-3\" (Don't forget the double quotes!)\n"
  help_message+="  --dataset_name  \"STR\"\n"
  help_message+="    Default: wikipedia-20220301.en,bookcorpusopen. An identifier for datasets during constructing intermediate directory, can be arbitrary strings actually\n"
  help_message+="  --dataset_path  \"STR\"\n"
  help_message+="    Default: data/wikipedia-20220301.en,bookcorpus. Path to prepared datasets for pretraining.\n"
  help_message+="  --num_device {NUM_DEVICE}\n"
  help_message+="    Default: 2. Number of gpus in a single machine\n"
  help_message+="  --num_steps {NUM_STEPS}\n"
  help_message+="    Default: 23000. Number of steps with effective batch size\n"
  help_message+="  --prefix {PREFIX}\n"
  help_message+="    Default: \"default\". A prefix that specifies this prerain setting"

  # Parses arguments are passed by global variables
  arg_debug=0
  arg_remove_old_record=0
  arg_baseline_only=0
  arg_optimizer="adamw"
  arg_init_lr_search_list="1e-3"
  arg_dataset_name="wikipedia-20220301.en,bookcorpusopen"
  arg_dataset_path="data/wikipedia-20220301.en,bookcorpus"
  arg_num_device=2
  arg_num_steps=23000
  arg_prefix="default"

  while [[ $# -ge 1 ]]; do
    local key="$1"
    case ${key} in
      -h|--help)
        printf "${help_message}" 1>&2
        return 0
        ;;
      --debug)
        arg_debug=1
        ;;
      --remove-old-record)
        arg_remove_old_record=1
        ;;
      --baseline-only)
        arg_baseline_only=1
        ;;
      --optimizer)
        arg_optimizer="$2"
        shift
        ;;
      --init-lr-search-list)
        arg_init_lr_search_list="$2"
        shift
        ;;
      --dataset_name)
        arg_dataset_name="$2"
        shift
        ;;
      --dataset_path)
        arg_dataset_path="$2"
        shift
        ;;
      --num_device)
        arg_num_device=$2
        shift
        ;;
      --num_steps)
        arg_num_steps=$2
        shift
        ;;
      --prefix)
        arg_prefix="$2"
        shift
        ;;
      *)
        # Ignores other unknown options
    esac
    shift       # Moves to next argument
  done

  # Arguments check
  if [ "${arg_optimizer}" != "adamw" -a \
       "${arg_optimizer}" != "vanilla-sgd" -a \
       "${arg_optimizer}" != "sgd-momentum" ]; then
    printf "Unsupported optimizer \"${arg_optimizer}\" (Supported: [adam|vanilla-sgd|sgd-momentum])\n" 1>&2
    return 1
  fi

  return 0
}


function run_hyperparam_search() {
  local dataset_name=$1
  local dataset_path=$2
  local data_loader_type=$3
  local effective_batch_size=$4
  local warmup_proportion=$5
  local num_step=$6
  local micro_batch_size=$7
  local optimizer_type=$8
  local optimizer_conf_path=$9
  local init_lr_search_list="${10}"
  local num_device=${11}


  # If scheduler needs to be activated after a certain portion of training
  local activation_portion=0     # Schedulers are activated at iteration 0
  local num_restart=0            # No restart of schedulers during the training

  local full_num_iter=${num_step}     # Currently one iteration means one step,
                                      # NOT one micro step.
  local num_iter=$(python3 -c "print(int(${full_num_iter} * (1 - ${warmup_proportion})))")

  local activation_point=$(python3 -c "print(int(${num_iter} * ${activation_portion}))")
  local num_iter=$(python3 -c "print(int(${num_iter} * (1 - ${activation_portion})))")

  # If schedulers needs to be restarted in a cyclic manner
  local restarting_points=0
  if [ ${num_restart} -gt 0 ]; then
    restarting_points=$(python3 -c "print(','.join([str((i + 1) * int(${num_iter} // (${num_restart} + 1))) for i in range(${num_restart})]))")
  fi
  local num_iter_per_restart=$(python3 -c "print(${num_iter} // (${num_restart} + 1))")

  # Search!
  echo "$(date): start training"
  echo "========== 24h-bert =========="
  echo "dataset_name = ${dataset_name}"
  echo "dataset_path = ${dataset_path}"
  echo "optimizer_conf_path = ${optimizer_conf_path}"
  echo "init_lr = ${init_lr_search_list}"

  echo

  echo "effective_batch_size = ${effective_batch_size}"
  echo "micro_batch_size (per device) = ${micro_batch_size}"
  echo "warmup_proportion = ${warmup_proportion}"
  echo "num_iter = ${full_num_iter}"
  echo "num_iter (after warmup) = ${num_iter}"
  echo "num_device = ${num_device}"

  local conf_dir="tmp/conf/${dataset_name}"
  mkdir -p ${conf_dir}

  # Default hyperparameters for Elastic Step Decay scheduler
  interval_shrink_rate=1.4142
  cr_k=6
  init_lr=${init_lr_search_list}
  prefix=${arg_prefix}

  # Prepares optimizer config file
  local conf_path="${conf_dir}/${prefix}.conf"
    cat << EOF > ${conf_path}
[general]
type = elastic_step_decay

[hyperparams]
activation_point = ${activation_point}
restarting_points = ${restarting_points}
num_iter = ${num_iter_per_restart}
interval_shrink_rate = ${interval_shrink_rate}
cr_k = ${cr_k}
EOF

  echo "$(date): training..."
  ./run_pretraining.sh ${dataset_name} ${dataset_path} ${effective_batch_size} ${warmup_proportion} ${num_step} ${micro_batch_size} ${data_loader_type} ${prefix} ${init_lr} ${conf_path} ${optimizer_conf_path} ${num_device}
}


function main() {
  # We call update of each "effective batch size" one **step**, and
  # each computation of "devices * micro batch size" one **micro step**.
  # An update step may contain multiple steps, since it may involve gradient
  # accumulation.
  #
  # Currently our scheduler is **iteration-based** or **step-based**, which is
  # similar to time-based except it is more reproducible.
  #
  # We may support **micro-step-based** schedulers in the future.
  parse_args "$@"
  if [ $? -ne 0 ]; then     # Check return codes of last command
    return 1
  fi

  if [ ${arg_debug} -eq 1 ]; then
    # ===== Debug settings
    local dataset_name="bookcorpus-evenly-article-partition"
    local dataset_path="data_generated/bookcorpus_evenly-article-partition"
    local effective_batch_size=64
    local num_device=1
    local num_step=1000
    local micro_batch_size=8
    local warmup_proportion=0.06
  else
    # ===== Standard pretrain settings
    local dataset_name="${arg_dataset_name}"
    local dataset_path="${arg_dataset_path}"
    local num_device=${arg_num_device}
    local num_step=${arg_num_steps}
    local effective_batch_size=4096
    # local num_step=23000            # 256M / (8*32) / 2.7 days = steps in 1d
                                    # ~= 23k update steps
                                    # p.s. The paper trains 256M samples in 2.7d
    local micro_batch_size=8
    local warmup_proportion=0.06
  fi

  local batch_size_per_step=$(( num_device * micro_batch_size ))

  # Sets optimizers
  local optimizer_type=${arg_optimizer}
  if [ "${optimizer_type}" = "vanilla-sgd" ]; then
    local optimizer_conf_path="conf/optimizer_vanilla-sgd.conf"
    local init_lr_search_list="1e3 1e2 1e1 1e0 1e-1 1e-2 1e-3"
  elif [ "${optimizer_type}" = "sgd-momentum" ]; then
    local optimizer_conf_path="conf/optimizer_sgd-momentum.conf"
    local init_lr_search_list="1e2 1e1 1e0 1e-1 1e-2 1e-3"
  elif [ "${optimizer_type}" = "adamw" ]; then
    local optimizer_conf_path="conf/optimizer_adamw.conf"
    local init_lr_search_list="1e-3"
  fi

  if [ "${arg_init_lr_search_list}" != "" ]; then
    local init_lr_search_list="${arg_init_lr_search_list}"
  fi

  # By default, searched hyperparameters are skipped.
  # But with --remove-old-record, we will remove those records and do the
  # grid search from scratch. Think carefully when you use this option!
  if [ ${arg_remove_old_record} -eq 1 ]; then
    rm tmp/pretrain/${dataset_name}/*.mark
  fi

  if [ ${arg_debug} -eq 1 ]; then
    local data_loader_type="per_device"
  else
    local data_loader_type="dist"
  fi

  run_hyperparam_search \
      "${dataset_name}" \
      "${dataset_path}" \
      ${data_loader_type} \
      ${effective_batch_size} \
      ${warmup_proportion} \
      ${num_step} \
      ${micro_batch_size} \
      ${optimizer_type} \
      ${optimizer_conf_path} \
      "${init_lr_search_list}" \
      ${num_device}
}

main "$@"
