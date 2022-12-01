# coding=utf-8
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
""" The main pipeline:
  prepare data -> pretraining -> finetuning -> test result collection
"""
import argparse
import filecmp
import os
import pathlib
import shutil
import subprocess
import sys
import time

from yacs.config import CfgNode as ConfigNode


HUGGINGFACE_DATASET_LIST_FILE = 'huggingface_dataset_list.txt'
CUSTOMIZED_DATASET_LIST_FILE = 'customized_dataset_list.txt'


def get_default_config():
    config = ConfigNode()
    config.SYSTEM = ConfigNode()
    config.SYSTEM.NUM_GPUS = 4
    config.SYSTEM.MAX_MEMORY_IN_GB = 16

    config.WANDB = ConfigNode()
    config.WANDB.API_KEY = 'a8c4ad2a085ea623fb44f0a5558cd9b4f7ebf7b1'

    config.DATASET = ConfigNode()
    config.DATASET.ENABLED = True
    config.DATASET.HUGGINGFACE_DATASETS = []
    config.DATASET.CUSTOMIZED_DATASETS = []
    config.DATASET.ID = None
    config.DATASET.TMP_DIR = None
    config.DATASET.OUTPUT_DIR = None
    config.DATASET.LOG_DIR = 'log/dataset'

    config.PRETRAIN = ConfigNode()
    config.PRETRAIN.ENABLED = True
    config.PRETRAIN.OPTIMIZER = 'adamw'
    config.PRETRAIN.NUM_STEPS = 23000
    config.PRETRAIN.LEARNING_RATE = 1e-3
    config.PRETRAIN.ID = None

    config.FINETUNE = ConfigNode()
    config.FINETUNE.ENABLED = True
    config.FINETUNE.MULTIPROCESS_GPU_LIST = None

    config.RESULT_COLLECTION = ConfigNode()
    config.RESULT_COLLECTION.ENABLED = True

    config.TOKENIZER = ConfigNode()
    config.TOKENIZER.NAME_OR_PATH = "bert-large-uncased"

    return config.clone()


def get_md5sum_of_file(path):
    stdout_content = subprocess.check_output(['md5sum', f'{path}'])
    # e.g. "53f31ebaf51cfa144ada1affe63807c9  example.txt"
    md5sum = stdout_content.decode(sys.stdout.encoding).split(' ')[0].strip()
    return md5sum


def get_md5sum_of_str(string):
    # `echo ${string} | md5sum`
    process = subprocess.Popen(['echo', f'"{string}"'], stdout=subprocess.PIPE)
    stdout_content = subprocess.check_output('md5sum', stdin=process.stdout)
    process.wait()

    md5sum = stdout_content.decode(sys.stdout.encoding).split(' ')[0].strip()
    return md5sum


def get_dataset_list(config):
    customized_data_list = []
    for dataset_dir in config.DATASET.CUSTOMIZED_DATASETS:
        file_list = sorted(os.listdir(dataset_dir))
        for data_file in file_list:
            if data_file.endswith('.txt'):
                data_path = os.path.join(dataset_dir, data_file)
                customized_data_list.append(f'{data_path}')

    huggingface_data_list = []
    for dataset_name, subset_name in config.DATASET.HUGGINGFACE_DATASETS:
        huggingface_data_list.append(f'{dataset_name}.{subset_name}')

    return customized_data_list, huggingface_data_list


def read_str_from_file(file_path):
    file_content = ''
    if file_path.is_file():
        with open(file_path, 'r') as fin:
            file_content = fin.read()
    return file_content


def write_str_to_file(string, file_path):
    with open(file_path, 'w') as fout:
        fout.write(string)


def is_same_file_list(list_a, list_b):
    if len(list_a) != len(list_b):
        return False

    for file_a, file_b in zip(list_a, list_b):
        is_file_same = filecmp.cmp(file_a, file_b, shallow=False)
        if not is_file_same:
            return False
    return True


def set_autogen_dataset_id(config):
    counter = -1    # Adds counter until there is no conflict IDs
                    # Normally the hash conflict probability is extremely low

    while True:
        counter += 1

        # Get customized & huggingface dataset list
        customized_data_list, huggingface_data_list = get_dataset_list(config)
        customized_data_str = ';'.join(customized_data_list)
        huggingface_data_str = ';'.join(huggingface_data_list)

        # Gets auto-generated ID
        hash_list = []
        for data_path in customized_data_list:
            hash_id = get_md5sum_of_file(data_path)
            hash_list.append(hash_id)

        hash_list.append(get_md5sum_of_str(huggingface_data_str))
        hash_list.append(get_md5sum_of_str(str(counter)))   # Avoid conflicts

        hash_str = ''.join(hash_list)
        final_hash = get_md5sum_of_str(hash_str)        # Two-layer md5sum

        config.DATASET.ID = final_hash
        config.DATASET.TMP_DIR = f'tmp/dataset/{config.DATASET.ID}'
        config.DATASET.OUTPUT_DIR = f'data/{config.DATASET.ID}'

        # Checks if the ID is already used
        project_dir = get_script_dir()
        tmp_dir = pathlib.Path(project_dir, config.DATASET.TMP_DIR)
        output_dir = pathlib.Path(project_dir, config.DATASET.OUTPUT_DIR)

        huggingface_file = pathlib.Path(tmp_dir, HUGGINGFACE_DATASET_LIST_FILE)
        customized_file = pathlib.Path(tmp_dir, CUSTOMIZED_DATASET_LIST_FILE)

        if not tmp_dir.is_dir() and not output_dir.is_dir():
            # Not used, new dataset group, generates dataset list file
            tmp_dir.mkdir(parents=True, exist_ok=True)
            write_str_to_file(huggingface_data_str, huggingface_file)
            write_str_to_file(customized_data_str, customized_file)
            break

        # ID used, check if it was used by the same dataset group
        old_datalist_str = read_str_from_file(huggingface_file)
        if old_datalist_str != huggingface_data_str:
            continue        # ID used by a different dataset group

        old_datalist_str = read_str_from_file(customized_file)
        old_data_list = old_datalist_str.split(';')
        old_data_list = [ path for path in old_data_list if path != '' ]

        is_same = is_same_file_list(old_data_list, customized_data_list)
        if not is_same:
            continue        # ID used by a different dataset group
        else:
            break           # ID used by the same dataset group

    return config


def setup_config(args):
    config = get_default_config()
    config.merge_from_file(args.config_file)

    if config.DATASET.ID is None:
        config = set_autogen_dataset_id(config)
    else:
        config.DATASET.TMP_DIR = f'tmp/dataset/{config.DATASET.ID}'
        config.DATASET.OUTPUT_DIR = f'data/{config.DATASET.ID}'

    if config.PRETRAIN.ID is None:
        config.PRETRAIN.ID = f'esd_optimizer-{config.PRETRAIN.OPTIMIZER}'
        config.PRETRAIN.ID += f'_num-iter-{config.PRETRAIN.NUM_STEPS}'
        config.PRETRAIN.ID += f'_lr-{config.PRETRAIN.LEARNING_RATE}'

    config.freeze()

    if config.WANDB.API_KEY is None:
        raise ValueError(
            'WANDB.API_KEY not provided, '
            'please see "https://docs.wandb.ai/quickstart" for more details'
        )
    return config


def get_date():
    return subprocess.check_output('date').decode(sys.stdout.encoding).strip()


def logging(message):
    print(f'{get_date()}: ' + message, flush=True)


def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))


def run_bash(command):
    process = subprocess.run(
        command,
        shell=True,
        executable='/bin/bash',
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )
    return process


def prepare_dataset(config, args=None):
    if not config.DATASET.ENABLED:
        return

    logging('########## prepare dateset start...')
    project_dir = get_script_dir()

    # The prepared dataset for pretraining is stored in {output_dir}
    output_dir = pathlib.Path(project_dir, config.DATASET.OUTPUT_DIR)
    tmp_dir = pathlib.Path(project_dir, config.DATASET.TMP_DIR, 'content')
    log_dir = pathlib.Path(project_dir, config.DATASET.LOG_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # If the dataset file is prepared, skip
    skip_mark_file = pathlib.Path(tmp_dir, f'{config.DATASET.ID}.mark')
    if skip_mark_file.is_file():
        if args is None or not args.clear_cache:
            logging(
                f'Dataset for this ID "{config.DATASET.ID}"'
                ' has already been prepared, skip preprocessing...'
            )
            logging('########## prepare dateset end')
            return

    # Clear the temporary directory and output directory
    shutil.rmtree(output_dir)
    shutil.rmtree(tmp_dir)

    os.chdir('dataset')     # Goto {project_dir}/dataset

    logging('===== data sharding start...')
    shard_data_command = [
        'python shard_data.py',
        '  --num_train_shards 256',
        '  --num_test_shards 128',
        '  --frac_test 0.1',
        f'  --output_dir {tmp_dir}',
        f'  --max_memory {config.SYSTEM.MAX_MEMORY_IN_GB}',
    ]
    for dataset_name, subset_name in config.DATASET.HUGGINGFACE_DATASETS:
        shard_data_command.append(f'  --dataset {dataset_name} {subset_name}')
    for dataset_dir in config.DATASET.CUSTOMIZED_DATASETS:
        shard_data_command.append(f'  --dataset custom {dataset_dir}')

    shard_data_command.extend([
        f'  > {log_dir}/shard_data.log',
        f'  2> {log_dir}/shard_data.err'
    ])

    logging('See {log_dir}/shard_data.[log|err] for detailed stdout/stderr')
    run_bash(''.join(shard_data_command))
    logging('===== data sharding end...')

    logging('===== sample generation start...')
    tokenizer_name = config.TOKENIZER.NAME_OR_PATH

    generate_sample_command = [
        'python generate_samples.py',
        '  --dup_factor 10',
        '  --seed 42',
        '  --do_lower_case 1',
        '  --masked_lm_prob 0.15',
        '  --max_seq_length 128',
        '  --model_name bert-large-uncased',
        '  --max_predictions_per_seq 20',
        '  --n_processes 8',
        f'  --dir {tmp_dir}',
        f'  -o {output_dir}',
        f'  --tokenizer_name {tokenizer_name}',
        f'  > {log_dir}/generate_sample.log',
        f'  2> {log_dir}/generate_sample.err',
    ]

    logging(f'See {log_dir}/generate_sample.[log|err] for detailed'
             ' stdout/stderr')
    run_bash(''.join(generate_sample_command))
    logging('===== sample generation end')

    os.chdir('..')     # Goes back {project_dir}

    # Creates skip mark
    skip_mark_file.touch()
    logging('########## prepare dateset end')


def pretrain(config, args=None):
    if not config.PRETRAIN.ENABLED:
        return

    logging('########## pretrain start...')
    project_dir = get_script_dir()
    dataset_path = pathlib.Path(project_dir, config.DATASET.OUTPUT_DIR)

    import wandb
    wandb.login(key=config.WANDB.API_KEY)

    pretrain_command = [
        './pretrain_search.sh',
        f'  --dataset_name {config.DATASET.ID}',
        f'  --dataset_path {dataset_path}',
        f'  --num_device {config.SYSTEM.NUM_GPUS}',
        f'  --init-lr-search-list "{config.PRETRAIN.LEARNING_RATE}"',
        f'  --optimizer {config.PRETRAIN.OPTIMIZER}',
        f'  --num_steps {config.PRETRAIN.NUM_STEPS}',
        f'  --prefix {config.PRETRAIN.ID}',
    ]

    if args and args.clear_cache:
        pretrain_command.append(' --remove-old-record')

    logging(f'See {project_dir}/log/pretrain/{config.DATASET.ID}'
            f'/{config.PRETRAIN.ID}.[log|err] for detailed stdout/stderr')
    run_bash(''.join(pretrain_command))
    logging('########## pretrain end')


def finetune(config, args=None):
    if not config.FINETUNE.ENABLED:
        return

    project_dir = get_script_dir()
    tmp_dir = pathlib.Path(project_dir, f'tmp/finetune/{config.DATASET.ID}')
    log_dir = pathlib.Path(project_dir, f'log/finetune/{config.DATASET.ID}')

    if args and args.clear_cache:
        shutil.rmtree(tmp_dir)
        shutil.rmtree(log_dir)

    logging('########## finetune start...')
    logging(f'See {log_dir}/*/{config.PRETRAIN.ID}/*.[log|err]'
             ' for detailed stdout/stderr')

    dataset_name = config.DATASET.ID
    num_gpu = config.SYSTEM.NUM_GPUS
    pretrain_id = config.PRETRAIN.ID
    if config.FINETUNE.MULTIPROCESS_GPU_LIST is None:        # Single process
        run_bash(f'./finetune_search.sh {dataset_name} {num_gpu} {pretrain_id}')
    else:
        # Checks configuration before spawning child processes
        for gpu_list in config.FINETUNE.MULTIPROCESS_GPU_LIST:
            for gpu in gpu_list:
                if gpu < 0 or gpu >= num_gpu:
                    raise ValueError(
                        f'gpu id {gpu} not in 0-{num_gpu - 1}'
                    )

        # Spawns child processes one by one
        process_list = []
        for gpu_list in config.FINETUNE.MULTIPROCESS_GPU_LIST:
            num_gpu_this_proc = len(gpu_list)

            command = 'export CUDA_VISIBLE_DEVICES='
            command += ','.join([str(gpu) for gpu in gpu_list])
            command += ('; ./finetune_search.sh'
                        f' {dataset_name} {num_gpu_this_proc} {pretrain_id}')

            process = subprocess.Popen(
                command,
                shell=True,
                executable='/bin/bash',
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            process_list.append(process)
            time.sleep(10)

        return_code = 0
        for child_process in process_list:
            child_return_code = child_process.wait()
            if child_return_code != 0:
                return_code = 1

        # Raises error only after all child process ends
        if return_code != 0:
            raise RuntimeError('Some child finetune processes run into error!')

    logging('########## finetune end')


def collect_test_result(config):
    if not config.RESULT_COLLECTION.ENABLED:
        return

    logging('########## collect test result start...')
    dataset_name = config.DATASET.ID
    pretrain_id = config.PRETRAIN.ID
    run_bash(f'./summarize_val.sh {dataset_name} {pretrain_id}')
    run_bash(f'./collect_best_val.sh {dataset_name} {pretrain_id}')
    run_bash(f'./translate_test_result.sh {dataset_name} {pretrain_id}')
    logging('########## collect test result end...')


def main(args):
    """The main pipeline."""
    config = setup_config(args)
    prepare_dataset(config, args)
    pretrain(config, args)
    finetune(config, args)
    collect_test_result(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        required=True,
        type=str,
        default='configs/24h-bert.yaml',
        help='path to config file'
    )
    parser.add_argument(
        '--clear_cache',
        default=False,
        action='store_true'
    )
    args = parser.parse_args()
    main(args)
