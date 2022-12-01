# coding=utf-8
# Copyright 2022 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
# code taken from commit: ea000838156e3be251699ad6a3c8b1339c76e987
# https://github.com/IntelLabs/academic-budget-bert
# Copyright 2021 Intel Corporation. All rights reserved.
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


import argparse
import os
import pathlib
import sys
import textwrap

from argparse import RawTextHelpFormatter
from os import listdir
from os.path import join
from xmlrpc.client import boolean
import random
import time
from datasets import load_dataset

from data import TextSharding
import psutil
from tqdm import tqdm


TOTAL_UNIT = 100
STAGE_NUM = 4
STAGE_UNIT = TOTAL_UNIT/STAGE_NUM
TMP_DIR = '../tmp/dataset'
COMMUN_PATH = TMP_DIR+'/commun.log'



def get_current_memory_gb(): 
    '''gets the memory usage of the process'''
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. / 1024.

def parse_args(cmdline_argv):
    '''argument parsing'''
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--num_train_shards', type=int, default=256, help='Number of training shards')
    parser.add_argument('--num_test_shards', type=int, default=256, help='Number of test shards')
    parser.add_argument('--frac_test', type=float, default=0.1, help='Fraction of dataset to reserve for the test set')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory where the shard files will be written')
    parser.add_argument('--huggingface_cache_dir', type=str, default='~/.cache/huggingface/datasets/', help='The cache dir path to save huggingface dataset')
    parser.add_argument('--max_memory', type=float, default=64, help='The max memory allocated to the process (GB)')
    parser.add_argument('--verbose', type=boolean, default=False, help='Print all logs')
    parser.add_argument('--n_machine', type=int, default=1, help='Number of machines')
    parser.add_argument('--master', type=boolean, default=False, help='Assign a machine to delete communication files')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        nargs=2,
        action='append',
        help=(textwrap.dedent("""
            Add a dataset with one of the following two types:
            (1) Huggingface datasets (2) Customized datasets
            -----
            (1) For Huggingface datasets, use following format:
                `--dataset {huggingface_dataset_name} {subset_name}`,
            e.g.
                `--dataset wikipedia 20220301.en`
                `--dataset bookcorpusopen plain_text`
            For more details, please refer to https://huggingface.co/datasets/*,
            e.g.
                https://huggingface.co/datasets/bookcorpusopen
            -----
            (2) For customized datasets, use following format:
                `--dataset custom {custom_path}`,
            e.g.
                `--dataset custom ~/data/custom/`,
            where "~/data/custom/" stores the dataset files, e.g.
                ```
                ~/data/custom
                    |- science.txt
                    |- nature.txt
                    |- ...
                ```
            with all file follows the format below:
                ```
                {segment_1,1}\\n{segment_1,2}\\n...{segment_1,n1}

                {segment_2,1}\\n{segment_2,2}\\n...{segment_2,n2}

                ...
                ```
            Here each chunk can contain a single sentence, multiple sentences,
            or a full article. The sharding process will split only based on
            double newline chararacters '\\n\\n' and treat segment_{i,*} as a
            whole chunk.
        """)),
    )

    args = parser.parse_args(cmdline_argv)
    return args

def distribution():
    '''distributes data to certain number of partitions'''
    print("Distribute dataset")
    n_machine = args.n_machine
    print(f"Distributing datasets into {n_machine} files")
    shards_dir = pathlib.Path(args.output_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    input_files = []
    
    if not args.verbose:
        total_tqdm = tqdm(total=TOTAL_UNIT)
    else:
        total_tqdm = None

    custom_counter = 0
    for dataset_tuple in args.dataset:
        dataset_name = dataset_tuple[0]
        # print(f'Processing {dataset_name}...')
        len_between = STAGE_UNIT/len(args.dataset)
        

        if dataset_name == 'custom':        # ===== Customized datasets
            dataset_dir = pathlib.Path(dataset_tuple[1])
            if not dataset_dir.is_dir():
                raise RuntimeError('The dataset directory does not exist')

            download_dir= pathlib.Path(args.output_dir, 'custom')
            download_dir.mkdir(parents=True, exist_ok=True)

            download_file_list = []
            for i in range(n_machine):
                filename = '%08d.%d.txt' % (custom_counter, i)
                download_file = pathlib.Path(download_dir, filename)
                download_file_list.append(download_file)

            input_file_list = [join(dataset_dir, f) for f in listdir(dataset_dir) if f.endswith('.txt')]

            # Removes '\n' in chunks and separate chunks with '\n\n'
            for i, download_file in enumerate(download_file_list):
                with open(download_file, mode='w', encoding='utf-8') as fout:
                    interval = len_between/len(input_file_list)
                    if args.verbose:
                        input_file_list = tqdm(input_file_list)
                    j = 0
                    for count, input_file in enumerate(input_file_list):
                        if args.verbose:
                            input_file_list.set_description(f"Processing files in {dataset_name}")
                        else:
                            total_tqdm.set_description("Processing files in custom")
                            total_tqdm.update(interval)
                            
                        with open(input_file, mode='r', encoding='utf-8') as fin:
                            prev_line = ''
                            
                            for line in fin:
                                if j%n_machine == i:
                                    if line == '\n' and prev_line != '\n':
                                        fout.write('\n\n')
                                        j += 1
                                    elif line != '\n':
                                        fout.write(line.replace('\n', ''))
                                    prev_line = line
                                else:
                                    if line == '\n' and prev_line != '\n':
                                        j += 1
                                    prev_line = line

                        if (prev_line != '\n') and (j%n_machine == i):
                            fout.write('\n\n')
                            j += 1

                        if (prev_line != '\n') and (j%n_machine != i):
                            j += 1

                input_files += [download_file]
            custom_counter += 1

        else:       # ===== Huggingface datasets
            dataset_subset = dataset_tuple[1]
            dataset = load_dataset(dataset_name, dataset_subset, cache_dir=args.huggingface_cache_dir)
            download_dir = pathlib.Path(args.output_dir, f'{dataset_name}')
            download_dir.mkdir(parents=True, exist_ok=True)
            download_file_list=[]
            for i in range(n_machine):
                filename = f'{dataset_subset}.{i}.txt'
                download_file = pathlib.Path(download_dir, filename)
                download_file_list.append(download_file)

            interval = len_between/len(dataset['train'])
            count_tqdm = 0
            for i, download_file in enumerate(download_file_list):
                with open(download_file, mode='w', encoding='utf-8') as fout:
                    huggingface_tqdm = tqdm(dataset['train']) if args.verbose else dataset['train']
                    for count, article in enumerate(huggingface_tqdm):
                        if count%n_machine == i:
                            count_tqdm += 1
                            if args.verbose:
                                huggingface_tqdm.set_description(f"Processing articles in {dataset_name}")
                            else:
                                total_tqdm.update(interval)
                                total_tqdm.set_description(f"Processing articles in {dataset_name}")
                            fout.write(article['text'].replace('\n\n', ' ').replace('\n', '') + '\n\n')
                        
                input_files += [download_file]

    with open(TMP_DIR+'/file_list.txt', mode='w', encoding='utf-8') as ostream:
        for file in input_files:
            ostream.write(str(file)+'\n')

    if not args.verbose:
        if total_tqdm.n < STAGE_UNIT:
            total_tqdm.update(STAGE_UNIT-total_tqdm.n)
            total_tqdm.set_description("Finish processing articles")


if __name__ == '__main__':
    time.sleep(random.random()*10) # avoid deadlock

    args = parse_args(sys.argv[1:])

    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    
    if (args.master and os.path.exists(COMMUN_PATH)) or (args.n_machine == 1 and os.path.exists(COMMUN_PATH)):
        os.remove(COMMUN_PATH)
        print('Removed communication file')


    if not os.path.exists(COMMUN_PATH):
        with open(COMMUN_PATH, mode='w', encoding='utf-8') as ostream:
            ostream.write('')
    with open(COMMUN_PATH, mode='r', encoding='utf-8') as istream:
        text = istream.readlines()

    # determine master machine and distribute the data
    if ('Distributing Dataset\n' not in text):
        with open(COMMUN_PATH, mode='w', encoding='utf-8') as ostream:
            ostream.write('Distributing Dataset\n')
            ostream.write('0\n')
        machine_id = 0
        distribution()
        with open(COMMUN_PATH, mode='a', encoding='utf-8') as ostream:
            ostream.write('Completed Distribution\n')

    # if distribution of data is completed, move to next step
    elif ('Distributing Dataset\n' in text) and ('Completed Distribution\n' in text):
        for i in range(args.n_machine):
            if f'{i}\n' not in text:
                with open(COMMUN_PATH, mode='a', encoding='utf-8') as ostream:
                    ostream.write(f'{i}\n')
                machine_id = i
                break

    # if distribution of data is not completed, other machines should wait until done
    elif ('Distributing Dataset\n' in text) and ('Completed Distribution\n' not in text):
        for i in range(args.n_machine):
            if f'{i}\n' not in text:
                with open(COMMUN_PATH, mode='a', encoding='utf-8') as ostream:
                    ostream.write(f'{i}\n')
                machine_id = i
                break

        print('Waiting master to distribute data...')
        while True:
            time.sleep(10)
            with open(COMMUN_PATH, mode='r', encoding='utf-8') as istream:
                text = istream.readlines()
                if ('Distributing Dataset\n' in text) and ('Completed Distribution\n' in text):
                    break

    total_tqdm = tqdm(total = TOTAL_UNIT)
    total_tqdm.update(STAGE_UNIT)

    with open(TMP_DIR+'/file_list.txt', mode='r', encoding='utf-8') as istream:
        file_list = istream.readlines()
    file_list = [file.replace('\n','') for file in file_list]
    input_files = [file for file in file_list if int(file.split('.')[-2]) == machine_id]
    shards_dir = pathlib.Path(args.output_dir)

    # distribute different data partitions to different machines
    train_shards_per_machine = args.num_train_shards//args.n_machine
    if train_shards_per_machine == 0:
        raise Exception('num_train_shards is smaller than n_machine')
    if machine_id == args.n_machine-1:
        train_shards_id_range = range(machine_id*train_shards_per_machine,args.num_train_shards)
    else:
        train_shards_id_range = range(machine_id*train_shards_per_machine,(machine_id+1)*train_shards_per_machine)

    test_shards_per_machine = args.num_test_shards//args.n_machine
    if test_shards_per_machine == 0:
        raise Exception('num_test_shards is smaller than n_machine')
    if machine_id == args.n_machine-1:
        test_shards_id_range = range(machine_id*test_shards_per_machine,args.num_test_shards)
    else:
        test_shards_id_range = range(machine_id*test_shards_per_machine,(machine_id+1)*test_shards_per_machine)

    # main sharding process
    segmenter = TextSharding.NLTKSegmenter()
    sharding = TextSharding.Sharding(
        input_files,
        str(shards_dir.absolute()) + os.sep,
        args.num_train_shards,
        args.num_test_shards,
        args.frac_test,
        args.max_memory,
        total_tqdm,
        args.verbose,
        train_shards_id_range,
        test_shards_id_range,
        machine_id
    ) 

    sharding.distribute_articles_over_shards(segmenter)
    sharding.write_shards_to_disk()
    total_tqdm.close()
