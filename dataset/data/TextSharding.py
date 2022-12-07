# coding=utf-8
# Copyright 2022 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
# code taken from commit: ea000838156e3be251699ad6a3c8b1339c76e987
# https://github.com/IntelLabs/academic-budget-bert
# Copyright 2021 Intel Corporation. All rights reserved.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from collections import defaultdict

import statistics
import pickle
import os
import gc
import psutil

from tqdm import tqdm

TOTAL_UNIT = 100
STAGE_NUM = 4
STAGE_UNIT = TOTAL_UNIT/STAGE_NUM

class Sharding:
    '''Main Sharding Class'''
    def __init__(
        self,
        input_files,
        output_name_prefix,
        n_training_shards,
        n_test_shards,
        fraction_test_set,
        max_memory,
        total_tqdm,
        verbose,
        train_shards_id_range,
        test_shards_id_range,
        machine_id
    ):
        assert len(input_files) > 0, "The input file list must contain at least one file."
        assert n_training_shards > 0, "There must be at least one output shard."
        assert n_test_shards > 0, "There must be at least one output shard."

        self.n_training_shards = n_training_shards
        self.n_test_shards = n_test_shards
        self.fraction_test_set = fraction_test_set
        self.max_memory = max_memory

        self.input_files = input_files

        self.output_name_prefix = output_name_prefix
        self.output_training_identifier = "training"
        self.output_test_identifier = "test"
        self.output_file_extension = ".txt"

        self.articles = {}  # key: integer identifier, value: list of articles
        self.sentences = {}  # key: integer identifier, value: list of sentences
        self.output_training_files = {}  # key: filename, value: list of articles to go into file
        self.output_test_files = {}  # key: filename, value: list of articles to go into file
        
        self.sentence_amount = {} # key: integer identifier, value: amount of sentences in an article
        self.log_point = [] # save the article index of different tmp file
        self.file_amount = 0
        self.total_tqdm =  total_tqdm
        self.verbose =  verbose
        self.train_shards_id_range = train_shards_id_range
        self.test_shards_id_range = test_shards_id_range
        self.machine_id = machine_id

        self.init_output_files()

    def get_current_memory_gb(self):
        '''get memory usage of process in gb unit'''
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        return info.uss / 1024. / 1024. / 1024.

    def init_output_files(self):
        assert (
            len(self.output_training_files)== 0
        ), "Internal storage self.output_files already contains data. This function is intended to be used by the constructor only."
        assert (
            len(self.output_test_files) == 0
        ), "Internal storage self.output_files already contains data. This function is intended to be used by the constructor only."

        for i in self.train_shards_id_range:
            name = (
                self.output_name_prefix
                + self.output_training_identifier
                + str(i)
                + self.output_file_extension
            )
            self.output_training_files[name] = []

        for i in self.test_shards_id_range:
            name = (
                self.output_name_prefix
                + self.output_test_identifier
                + str(i)
                + self.output_file_extension
            )
            self.output_test_files[name] = []

    def get_sentences_per_shard(self, shard):
        '''get sentence amount of a shard'''
        result = 0
        for article_id in shard:
            result += self.sentence_amount[article_id]

        return result

    def distribute_articles_over_shards(self, segmenter):
        '''article distribution'''
        count = 0
        for input_file in self.input_files:
            with open(input_file, mode = "r", newline = "\n", encoding = 'utf-8') as fin:
                for i,line in enumerate(fin):
                    count += 1
        
        # Create dictionary with - key: sentence count per article, value: article id number
        sentence_counts = defaultdict(lambda: [])
        
        global_article_count = 0
        max_sentences = 0
        total_sentences = 0
        self.log_point.append(0) # for convenience in function 'write_single_shard'
        
        articles = {}

        interval = STAGE_UNIT/count
        if self.verbose:
            print("Start: Sentence Segmentation")
            pbar = tqdm(total=count)
            

        for input_file in self.input_files:
            with open(input_file, mode="r", newline="\n", encoding = 'utf-8') as fin:
                for _, line in enumerate(fin):
                    if self.verbose:
                        pbar.update(1)
                        pbar.set_description("Distributing")
                    else:
                        self.total_tqdm.update(interval)
                        self.total_tqdm.set_description("Distributing")
                    count -= 1
                    if count%100000 == 0:
                        if (self.get_current_memory_gb() >= self.max_memory-2) or (count == 0): # use self.max_memory-2 because of the fluctuation of memory usage
                            self.log_point.append(global_article_count)
                            with open(f'{self.output_name_prefix}seg_article_{self.file_amount}_{self.machine_id}.pickle','wb') as ostream:
                                pickle.dump(articles, ostream)
                            
                            self.file_amount += 1
                            articles.clear()
                            gc.collect()
                            self.max_memory += 1   # make sure the initail memory in the next loop would not exceed the max_memory

                    if line.strip():
                        sent = segmenter.segment_string(line.rstrip())
                        current_length = len(sent)
                        self.sentence_amount[global_article_count] = current_length
                        articles[global_article_count] = sent
                        sentence_counts[current_length].append(global_article_count)
                        max_sentences = max(max_sentences, current_length)
                        total_sentences += current_length
                        global_article_count += 1
        if self.verbose:
            pbar.close()
            print("End: Sentence Segmentation")
            print("Start: Distribute Articles Over Shards")
        else:
            if self.total_tqdm.n < STAGE_UNIT*2:
                self.total_tqdm.update(STAGE_UNIT*2 - self.total_tqdm.n)
                self.total_tqdm.set_description("Distributing")

        articles_amount = global_article_count
            
        assert (
            articles_amount >= self.n_training_shards + self.n_test_shards
        ), "There are fewer articles than shards. Please add more data or reduce the number of shards requested."


        n_sentences_assigned_to_training = int((1 - self.fraction_test_set) * total_sentences)
        nominal_sentences_per_training_shard = (
            n_sentences_assigned_to_training // self.n_training_shards
        )
        nominal_sentences_per_test_shard = (
            total_sentences - n_sentences_assigned_to_training
        ) // self.n_test_shards

        consumed_article_set = set({})
        unused_article_set = set(range(global_article_count))

        # Make first pass and add one article worth of lines per file
        for file, shard_id_list in self.output_training_files.items():
            current_article_id = sentence_counts[max_sentences][-1]
            sentence_counts[max_sentences].pop(-1)
            shard_id_list.append(current_article_id)
            consumed_article_set.add(current_article_id)
            unused_article_set.remove(current_article_id)

            # Maintain the max sentence count
            while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                max_sentences -= 1

            if self.sentence_amount[current_article_id] > nominal_sentences_per_training_shard:
                nominal_sentences_per_training_shard = self.sentence_amount[current_article_id]
                print(
                    "Warning: A single article contains more than the nominal number of sentences per training shard."
                )

        for file, shard_id_list in self.output_test_files.items():
            current_article_id = sentence_counts[max_sentences][-1]
            sentence_counts[max_sentences].pop(-1)
            shard_id_list.append(current_article_id)
            consumed_article_set.add(current_article_id)
            unused_article_set.remove(current_article_id)

            # Maintain the max sentence count
            while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                max_sentences -= 1

            if self.sentence_amount[current_article_id] > nominal_sentences_per_test_shard:
                nominal_sentences_per_test_shard = self.sentence_amount[current_article_id]
                print(
                    "Warning: A single article contains more than the nominal number of sentences per test shard."
                )

        training_counts = []
        test_counts = []

        for _, shard_id_list in self.output_training_files.items():
            training_counts.append(self.get_sentences_per_shard(shard_id_list))

        for _, shard_id_list in self.output_test_files.items():
            test_counts.append(self.get_sentences_per_shard(shard_id_list))

        training_median = statistics.median(training_counts)
        test_median = statistics.median(test_counts)

        # Make subsequent passes over files to find articles to add without going over limit
        history_remaining = []
        n_history_remaining = 4

        interval = STAGE_UNIT/len(unused_article_set)
        if self.verbose:
            pbar = tqdm(total=len(unused_article_set))
        count = 0
        while len(consumed_article_set) < articles_amount:
            for fidx, file in enumerate(self.output_training_files):
                nominal_next_article_size = min(
                    nominal_sentences_per_training_shard - training_counts[fidx], max_sentences
                )

                # Maintain the max sentence count
                while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                    max_sentences -= 1

                while (
                    len(sentence_counts[nominal_next_article_size]) == 0
                    and nominal_next_article_size > 0
                ):
                    nominal_next_article_size -= 1

                if (
                    nominal_next_article_size not in sentence_counts
                    or nominal_next_article_size == 0
                    or training_counts[fidx] > training_median
                ):
                    continue  # skip adding to this file, will come back later if no file can accept unused articles

                count += 1
                if self.verbose:
                    pbar.update(1)
                    pbar.set_description('Sharding Data')
                else:
                    self.total_tqdm.update(interval)
                    self.total_tqdm.set_description("Sharding Data")

                current_article_id = sentence_counts[nominal_next_article_size][-1]
                sentence_counts[nominal_next_article_size].pop(-1)

                self.output_training_files[file].append(current_article_id)
                consumed_article_set.add(current_article_id)
                unused_article_set.remove(current_article_id)
                

            for fidx, file in enumerate(self.output_test_files):
                nominal_next_article_size = min(
                    nominal_sentences_per_test_shard - test_counts[fidx], max_sentences
                )

                # Maintain the max sentence count
                while len(sentence_counts[max_sentences]) == 0 and max_sentences > 0:
                    max_sentences -= 1

                while (
                    len(sentence_counts[nominal_next_article_size]) == 0
                    and nominal_next_article_size > 0
                ):
                    nominal_next_article_size -= 1

                if (
                    nominal_next_article_size not in sentence_counts
                    or nominal_next_article_size == 0
                    or test_counts[fidx] > test_median
                ):
                    continue  # skip adding to this file, will come back later if no file can accept unused articles

                count += 1
                if self.verbose:
                    pbar.update(1)
                    pbar.set_description('Sharding Data')
                else:
                    self.total_tqdm.update(interval)
                    self.total_tqdm.set_description("Sharding Data")
                current_article_id = sentence_counts[nominal_next_article_size][-1]
                sentence_counts[nominal_next_article_size].pop(-1)

                self.output_test_files[file].append(current_article_id)
                consumed_article_set.add(current_article_id)
                unused_article_set.remove(current_article_id)
                

            # If unable to place articles a few times, bump up nominal sizes by fraction until articles get placed
            if len(history_remaining) == n_history_remaining:
                history_remaining.pop(0)
            history_remaining.append(len(unused_article_set))

            history_same = True
            for i in range(1, len(history_remaining)):
                history_same = history_same and (history_remaining[i - 1] == history_remaining[i])

            if history_same:
                nominal_sentences_per_training_shard += 1
                # nominal_sentences_per_test_shard += 1

            training_counts = []
            test_counts = []
            for _, shard_id_list in self.output_training_files.items():
                training_counts.append(
                    self.get_sentences_per_shard(shard_id_list)
                )

            for _, shard_id_list in self.output_test_files.items():
                test_counts.append(self.get_sentences_per_shard(shard_id_list))

            training_median = statistics.median(training_counts)
            test_median = statistics.median(test_counts)
            # if self.verbose:
            #     print("Distributing data over shards:", len(unused_article_set), "articles remaining.")
        
        if len(unused_article_set) != 0:
            print("Warning: Some articles did not make it into output files.")

        if self.verbose:
            for _, shard_id_list in self.output_training_files.items():
                print(
                    "Training shard:", self.get_sentences_per_shard(shard_id_list)
                )

            for _, shard_id_list in self.output_test_files.items():
                print("Test shard:", self.get_sentences_per_shard(shard_id_list))

            print("End: Distribute Articles Over Shards")
            pbar.close()
        else:
            if self.total_tqdm.n < STAGE_UNIT*3:
                self.total_tqdm.update(STAGE_UNIT*3-self.total_tqdm.n)

    def write_shards_to_disk(self):
        '''write shards to disk'''
        file_num = len(self.output_training_files)+len(self.output_test_files)
        if self.verbose:
            print("Start: Write Shards to Disk")
            pbar = tqdm(total=file_num)

        interval = STAGE_UNIT/file_num
        
        count = 0
        for shard, shard_id_list in self.output_training_files.items():
            count+=1
            if self.verbose:
                pbar.update(1)
                pbar.set_description("Writing Train Data")
            else:
                self.total_tqdm.update(interval)
                self.total_tqdm.set_description("Writing Train Data")
            self.write_single_shard(shard, shard_id_list)

        for shard, shard_id_list in self.output_test_files.items():
            count+=1
            if self.verbose:
                pbar.update(1)
                pbar.set_description("Writing Test Data")
            else:
                self.total_tqdm.update(interval)
                self.total_tqdm.set_description("Writing Test Data")
            self.write_single_shard(shard, shard_id_list)

        if self.verbose:
            pbar.close()
            print("End: Write Shards to Disk")
        else:
            if self.total_tqdm.n < STAGE_UNIT*4:
                self.total_tqdm.update(STAGE_UNIT*4-self.total_tqdm.n)            

    # get data from different pickles and save in several shard files
    def write_single_shard(self, shard_name, shard):
        '''sub function to write single shard file'''
        missing_article_id = []
        with open(shard_name, mode="w", newline="\n", encoding='utf-8') as f:
            article_dict = {id:[] for id in shard}
            for i in range(self.file_amount):
                with open(f'{self.output_name_prefix}seg_article_{i}_{self.machine_id}.pickle','rb') as istream:
                    sentences = pickle.load(istream)
                article_list = [id for id in shard if (id>=self.log_point[i]) and (id<self.log_point[i+1])]   
                
                for article_id in article_list:
                    try:
                        article_dict[article_id] = sentences[article_id]
                    except KeyError:
                        missing_article_id.append(article_id)
                        
            for article_id in article_dict:
                for line in article_dict[article_id]:
                    f.write(line + "\n")
                f.write("\n")  # Line break between article
        if len(missing_article_id) != 0 :
            print(f'\n{shard_name} have missing articles:\n')
            for id in missing_article_id:
                print('missing key: ',id,'\n')

try:
    import nltk
    nltk.download("punkt")
except ModuleNotFoundError or ImportError as e:
    print("nltk is required for sharding. please install before running.")

class NLTKSegmenter:
    def segment_string(self, article):
        return nltk.tokenize.sent_tokenize(article)
