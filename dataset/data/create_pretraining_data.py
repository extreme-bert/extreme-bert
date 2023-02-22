# coding=utf-8
# Copyright 2022 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
# code taken from commit: ea000838156e3be251699ad6a3c8b1339c76e987
# https://github.com/IntelLabs/academic-budget-bert
# Copyright 2021 Intel Corporation. All rights reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

"""Create masked LM/next sentence masked_lm TF examples for BERT."""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import collections
import os
import random
from io import open

import h5py
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.data.utils import convert_to_unicode


class TDNANgramDict(object):
    """
    Dict class to store the ngram
    """

    def __init__(self, ngram_freq_path, max_ngram_in_seq=20):
        """Constructs TDNANgramDict
        :param ngram_freq_path: ngrams with frequency
        """
        self.ngram_freq_path = ngram_freq_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.id_to_ngram_list = []
        self.ngram_to_id_dict = {}

        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                ngram = line.strip()
                self.id_to_ngram_list.append(ngram)
                self.ngram_to_id_dict[ngram] = i

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram, freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next, input_Ngram_ids=None,
                 Ngram_attention_mask=None, Ngram_token_type_ids=None, Ngram_position_matrix=None):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.input_Ngram_ids = input_Ngram_ids
        self.Ngram_attention_mask = Ngram_attention_mask
        self.Ngram_token_type_ids = Ngram_token_type_ids
        self.Ngram_position_matrix = Ngram_position_matrix
    # def __str__(self):
    #   s = ""
    #   s += "tokens: %s\n" % (" ".join(
    #       [tokenization.printable_text(x) for x in self.tokens]))
    #   s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    #   s += "is_random_next: %s\n" % self.is_random_next
    #   s += "masked_lm_positions: %s\n" % (" ".join(
    #       [str(x) for x in self.masked_lm_positions]))
    #   s += "masked_lm_labels: %s\n" % (" ".join(
    #       [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    #   s += "\n"
    #   return s

    # def __repr__(self):
    #   return self.__str__()


def write_instance_to_example_file(
        instances, Ngram_dict, Ngram_flag, tokenizer, max_seq_length, max_predictions_per_seq, output_file, no_nsp
):
    """Create TF example files from `TrainingInstance`s."""

    total_written = 0
    features = collections.OrderedDict()

    num_instances = len(instances)
    features["input_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["input_mask"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["segment_ids"] = np.zeros([num_instances, max_seq_length], dtype="int32")
    features["masked_lm_positions"] = np.zeros(
        [num_instances, max_predictions_per_seq], dtype="int32"
    )
    features["masked_lm_ids"] = np.zeros([num_instances, max_predictions_per_seq], dtype="int32")

    if not no_nsp:
        features["next_sentence_labels"] = np.zeros(num_instances, dtype="int32")

    if Ngram_flag:
        max_ngram_per_seq = Ngram_dict.max_ngram_in_seq
        features["input_Ngram_ids"] = np.zeros([num_instances, max_ngram_per_seq], dtype="int32")
        features['Ngram_attention_mask'] = np.zeros([num_instances, max_ngram_per_seq], dtype="int32")
        features['Ngram_token_type_ids'] = np.zeros([num_instances, max_ngram_per_seq], dtype="int32")
        features['Ngram_position_matrix'] = np.zeros([num_instances, max_seq_length, max_ngram_per_seq], dtype="int32")

    for inst_index, instance in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        # next_sentence_label = 1 if instance.is_random_next else 0

        features["input_ids"][inst_index] = input_ids
        features["input_mask"][inst_index] = input_mask
        features["segment_ids"][inst_index] = segment_ids
        features["masked_lm_positions"][inst_index] = masked_lm_positions
        features["masked_lm_ids"][inst_index] = masked_lm_ids

        if Ngram_flag:
            features['input_Ngram_ids'][inst_index] = instance.input_Ngram_ids
            features['Ngram_attention_mask'][inst_index] = instance.Ngram_attention_mask
            features['Ngram_token_type_ids'][inst_index] = instance.Ngram_token_type_ids
            features['Ngram_position_matrix'][inst_index] = instance.Ngram_position_matrix

        if not no_nsp:
            features["next_sentence_labels"][inst_index] = 1 if instance.is_random_next else 0

        total_written += 1

        # if inst_index < 20:
        #   tf.logging.info("*** Example ***")
        #   tf.logging.info("tokens: %s" % " ".join(
        #       [tokenization.printable_text(x) for x in instance.tokens]))

        #   for feature_name in features.keys():
        #     feature = features[feature_name]
        #     values = []
        #     if feature.int64_list.value:
        #       values = feature.int64_list.value
        #     elif feature.float_list.value:
        #       values = feature.float_list.value
        #     tf.logging.info(
        #         "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    # print("saving data")
    f = h5py.File(output_file, "w")
    f.create_dataset("input_ids", data=features["input_ids"], dtype="i4", compression="gzip")
    f.create_dataset("input_mask", data=features["input_mask"], dtype="i1", compression="gzip")
    f.create_dataset("segment_ids", data=features["segment_ids"], dtype="i1", compression="gzip")
    f.create_dataset(
        "masked_lm_positions", data=features["masked_lm_positions"], dtype="i4", compression="gzip"
    )
    f.create_dataset(
        "masked_lm_ids", data=features["masked_lm_ids"], dtype="i4", compression="gzip"
    )

    if Ngram_flag:
        f.create_dataset("input_Ngram_ids", data=features["input_Ngram_ids"], dtype="i4", compression="gzip")
        f.create_dataset("Ngram_attention_mask", data=features["Ngram_attention_mask"], dtype="i4", compression="gzip")
        f.create_dataset("Ngram_token_type_ids", data=features["Ngram_token_type_ids"], dtype="i4", compression="gzip")
        f.create_dataset("Ngram_position_matrix", data=features["Ngram_position_matrix"], dtype="i4",
                         compression="gzip")

    if not no_nsp:
        f.create_dataset(
            "next_sentence_labels",
            data=features["next_sentence_labels"],
            dtype="i1",
            compression="gzip",
        )
    f.flush()
    f.close()


def create_training_instances(
        input_files,  # 当input_file是一个文件夹路径时，将其文件夹下面的所有以txt结尾的文件的路径加入到input_files下
        tokenizer,
        Ngram_dict,  # guhao
        Ngram_flag,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        rng,
        no_nsp,
):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    # count = 0
    # print(count)
    # for input_file in input_files:
    #     with open(input_file, "r") as reader:
    #         while True:
    #             count += 1

    # for _ in range(dupe_factor):
    #     for document_index in range(len(all_documents)):
    #         count += 1
    # print(count)
    # pbar = tqdm(total=count,position=1)

    for input_file in input_files:
        # print("creating instance from {}".format(input_file))
        with open(input_file, "r", encoding='utf-8') as reader:
            while True:
                # pbar.update(1)
                # pbar.set_description("creating instance from {}".format(input_file))
                line = convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            # pbar.update(1)
            # pbar.set_description("creating instance")
            if no_nsp:
                instances.extend(
                    create_instances_from_document_no_nsp(
                        all_documents,
                        Ngram_dict,  # guhao
                        Ngram_flag,
                        document_index,
                        max_seq_length,
                        short_seq_prob,
                        masked_lm_prob,
                        max_predictions_per_seq,
                        vocab_words,
                        rng,
                        tokenizer,
                    )
                )
            else:
                instances.extend(
                    create_instances_from_document(
                        all_documents,
                        document_index,
                        max_seq_length,
                        short_seq_prob,
                        masked_lm_prob,
                        max_predictions_per_seq,
                        vocab_words,
                        rng,
                        tokenizer
                    )
                )

    rng.shuffle(instances)
    return instances


def create_instances_from_document_no_nsp(
        all_documents,
        Ngram_dict,  # guhao
        Ngram_flag,
        document_index,
        max_seq_length,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_words,
        rng,
        tokenizer,
):
    """Creates `TrainingInstance`s for a single document."""
    """Generate single sentences (NO 2nd segment for NSP task)"""
    document = all_documents[document_index]

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(20, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        # current_chunk是以每一行为单位的
        current_length += len(segment)
        if current_length >= target_seq_length:
            if current_chunk:
                tokens_a = []
                for j in range(len(current_chunk)):
                    tokens_a.extend(current_chunk[j])

                truncate_single_seq(tokens_a, max_num_tokens, rng)
                # tokens_a 为截断后的一个单词列表,设长度为len_a
                assert len(tokens_a) >= 1

                tokens = []
                segment_ids = []
                # tokens.append("[CLS]")
                tokens.append(tokenizer.cls_token)
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                # tokens.append("[SEP]")
                tokens.append(tokenizer.sep_token)
                segment_ids.append(0)
                # tokens 为sentence_a加入[cls]和[sep]的句子
                # segment_ids为句子的id,0对应sentence_a

                assert len(tokens) <= max_seq_length
                '''_______guhao______'''
                ngram_ids = None
                ngram_mask_array = None
                ngram_seg_ids = None
                ngram_positions_matrix = None
                if Ngram_flag:
                    ngram_matches = []
                    #  Filter the word segment from 2 to 7 to check whether there is a word
                    for p in range(2, 8):
                        for q in range(0, len(tokens) - p + 1):
                            character_segment = tokens[q:q + p]
                            tmp_text = ''.join([tmp_x for tmp_x in character_segment])
                            character_segment = tmp_text.replace('Ġ', ' ').strip()
                            if character_segment in Ngram_dict.ngram_to_id_dict:
                                ngram_index = Ngram_dict.ngram_to_id_dict[character_segment]
                                ngram_matches.append([ngram_index, q, p, character_segment])
                    #  这里是对长度从2到8进行遍历,开始的距离从0开始，然后取得tokens的连续ngram,判断是否在N_gram字典里面
                    #  记录N_gram的id,N_gram的长度,开始的位置,以及N_gram自己

                    max_word_in_seq_proportion = Ngram_dict.max_ngram_in_seq
                    if len(ngram_matches) > max_word_in_seq_proportion:
                        ngram_matches = ngram_matches[:max_word_in_seq_proportion]

                    ngram_ids = [ngram[0] for ngram in ngram_matches]
                    ngram_positions = [ngram[1] for ngram in ngram_matches]
                    ngram_lengths = [ngram[2] for ngram in ngram_matches]
                    ngram_tuples = [ngram[3] for ngram in ngram_matches]
                    ngram_seg_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]
                    # ngram_ids为N_gram的id,ngram_positions为N_gram开始的位置,ngram_lengths为N_gram开始的位置
                    # ngram_tuples为哪一个N_gram,ngram_seg_ids为N_gram对应的句子是sentence_a

                    ngram_mask_array = np.zeros(Ngram_dict.max_ngram_in_seq)
                    ngram_mask_array[:len(ngram_ids)] = 1  # 有N_gram的为1,0对应于pad部分

                    # record the masked positions
                    ngram_positions_matrix = np.zeros(shape=(max_seq_length, Ngram_dict.max_ngram_in_seq),
                                                      dtype=np.int32)
                    for j in range(len(ngram_ids)):
                        ngram_positions_matrix[ngram_positions[j]:ngram_positions[j] + ngram_lengths[j], j] = 1.0

                    # Zero-pad up to the max word in seq length.
                    padding = [0] * (Ngram_dict.max_ngram_in_seq - len(ngram_ids))
                    ngram_ids += padding  # ngram_ids 进行pad,0对应于没有ngram的pad部分
                    ngram_lengths += padding
                    ngram_seg_ids += padding
                    # ngram_ids,ngram_lengths,ngram_seg_ids,ngram_mask_array均为[Ngram_dict.max_ngram_in_seq,]的列表
                    # Ngram_position_matrix表示token与N_gram的位置关系,为[max_seq_length,Ngram_dict.max_ngram_in_seq]

                '''_______guhao______'''

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer,
                )
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=False,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    input_Ngram_ids=ngram_ids,
                    Ngram_attention_mask=ngram_mask_array,
                    Ngram_token_type_ids=ngram_seg_ids,
                    Ngram_position_matrix=ngram_positions_matrix
                )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def create_instances_from_document(
        all_documents,
        document_index,
        max_seq_length,
        short_seq_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_words,
        rng,
        tokenizer,
):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    # If picked random document is the same as the current document
                    if random_document_index == document_index:
                        is_random_next = False

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                # tokens.append("[CLS]")
                tokens.append(tokenizer.cls_token)
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                # tokens.append("[SEP]")
                tokens.append(tokenizer.sep_token)
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append(tokenizer.sep_token)
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer
                )
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, tokenizer):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == tokenizer.cls_token or token == tokenizer.sep_token:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            # masked_token = "[MASK]"
            masked_token = tokenizer.mask_token
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                if "bert" in tokenizer.name_or_path:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                elif "roberta" in tokenizer.name_or_path:
                    masked_token = vocab_words[rng.randint(3, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def truncate_single_seq(tokens, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens)
        if total_length <= max_num_tokens:
            break
        assert len(tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "--tokenizer_name",
        default='bert-large-uncased',
        type=str,
        # required=True,
        help="The name of tokenizer",
    )
    parser.add_argument(
        "--input_file",
        default='../data_sharded/training0.txt',
        type=str,
        # required=True,
        help="The input train corpus. can be directory with .txt files or a path to a single file",
    )
    parser.add_argument(
        "--output_file",
        default='./output',
        type=str,
        # required=True,
        help="The output file where the model checkpoints will be written.",
    )

    ## Other parameters

    # str
    parser.add_argument(
        "--bert_model",
        default="bert-large-uncased",
        type=str,
        required=False,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    # int
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, and sequences shorter \n"
             "than this will be padded.",
    )
    parser.add_argument(
        "--dupe_factor",
        default=1,
        type=int,
        help="Number of times to duplicate the input data (with different masks).",
    )
    parser.add_argument(
        "--max_predictions_per_seq", default=20, type=int, help="Maximum sequence length."
    )

    # floats

    parser.add_argument("--masked_lm_prob", default=0.15, type=float, help="Masked LM probability.")

    parser.add_argument(
        "--short_seq_prob",
        default=0.1,
        type=float,
        help="Probability to create a sequence shorter than maximum sequence length",
    )

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=12345, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_nsp",
        default=True,
        action="store_true",
        help="Generate samples without 2nd sentence segments (no NSP task)",
    )

    parser.add_argument("--Ngram_path",
                        default='./ngram.txt',
                        type=str,
                        # required=True,
                        help="Path to Ngram path")
    

    args = parser.parse_args()
    
    args.Ngram_flag = False
    if args.Ngram_path is not None:
        args.Ngram_flag = True
    Ngram_dict = TDNANgramDict(args.Ngram_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, do_lower_case=args.do_lower_case,
                                              max_len=1024)

    input_files = []
    if os.path.isfile(args.input_file):
        input_files.append(args.input_file)
    elif os.path.isdir(args.input_file):
        input_files = [
            os.path.join(args.input_file, f)
            for f in os.listdir(args.input_file)
            if (os.path.isfile(os.path.join(args.input_file, f)) and f.endswith(".txt"))
        ]
    else:
        raise ValueError("{} is not a valid path".format(args.input_file))

    rng = random.Random(args.random_seed)
    instances = create_training_instances(
        input_files,
        tokenizer,
        Ngram_dict,
        args.Ngram_flag,
        args.max_seq_length,
        args.dupe_factor,
        args.short_seq_prob,
        args.masked_lm_prob,
        args.max_predictions_per_seq,
        rng,
        args.no_nsp,
    )

    output_file = args.output_file

    write_instance_to_example_file(
        instances,
        Ngram_dict,
        args.Ngram_flag,
        tokenizer,
        args.max_seq_length,
        args.max_predictions_per_seq,
        output_file,
        args.no_nsp,
    )


if __name__ == "__main__":
    main()
