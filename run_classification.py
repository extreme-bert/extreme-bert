# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
from tqdm.auto import tqdm, trange
from typing import Dict, NamedTuple, Any, NewType
from sklearn.metrics import f1_score
import numpy as np
import shutil
import json
from enum import Enum
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union
from os.path import dirname, abspath
import random
import math
import re
from typing import Callable, Iterable, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers import RobertaTokenizer
from pretraining.modeling_roberta import RobertaForSequenceClassification
from transformers import PreTrainedModel
from pretraining.modeling import BertForSequenceClassification
from transformers import BertConfig
from transformers import PretrainedConfig
from transformers import PreTrainedTokenizer
from filelock import FileLock
from packaging import version
import time
import torch
import argparse
from enum import Enum
import socket
from datetime import datetime
import csv
from torch.utils.data.dataset import Dataset
from glue import glue_processors
from torch import nn
from pathlib import Path


class EvaluationStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class EvalPrediction(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


logger = logging.getLogger(__name__)


############################ for config ###########################
class RobertaConfig(BertConfig):
    model_type = "roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, Ngram_vocab_size=6119, num_hidden_Ngram_layers=1,
                 layer_norm_type='pytorch', encoder_ln_mode="post-ln", fused_linear_layer=True,
                 layernorm_embedding=False, sparse_mask_prediction=True, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.Ngram_size = Ngram_vocab_size
        self.num_hidden_Ngram_layers = num_hidden_Ngram_layers
        self.layer_norm_type = layer_norm_type
        self.encoder_ln_mode = encoder_ln_mode
        self.fused_linear_layer = fused_linear_layer
        self.layernorm_embedding = layernorm_embedding
        self.sparse_mask_prediction = sparse_mask_prediction
        self.useLN = True


class AutoConfig:
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return RobertaConfig.from_dict(config_dict, **kwargs)


############################ for config ###########################

############################ for dataset and accuracy ###########################
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

        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                ngram = line.strip()
                self.id_to_ngram_list.append(ngram)
                self.ngram_to_id_dict[ngram] = i

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram, freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


glue_tasks_num_labels = {
    "citation_intent": 6,
    "ag": 4,
    "amazon": 2,
    "chemprot": 13,
    "hyperpartisan_news": 2,
    "imdb": 2,
    "rct-20k": 5,
    "sciie": 7,
    "SST2": 2
}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "citation_intent":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "ag":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "amazon":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "chemprot":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "hyperpartisan_news":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "imdb":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "rct-20k":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "sciie":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    elif task_name == "SST2":
        return {"acc_and_f1": acc_and_f1(preds, labels)}
    else:
        raise KeyError(task_name)


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


glue_output_modes = {
    "citation_intent": "classification",
    "ag": "classification",
    "amazon": "classification",
    "chemprot": "classification",
    "hyperpartisan_news": "classification",
    "imdb": "classification",
    "rct-20k": "classification",
    "sciie": "classification",
    "SST2": "classification"
}


############################ for dataset and accuracy ###########################

############################ for process of dataset #############################

@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    input_Ngram_ids: List[int]
    Ngram_attention_mask: List[int]
    Ngram_token_type_ids: List[int]
    Ngram_position_matrix: List[int]
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def glue_convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        Ngram_dict,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
):
    return _glue_convert_examples_to_features(
        examples, tokenizer, Ngram_dict, max_length=max_length, task=task, label_list=label_list,
        output_mode=output_mode
    )


def _glue_convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        Ngram_dict,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
):
    label_map = {label: i for i, label in enumerate(label_list)}

    # amazon ['helpful', 'unhelpful'] for example
    # label_map得到的是label到索引的字典
    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":  # 这里是根据example的label得到对应的索引
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    # 得到的labels为所有examples的label
    labels = [label_from_example(example) for example in examples]

    num_tokens = 0
    num_unk = 0

    features = []
    for i in range(len(examples)):
        if i % 10000 == 0:
            logger.info("Writing example %d of %d" % (i, len(examples)))

        # 对examples[i]的text_a进行tokenizer分解，如果大于max_length-2,则取前max_length-2个token
        tokens_a = tokenizer.tokenize(examples[i].text_a)  # tokens_a is a python list
        tokens_b = None
        if len(tokens_a) > max_length - 2:
            tokens_a = tokens_a[:(max_length - 2)]

        # 为tokens加上['<s>']和['</s>']
        tokens = ["<s>"] + tokens_a + ["</s>"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # list of token id (int)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        # input_padding为tokenizer里面的pad_token_id的id乘以最长长度减去输入的长度,zero_padding为mask，0表示mask
        input_padding = [tokenizer.pad_token_id] * (max_length - len(input_ids))
        zero_padding = [0] * (max_length - len(input_ids))
        input_ids += input_padding
        input_mask += zero_padding
        segment_ids += zero_padding

        # 最后input_ids 的前面为句子的id，后面是pad的id
        # input_mask前面句子的id部分为1，后面的pad部分为0
        # segment_ids为全为0
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length

        num_tokens += len(input_ids)

        num_unk += input_ids.count(3)

        ngram_matches = []
        #  Filter the word segment from 2 to 7 to check whether there is a word
        for p in range(2, 8):
            for q in range(0, len(tokens) - p + 1):  # tokens = ['<s>',tokens_a,'</s>']
                # 对长度为2到8进行遍历，对q从长度进行遍历，得到character_segment,然后如果它在Ngram_dict.ngram_to_id_dict里面
                # 则在ngram_matches列表中添加n_gram的索引，开始的位置，长度，以及对应的character_segment
                character_segment = tokens[q:q + p]
                tmp_text = ''.join([tmp_x for tmp_x in character_segment])
                character_segment = tmp_text.replace('Ġ', ' ').strip()
                if character_segment in Ngram_dict.ngram_to_id_dict:
                    ngram_index = Ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        # 得到每一个seq的最多的n_gram的个数，如果大于的话，就取前最多的n_gram的个数
        max_word_in_seq_proportion = Ngram_dict.max_ngram_in_seq
        if len(ngram_matches) > max_word_in_seq_proportion:
            ngram_matches = ngram_matches[:max_word_in_seq_proportion]

        ngram_ids = [ngram[0] for ngram in ngram_matches]  # ngram_ids保存了第几个n_garm
        ngram_positions = [ngram[1] for ngram in ngram_matches]  # ngram_positions保存了开始的位置
        ngram_lengths = [ngram[2] for ngram in ngram_matches]  # ngram_lengths保存了长度
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_seg_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]

        import numpy as np
        # ngram_mask_array 为长度为Ngram_dict长度的array，前ngram_ids为1，其他为0
        ngram_mask_array = np.zeros(Ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_mask_array[:len(ngram_ids)] = 1

        # record the masked positions
        # ngram_positions_matrix 为max_length和max_ngram_in_seq的矩阵，每一列对应一个一个N_gram
        # 每一列，从ngram开始的位置到结束的位置为1
        ngram_positions_matrix = np.zeros(shape=(max_length, Ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for j in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[j]:ngram_positions[j] + ngram_lengths[j], j] = 1.0

        # Zero-pad up to the max word in seq length.
        padding = [0] * (Ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_seg_ids += padding

        # 'Ngram_tuples': ngram_tuples,
        # 'Ngram_lengths': ngram_lengths,
        inputs = {'input_ids': input_ids,  # input_ids 的前面为句子的id，后面是pad的id
                  'attention_mask': input_mask,  # input_mask的前面是1，后面pad的是0
                  'token_type_ids': segment_ids,
                  'input_Ngram_ids': ngram_ids,  # ngram_ids的前面是对应于哪一个ngram，后面是0
                  'Ngram_attention_mask': ngram_mask_array,  # ngram_mask_array的对应有ngram_ids的位置为1，ngram_ids进行pad的位置为0
                  'Ngram_token_type_ids': ngram_seg_ids,
                  'Ngram_position_matrix': ngram_positions_matrix,
                  }  # ngram_positions_matrix对应于[max_seq,max_ngram_in_seq]
        #  每一列对应一个n_gram,且开始的位置到结束的位置为1
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)  # 输出为features的列表，其中每一个元素为InputFeatures数据类型的数据

    if num_unk > 0:
        print("there exists {num} [UNK] in input sentence".format(num=num_unk))

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class GlueDataset(Dataset):

    def __init__(
            self,
            args,
            Ngram_dict,
            tokenizer,
            limit_length=None,
            mode=Split.train,
            cache_dir=None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        # 是classification 还是 regression
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )
        label_list = self.processor.get_labels()  # ['0', '1'] for imdb
        self.label_list = label_list

        if mode == Split.dev:
            examples = self.processor.get_dev_examples(args.data_dir)
        elif mode == Split.test:
            examples = self.processor.get_test_examples(args.data_dir)
        else:
            examples = self.processor.get_train_examples(args.data_dir)
        if limit_length is not None:
            examples = examples[:limit_length]
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            Ngram_dict,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=self.output_mode,
        )
        start = time.time()
        torch.save(self.features, cached_features_file)
        # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
        logger.info(
            "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


############################ for process of dataset #############################

############################ optimal and schedule  ##############################
class AdamW(Optimizer):

    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


############################ optimal and schedule  ##############################


###########################  trainer  ##########################################
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
InputDataClass = NewType("InputDataClass", Any)
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])
PREFIX_CHECKPOINT_DIR = "checkpoint"


def nested_concat(tensors, new_tensors, dim=0):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, dim) for t, n in zip(tensors, new_tensors))
    return torch.cat((tensors, new_tensors), dim=dim)


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def distributed_broadcast_scalars(
        scalars: List[Union[int, float]], num_total_examples: Optional[int] = None
) -> "torch.Tensor":
    try:
        tensorized_scalar = torch.Tensor(scalars).cuda()
        output_tensors = [tensorized_scalar.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensorized_scalar)
        concat = torch.cat(output_tensors, dim=0)

        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


class Trainer:

    def __init__(
            self,
            model: PreTrainedModel = None,
            # ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            # The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
            args=None,
            # The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            # `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
            data_collator: Optional[DataCollator] = None,
            # The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            # default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            # [`DataCollatorWithPadding`] otherwise.
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            # The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            #  maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            #  interrupted training or reuse the fine-tuned model.
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            # The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            # a dictionary string to metric values
            tb_writer: Optional["SummaryWriter"] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            # A tuple
            # containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
            # and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
            **kwargs,
    ):
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        self.model = model.to(args.device) if model is not None else None
        default_collator = default_data_collator
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.tb_writer = tb_writer
        self.log_history = []
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a future version. Use `args.prediction_loss_only` instead.",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.global_step = None
        self.epoch = None
        self.total_flos = None
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        if self.args.label_names is None:
            self.args.label_names = (["labels"])  # 存储答案的索引

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = SequentialSampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def num_examples(self, dataloader):
        return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None, trial=None):

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:  # arg.max_steps:最多经过多少次梯度更新
            t_total = self.args.max_steps
            # num_train_epochs 为经过多少次梯度更新除以一个epoch更新梯度的次数，一个epoch为数据集过一遍
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            # 如果没有指定最多经历多少次的梯度更新，则拿train_epoch的数量乘以一个epoch梯度更新的次数得到t_total
            # 将t_total赋值给max_steps
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        model = self.model
        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                self.total_flos = getattr(model.config, "total_flos", 0)
                # epochs_trained为已经训练的epoch的次数
                # step_trained_in_current_epoch为下一次epoch已经训练的step的次数
                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        disable_tqdm = self.args.disable_tqdm
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)

        best_eval_loss = float('inf')
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue

                tr_loss += self.training_step(model, inputs)  # 每一个gpu上前向传播传播一次的loss
                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # global_step 为梯度更新的次数，当除以save_steps的整数倍数时，保存一次模型
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"

                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        # self.store_flos()
                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)

                        if self.is_world_process_zero():
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # if hasattr(self, "_training_step"):
        #     warnings.warn(
        #         "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
        #         FutureWarning,
        #     )
        #     return self._training_step(model, inputs, self.optimizer)

        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        return outputs[0]

    def is_world_process_zero(self) -> bool:
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        # if is_torch_tpu_available():
        #     self._save_tpu(output_dir)
        if self.is_world_process_zero():
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        json.dump(
            self.log_history, open(os.path.join(output_dir, "log_history.json"), "w"), indent=2, ensure_ascii=False
        )

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        return output

    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only=None):

        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = self.args.disable_tqdm
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            if loss is not None:
                eval_losses.extend([loss] * batch_size)  # [batch_size个loss]
            if logits is not None:
                # 连接起来
                preds = logits if preds is None else nested_concat(preds, logits, dim=0)
            if labels is not None:
                label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = nested_numpify(preds)
        if label_ids is not None:
            label_ids = nested_numpify(label_ids)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            # 计算准确率，f1值
        else:
            metrics = {}
        if len(eval_losses) > 0:
            if self.args.local_rank != -1:
                metrics["eval_loss"] = (
                    distributed_broadcast_scalars(eval_losses, num_total_examples=self.num_examples(dataloader))
                    .mean()
                    .item()
                )
            else:
                metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        # 加上前缀
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                # The .mean() is to reduce in case of distributed training
                loss = outputs[0].mean().item()
                logits = outputs[1:]
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = tuple(logit.detach() for logit in logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):

        if isinstance(self.model, torch.nn.DataParallel) or isinstance(
                self.model, torch.nn.parallel.DistributedDataParallel
        ):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, "floating_point_ops"):
            return model.floating_point_ops(inputs)

        else:
            return 0


###########################  trainer  ##########################################

def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = argparse.ArgumentParser()
    # model_name_or_path Huggingface的预训练模型
    # pretrained fastTest model预训练好的fastText
    # N_gram: 提取的N_gram模型
    parser.add_argument("--model_name_or_path",
                        default='roberta-base',
                        type=str,
                        required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--fasttext_model_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Path to pretrained fastText model for initializing ngram embeddings")
    parser.add_argument("--is_Ngram",
                        default=True,
                        help="whether to use a Ngram module")
    parser.add_argument("--num_hidden_Ngram_layers",
                        default=1,
                        help="the number of ngram layers")
    parser.add_argument("--Ngram_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to Ngram path")
    parser.add_argument("--config_name",
                        default=None,
                        type=str,
                        required=False,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",
                        default=None,
                        type=str,
                        required=False,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    # cache_dir 存放预训练模型的地址
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="Where do you want to store the pretrained models downloaded from s3")

    parser.add_argument("--task_name",
                        default='amazon',
                        type=str,
                        required=True,
                        help="The name of the task")
    parser.add_argument("--data_dir",  # 在utils.py中的GlueDataset中
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        required=False,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir",
                        # action='store_true',
                        default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--do_train",
                        # action='store_false',
                        default=True,
                        help="Whether to run training")
    parser.add_argument("--do_eval",
                        # action='store_false',
                        default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        # action='store_false',
                        default=True,
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",
                        action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--evaluation_strategy",
                        default="no",
                        type=EvaluationStrategy,
                        required=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--prediction_loss_only",
                        action='store_true',
                        help="When performing evaluation and predictions, only returns the loss.")
    parser.add_argument("--per_device_train_batch_size",
                        default=8,
                        type=int,
                        required=False,
                        help="Batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size",
                        default=8,
                        type=int,
                        required=False,
                        help="Batch size per GPU/TPU core/CPU for evaluation.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=None,
                        type=int,
                        required=False,
                        help="Deprecated, the use of `--per_device_train_batch_size` is preferred. ")

    parser.add_argument("--per_gpu_eval_batch_size",
                        default=None,
                        type=int,
                        required=False,
                        help="Deprecated, the use of `--per_device_eval_batch_size` is preferred.")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        required=False,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        required=False,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        required=False,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_beta1",
                        default=0.9,
                        type=float,
                        required=False,
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--adam_beta2",
                        default=0.999,
                        type=float,
                        required=False,
                        help="Beta2 for Adam optimizer")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        required=False,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        required=False,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        required=False,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        required=False,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        required=False,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_dir",
                        default="runs/" + datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname(),
                        type=str,
                        required=False,
                        help="Tensorboard log dir.")
    parser.add_argument("--logging_first_step",
                        action='store_true',
                        help="Log and eval the first global_step")
    parser.add_argument("--logging_steps",
                        default=500,
                        type=int,
                        required=False,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        default=500000,
                        type=int,
                        required=False,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit",
                        default=None,
                        type=int,
                        required=False,
                        help="Limit the total amount of checkpoints.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Do not use CUDA even when it is available")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        required=False,
                        help="random seed for initialization")
    parser.add_argument("--fp16",
                        action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",
                        default="O1",
                        type=str,
                        required=False,
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        required=False,
                        help="For distributed training: local_rank")
    parser.add_argument("--tpu_num_cores",
                        default=None,
                        type=int,
                        required=False,
                        help="TPU: Number of TPU cores (automatically passed by launcher script)")
    parser.add_argument("--tpu_metrics_debug",
                        action='store_true',
                        help="Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Whether to print debug metrics on TPU")
    parser.add_argument("--dataloader_drop_last",
                        action='store_true',
                        help="Drop the last incomplete batch if it is not divisible by the batch size.")
    parser.add_argument("--eval_steps",
                        default=None,
                        type=int,
                        required=False,
                        help="Run an evaluation every X steps.")
    parser.add_argument("--dataloader_num_workers",
                        default=0,
                        type=int,
                        required=False,
                        help="Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.")
    parser.add_argument("--past_index",
                        default=-1,
                        type=int,
                        required=False,
                        help="If >=0, uses the corresponding part of the output as the past state for next step.")
    parser.add_argument("--run_name",
                        default=None,
                        type=str,
                        required=False,
                        help="An optional descriptor for the run. Notably used for wandb logging.")
    parser.add_argument("--disable_tqdm",
                        action='store_true',
                        help="Whether or not to disable the tqdm progress bars.")
    parser.add_argument("--remove_unused_columns",
                        action='store_false',
                        help="Remove columns not required by the model when using an nlp.Dataset.")
    parser.add_argument("--label_names",
                        default=None,
                        type=List[str],
                        required=False,
                        help="The list of keys in your dictionary of inputs that correspond to the label")

    args = parser.parse_args()

    if args.no_cuda:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    elif args.local_rank == -1:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.distributed.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args.seed)

    try:
        num_labels = glue_tasks_num_labels[args.task_name]
        output_mode = "classification"
    except KeyError:
        raise ValueError("Task not found: %s" % (args.task_name))

    Ngram_dict = TDNANgramDict(args.Ngram_path)  # Dict class to store the ngram
    args.Ngram_size = len(Ngram_dict.ngram_to_id_dict)
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        Ngram_vocab_size=len(Ngram_dict.id_to_ngram_list),
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        Ngram_vocab_size=len(Ngram_dict.id_to_ngram_list),
    )

    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
        args=args
    )
    # print(model)
    total_num = sum(p.numel() for p in model.parameters())

    if args.fasttext_model_path is not None:
        pretrained_embedding_np = np.load(args.fasttext_model_path)
        pretrained_embedding = torch.from_numpy(pretrained_embedding_np)
        model.roberta.Ngram_embeddings.word_embeddings.weight.data.copy_(pretrained_embedding)

    # Get datasets
    train_dataset = (
        GlueDataset(args, Ngram_dict, tokenizer=tokenizer, cache_dir=args.cache_dir) if args.do_train else None
    )
    eval_dataset = (
        GlueDataset(args, Ngram_dict, tokenizer=tokenizer, mode="dev", cache_dir=args.cache_dir)
        if args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(args, Ngram_dict, tokenizer=tokenizer, mode="test", cache_dir=args.cache_dir)
        if args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(args.task_name),
    )

    # Training
    if args.do_train:
        trainer.train(
            model_path=args.model_name_or_path if os.path.isdir(args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(args.output_dir)

    # Evaluation
    eval_results = {}
    if args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if args.task_name == "mnli":
            mnli_mm_args = dataclasses.replace(args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_args, tokenizer=tokenizer, mode="dev", cache_dir=args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            metrics = eval_result.metrics
            predictions = np.argmax(eval_result.predictions, axis=1) + 1
            label_ids = eval_result.label_ids + 1

            eval_pred_file = os.path.join(
                args.output_dir, f"eval_pred_{eval_dataset.args.task_name}.txt"
            )
            f_eval = open(eval_pred_file, "w", encoding='utf-8')
            f_eval.write("input" + "\t" + "label" + "\t" + "pred" + "\n")

            with open(os.path.join(args.data_dir, "dev.tsv"), "r", encoding="utf-8") as f:
                input_labels = list(csv.reader(f, delimiter="\t"))

            for line, pred in zip(input_labels, predictions):
                text_a = line[0]
                label = line[1]
                f_eval.write(text_a + '\t' + str(label) + '\t' + str(pred) + '\n')
            f_eval.close()

            output_eval_file = os.path.join(
                args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w", encoding='utf-8') as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in metrics.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(metrics)

    if args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=args.cache_dir)
            )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            test_result = trainer.evaluate(eval_dataset=test_dataset)
            metrics = test_result.metrics
            predictions = np.argmax(test_result.predictions, axis=1) + 1
            label_ids = test_result.label_ids + 1

            test_pred_file = os.path.join(
                args.output_dir, f"test_pred_{test_dataset.args.task_name}.txt"
            )
            f_test = open(test_pred_file, "w", encoding='utf-8')
            f_test.write("input" + "\t" + "label" + "\t" + "pred" + "\n")

            with open(os.path.join(args.data_dir, "test.tsv"), "r", encoding="utf-8") as f:
                input_labels = list(csv.reader(f, delimiter="\t"))
            for line, pred in zip(input_labels, predictions):
                text_a = line[0]
                label = line[1]
                f_test.write(text_a + '\t' + str(label) + '\t' + str(pred) + '\n')
            f_test.close()

            output_test_file = os.path.join(
                args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w", encoding='utf-8') as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in metrics.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
