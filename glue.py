from typing import Callable, Dict, List
from sklearn.metrics import f1_score
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
import csv
import os
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


class generalProcessor:
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) == 2:
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class sciieProcessor(generalProcessor):
    def get_labels(self):
        return ['FEATURE-OF', 'PART-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'COMPARE', 'USED-FOR', 'HYPONYM-OF']


class SST2Processor(generalProcessor):
    def get_labels(self):
        return ['0', '1']


class rct_20kProcessor(generalProcessor):
    def get_labels(self):
        return ['CONCLUSIONS', 'RESULTS', 'METHODS', 'OBJECTIVE', 'BACKGROUND']


class imdbProcessor(generalProcessor):
    def get_labels(self):
        return ['0', '1']


class hyperpartisan_newsProcessor(generalProcessor):
    def get_labels(self):
        return ['true', 'false']


class chemprotProcessor(generalProcessor):
    def get_labels(self):
        return ['INDIRECT-UPREGULATOR', 'UPREGULATOR', 'INHIBITOR', 'DOWNREGULATOR', 'SUBSTRATE', 'ACTIVATOR',
                'AGONIST-ACTIVATOR',
                'PRODUCT-OF', 'AGONIST-INHIBITOR', 'INDIRECT-DOWNREGULATOR', 'SUBSTRATE_PRODUCT-OF', 'ANTAGONIST',
                'AGONIST']


class amazonProcessor(generalProcessor):
    def get_labels(self):
        return ['helpful', 'unhelpful']


class agProcessor(generalProcessor):
    def get_labels(self):
        return ['1', '2', '3', '4']


class citation_intentProcessor(generalProcessor):
    def get_labels(self):
        return ['CompareOrContrast', 'Background', 'Uses', 'Extends', 'Motivation', 'Future']


glue_processors = {
    "citation_intent": citation_intentProcessor,
    "ag": agProcessor,
    "amazon": amazonProcessor,
    "chemprot": chemprotProcessor,
    "hyperpartisan_news": hyperpartisan_newsProcessor,
    "imdb": imdbProcessor,
    "rct-20k": rct_20kProcessor,
    "sciie": sciieProcessor,
    "SST2": SST2Processor
}

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