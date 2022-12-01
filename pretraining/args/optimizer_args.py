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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizerArguments:
    """
    Optimizer Arguments
    """

    _argument_group_name = "Optimizer Arguments"

    optimizer_type: Optional[str] = field(
        default="adamw",
        metadata={
            "help": "Optimizer type",
            "choices": [
                "adam",
                "adamw",
                "lamb",
                "adafactor",
                "customized",
            ],
        },
    )

    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "weight_decay"})
    bias_correction: Optional[bool] = field(default=False, metadata={"help": "bias_correction"})
    max_coeff: Optional[float] = field(default=0.3, metadata={"help": "max_coeff"})
    min_coeff: Optional[float] = field(default=0.01, metadata={"help": "min_coeff"})
    adam_beta1: Optional[float] = field(default=0.9, metadata={"help": "adam beta1"})
    adam_beta2: Optional[float] = field(default=0.999, metadata={"help": "adam beta2"})
    adam_eps: Optional[float] = field(default=1e-6, metadata={"help": "adam epsilon"})
    optimizer_conf_file: Optional[str] = field(default="", metadata={"help": "conf file path for customized optimizers"})

    def __post_init__(self):
        self.optimizer_type = str(self.optimizer_type).lower()
