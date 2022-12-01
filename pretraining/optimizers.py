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

from configparser import ConfigParser

import logging

from torch.optim import Optimizer, Adam, AdamW, SGD
from transformers.optimization import Adafactor

logger = logging.getLogger(__name__)


def get_lamb(optimizer_args, lr, model_params):
    try:
        import deepspeed
    except ImportError or ModuleNotFoundError:
        logger.info("Deepspeed not installed. To use Lamb optimizer please install Deepspeed")
        raise

    from deepspeed.ops.lamb import FusedLamb

    # DS optimizer name hack
    class lamb(FusedLamb):
        pass

    return lamb(
        model_params,
        lr=lr,
        bias_correction=optimizer_args.bias_correction,
        weight_decay=optimizer_args.weight_decay,
        max_coeff=optimizer_args.max_coeff,
        min_coeff=optimizer_args.min_coeff,
    )


def get_adafactor(args, lr, params):
    return Adafactor(params, lr=lr, relative_step=False, scale_parameter=False)


def get_adam(args, lr, params):
    return Adam(
        params,
        lr=lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def get_adamw(args, lr, params):
    return AdamW(
        params,
        lr=lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

def get_sgd(args, lr, params):
    # Parses raw info
    conf = ConfigParser()
    conf.read(args.optimizer_conf_file)
    momentum = conf.getfloat("hyperparams", "momentum")
    nesterov_str = conf.get("hyperparams", "nesterov")
    nesterov = False if nesterov_str in ["no", "false", "No", "False"] else True
    logging.info('customized optimizer: momentum = %.6lf', momentum)
    logging.info('customized optimizer: nesterov = %s', str(nesterov))

    # Handles errors
    if momentum < 0 or momentum > 1:
        raise ValueError("optimizer: momentum should in [0, 1]"
                         " (momentum = %.6lf)"
                         % momentum)

    return SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=args.weight_decay,
        nesterov=nesterov
    )

class DummyOptimizer(Optimizer):
    """
    An dummy optimizer that does nothing.

    Parameters:
        params (:obj:`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 0):
            The learning rate to use.
    """

    def __init__(
        self, params, lr=0.,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        return loss

def get_dummy(args, lr, params):
    return DummyOptimizer(params, lr=lr)

def get_customized(args, lr, params):
    conf = ConfigParser()
    conf.read(args.optimizer_conf_file)
    optimizer_type = conf.get("general", "type")
    logging.info('customized optimizer: type = %s', optimizer_type)

    optimizer_map = {
        "sgd": get_sgd,
        "dummy": get_dummy,
        "adam": get_adam,
        "adamw": get_adamw,
    }
    if not optimizer_type in optimizer_map:
        raise ValueError('unsupported optimizer "%s"' % optimizer_type)

    optimizer = optimizer_map[optimizer_type](args, lr, params)

    return optimizer


OPTIMIZERS = {
    "adam": get_adam,
    "adamw": get_adamw,
    "adafactor": get_adafactor,
    "lamb": get_lamb,
    "customized": get_customized,
}


def get_optimizer(args, lr, params):
    optimizer_type = args.optimizer_type
    if optimizer_type not in OPTIMIZERS:
        raise Exception(
            f"{optimizer_type} is not available. Please choose one of the following: {list(OPTIMIZERS.keys())}"
        )

    return OPTIMIZERS[optimizer_type](args, lr, params)
