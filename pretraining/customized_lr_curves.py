#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
'''
All rights reserved.
'''
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=no-member

from __future__ import absolute_import
from __future__ import print_function

import logging
import logging.config
import math


class Error(Exception):
    """Root for all errors."""

class ParseError(Error):
    """Unable to parse input data."""

def get_activation_curve(conf, return_by_t=True):
    """Parses the config for activation curves.

    Before the specified activation point, we use constant scheduling by
    default. Right after the activation point, the chosen curve is
    activated, and iteration number is counted since that point.

    Activation scheduling is always an additional component of other
    curves.  So its interface is different from other common curves.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
            Following configure information is used. For example,
                ```
                [hyperparams]
                activation_point = 100
                ```
    Returns:
        if return_by_t=True, returns a function which adjusts t according to
        activation settings,
            ```
            new_t = lr_curve(t)
            ```
        function arguments:
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.

        For example, with activation point being 8, the returned new_t will
            be
            ```
            t       0  1  2  3  4  5  6  7  8  9  10  11  12  13
            new_t   0  0  0  0  0  0  0  0  0  1  2   3   4   5
                                            ^
                                            activated
                    |--- use constant lr ---|
            ```

        Otherwise, returns a function which specifies whether the curve is
        activated,
            ```
            activated = lr_curve(t)
            # activated: True or False
            ```
    Raises:
        ParseError if hyperparameters are invalid.
    """
    if not conf.has_option('hyperparams', 'activation_point'):
        activation_point = 0
        logging.info('lr curve: activation option = False')
    else:
        activation_point = conf.getint('hyperparams', 'activation_point')
        logging.info('lr curve: activation option = True')
        logging.info('lr curve: activation point = %d', activation_point)

    if return_by_t:
        activation_curve = lambda t: max(0, t - activation_point)
    else:
        activation_curve = lambda t: t >= activation_point
    return activation_curve


def get_restart_curve(conf):
    """Parses the config for restart curves.

    Restart the chosen curve when at the specified restarting points.

    Restart scheduling is always an additional component of other curves.
    So its interface is different from other common curves.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
            Following configure information is used. For example,
                ```
                [hyperparams]
                restarting_points = 0, 10000        # Assumes to be sorted
                ```
    Returns:
        a function which adjusts t according to restart settings,
            ```
            new_t = lr_curve(t)
            ```
        function arguments:
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.

        For example, with restarting point being [3, 5, 10], the returned new_t
            will be
            ```
            t       0  1  2  3  4  5  6  7  8  9  10  11  12  13
            new_t   0  1  2  0  1  0  1  2  3  4  0   1   2   3
                             ^     ^              ^
                        restart  restart        restart
            ```
    Raises:
        ParseError if hyperparameters are invalid.
    """
    if not conf.has_option('hyperparams', 'restarting_points'):
        restart_point_list = []
        logging.info('lr curve: restart = False')
    else:
        raw_point_list = conf.get('hyperparams', 'restarting_points')
        restart_point_list = [int(p) for p in raw_point_list.split(',')]
        logging.info('lr curve: restart = True')
        logging.info('lr curve: restarting_points = %s',
                     str(restart_point_list))

    def restart_curve(t):
        new_t = t
        for restart_point in restart_point_list:
            if t >= restart_point:
                new_t = t - restart_point
        return new_t

    return restart_curve


def get_inverse_time_decay_curve(conf):
    """Parses the config for learning rate curve.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    lambda_ = conf.getfloat('hyperparams', 'lambda')
    logging.info('lr curve: type = inverse_time_decay_curve')
    logging.info('lr curve: lambda = %.6lf', lambda_)

    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def inverse_time_decay_curve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)
        return init_lr / (1 + init_lr * t * lambda_)

    return inverse_time_decay_curve

def get_piecewise_constant_curve(conf):
    """Parses the config for learning rate curve.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    raw_start_point_list = conf.get('hyperparams', 'starting_points')
    raw_factor_list = conf.get('hyperparams', 'factors')
    start_point_list = [int(s) for s in raw_start_point_list.split(',')]
    factor_list = [float(c) for c in raw_factor_list.split(',')]
    logging.info('lr curve: type = piecewise_constant_curve')
    logging.info('lr curve: start_points = %s', str(start_point_list))
    logging.info('lr curve: factors = %s', str(factor_list))

    # Handles errors
    if len(start_point_list) != len(factor_list):
        raise ParseError(
            'lr curve: number of starting points differs from factors')
    if start_point_list != sorted(start_point_list):
        raise ParseError(
            'lr curve: starting points should be sorted')

    # The generated lr curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def piecewise_constant_curve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)
        if not start_point_list or t < start_point_list[0]:
            return init_lr
        if t >= start_point_list[-1]:
            return init_lr * factor_list[-1]

        for i, _ in enumerate(start_point_list):
            if start_point_list[i] <= t < start_point_list[i+1]:
                return init_lr * factor_list[i]
        return init_lr

    return piecewise_constant_curve


def get_cosine_decay_curve(conf):
    """Parses the config for learning rate curve.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    t_0 = conf.getint('hyperparams', 't_0')
    t_mul = conf.getfloat('hyperparams', 't_mul')
    min_lr = conf.getfloat('hyperparams', 'min_lr')
    power = 1
    if conf.has_option('hyperparams', 'power'):
        power = conf.getfloat('hyperparams', 'power')

    logging.info('lr curve: type = cosine_decay_curve')
    logging.info('lr curve: t_0 = %d', t_0)
    logging.info('lr curve: t_mul = %.8lf', t_mul)
    logging.info('lr curve: min_lr = %.8lf', min_lr)
    logging.info('lr curve: power = %.8lf', power)

    # Handles errors
    if t_0 < 0:
        raise ParseError('lr curve: t_0 should >= 0 (t_0 = %d)' % t_0)
    if t_mul <= 1-1e-8:
        raise ParseError('lr curve: t_mul should >= 1 (t_mul = %.6lf)'
                         % t_mul)

    # Generated learning rate curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def cosine_decay_curve(init_lr, t):
        if cosine_decay_curve.last_t == t:
            return cosine_decay_curve.last_lr
        elif cosine_decay_curve.last_t != t - 1:
            logging.warning(
                'WARNING: lr curve: last_t != t-1 (%d != %d), incontinuous call'
                ' to lr scheduler happens, which will result in wrong values',
                cosine_decay_curve.last_t, t - 1
            )

        cosine_decay_curve.last_t = t

        t = activation_curve(t)
        t = restart_curve(t)
        if min_lr > init_lr:
            raise ParseError(
                'lr curve: min_lr should <= init_lr, (%.6lf > %.6lf)' % (
                    min_lr, init_lr))

        t_max = cosine_decay_curve.t_max
        t_cur = cosine_decay_curve.t_cur
        cos_decay_rate = 0.5 * (1 + math.cos(math.pi * t_cur / float(t_max)))
        learning_rate = min_lr + (init_lr - min_lr) * cos_decay_rate
        learning_rate = learning_rate ** power
        cosine_decay_curve.t_cur += 1

        # End of one segment
        if t_cur == t_max:
            cosine_decay_curve.t_max *= t_mul
            cosine_decay_curve.t_cur = 0

        cosine_decay_curve.last_lr = learning_rate
        return learning_rate

    cosine_decay_curve.t_max = t_0
    cosine_decay_curve.t_cur = 0

    cosine_decay_curve.last_t = -1
    cosine_decay_curve.last_lr = 0.

    return cosine_decay_curve


def get_exponential_decay_curve(conf):
    """Parses the config for learning rate curve.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    decay_step = conf.getint('hyperparams', 'decay_step')
    decay_rate = conf.getfloat('hyperparams', 'decay_rate')
    logging.info('lr curve: type = exponential_decay_curve')
    logging.info('lr curve: decay_step = %d', decay_step)
    logging.info('lr curve: decay_rate = %.6lf', decay_rate)

    # Handles errors
    if decay_step < 0:
        raise ParseError('lr curve: decay_step should >= 0 (decay_step = %d)'
                         % decay_step)
    if decay_rate > 1+1e-8 or decay_rate < 0:
        raise ParseError('lr curve: decay_rate should in [0, 1]'
                         ' (decay_rate = %.6lf)' % decay_rate)

    # Generated learning rate curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def exponential_decay_curve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)
        return init_lr * (decay_rate ** (t / decay_step))

    return exponential_decay_curve


def get_piecewise_inverse_time_curve(conf):
    """Parses the config for learning rate curve.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    raw_start_point_list = conf.get('hyperparams', 'starting_points')
    raw_a_list = conf.get('hyperparams', 'a')
    raw_b_list = conf.get('hyperparams', 'b')

    start_point_list = [int(s) for s in raw_start_point_list.split(',')]
    a_list = [float(a) for a in raw_a_list.split(',')]
    b_list = [float(b) for b in raw_b_list.split(',')]

    logging.info('lr curve: type = piecewise_inverse_time_curve')
    logging.info('lr curve: start_points = %s', str(start_point_list))
    logging.info('lr curve: a = %s', str(a_list))
    logging.info('lr curve: b = %s', str(b_list))

    # Handles errors
    if len(start_point_list) != len(a_list):
        raise ParseError(
            'lr curve: number of starting points differs from a')
    if len(start_point_list) != len(b_list):
        raise ParseError(
            'lr curve: number of starting points differs from b')
    if len(start_point_list) == 0:
        raise ParseError(
            'lr curve: starting points should have at least 1 point')
    if start_point_list[0] != 0:
        raise ParseError('lr_curve: first starting point must be 0')
    if start_point_list != sorted(start_point_list):
        raise ParseError('lr curve: starting points should be sorted')

    # The generated lr curve
    activation_indicator = get_activation_curve(conf, return_by_t=False)
    activation_curve = get_activation_curve(conf, return_by_t=True)
    restart_curve = get_restart_curve(conf)

    def piecewise_inverse_time_curve(init_lr, t):
        activated = activation_indicator(t)
        if not activated:
            return init_lr

        t = activation_curve(t)
        t = restart_curve(t)

        if t >= start_point_list[-1]:
            a = a_list[-1]
            b = b_list[-1]
            start_point = start_point_list[-1]
            return init_lr / (a * (t - start_point) + b)

        if t < start_point_list[0]:
          return init_lr

        left = 0
        right = len(start_point_list) - 1
        while left < right:
            middle = (left + right) // 2
            if start_point_list[middle + 1] <= t:
              left = middle + 1
            else:
              right = middle

        a = a_list[left]
        b = b_list[left]
        start_point = start_point_list[left]

        return init_lr / (a * (t - start_point) + b)

    if not conf.has_option('hyperparams', 'min_lr'):
        return piecewise_inverse_time_curve

    # Otherwise, 'min_lr' is specified.
    #   Does Linear scaling for all learning rates to make the last learning
    #   rate equal to 'min_lr'
    if not conf.has_option('hyperparams', 'num_iter'):
        raise ParseError('lr curve: missing option "num_iter"')

    min_lr = conf.getfloat('hyperparams', 'min_lr')
    num_iter = conf.getint('hyperparams', 'num_iter')
    logging.info('lr curve: min_lr = %.10lf', min_lr)
    logging.info('lr curve: num_iter = %d', num_iter)

    if min_lr < 0:
        raise ParseError('lr curve: negative min_lr %.10lf' % min_lr)
    if num_iter <= 0:
        raise ParseError('lr curve: non-positive num_iter %d' % num_iter)

    # Return scaled learning rate scheduling
    def scaled_piecewise_inverse_time_curve(init_lr, t):
        lr = piecewise_inverse_time_curve(init_lr, t)
        # Last iteration is num_iter - 1
        old_min_lr = piecewise_inverse_time_curve(init_lr, num_iter - 1)

        if init_lr < min_lr:
            raise RuntimeError('init_lr < min_lr (%.10lf < %.10lf)' % (
                init_lr, min_lr))
        if init_lr < old_min_lr:
            raise RuntimeError('init_lr < old_min_lr (%.10lf < %.10lf)' % (
                init_lr, old_min_lr))

        return ((lr - old_min_lr) / (init_lr - old_min_lr) * (init_lr - min_lr)
                + min_lr)

    return scaled_piecewise_inverse_time_curve


def get_continuous_eigencurve(conf):
    """Parses the config for learning rate curve.

    Original form:
        eta_t = 1 / [L + mu * T * f(t)]
        f(t) = 1/[kappa^((1-alpha)/2) - 1]
               * (alpha-1) / (alpha-3)
               * [(1 - t/T (1 - kappa^((1-alpha)/2))^(1 - 2/(alpha-1)) - 1]
        # alpha != 1, alpha != 3

    Generalized form:
        eta_t = 1 / [L + mu * T' * f(T')]
              = 1/L / [1 + 1/kappa * T' * f(T')]
              = eta_0 / [1 + 1/kappa * T' * f(T')]
        where T' = T + 1

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    num_iter = conf.getint('hyperparams', 'num_iter')
    alpha = conf.getfloat('hyperparams', 'alpha')
    kappa = conf.getfloat('hyperparams', 'kappa')
    min_lr = conf.getfloat('hyperparams', 'min_lr')
    logging.info('lr curve: type = continuous_eigencurve')
    logging.info('lr curve: num_iter = %d', num_iter)
    logging.info('lr curve: alpha = %.6lf', alpha)
    logging.info('lr curve: kappa = %.6lf', kappa)
    logging.info('lr curve: min_lr = %.6lf', min_lr)

    # Handles errors
    epsilon = 1e-10
    if num_iter <= 0:
        raise ParseError('lr curve: num_iter = %d should be positive'
                         % num_iter)
    if min_lr < epsilon:
        raise ParseError('lr curve: min_lr = %.10lf should be positive'
                         % min_lr)
    if kappa < 1:
        raise ParseError('lr curve: kappa = %.10lf should >= 1'
                         % kappa)
    if abs(alpha - 1) < epsilon or abs(alpha - 3) < epsilon:
        raise ParseError('lr curve: alpha = %.10lf cannot be 1 or 3'
                         % alpha)

    # Generated learning rate curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def continuous_eigencurve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)

        # f(t) = 1/[kappa^((1-alpha)/2) - 1]
        #        * (alpha-1) / (alpha-3)
        #        * [(1 - t/T (1 - kappa^((1-alpha)/2))^(1 - 2/(alpha-1)) - 1]
        #
        # eta_t = 1 / [L + mu * T' * f(T')]
        #       = 1/L / [1 + 1/kappa * T' * f(T')]
        #       = eta_0 / [1 + 1/kappa * T' * f(T')]
        # where T' = T + 1
        factor_t = 1 / (kappa ** ((1 - alpha) / 2) - 1)
        factor_t *= (alpha - 1) / (alpha - 3)
        factor_t *= ((1 - t/num_iter * (1 - kappa ** ((1-alpha)/2))) ** (1 - 2/(alpha-1))) - 1
        learning_rate = init_lr / (1 + 1/kappa * num_iter * factor_t)

        return learning_rate

    def scaled_continusous_eigencurve(init_lr, t): 
        lr = continusous_eigencurve(init_lr, t)
        old_min_lr = continusous_eigencurve(init_lr, num_iter - 1)

        if init_lr < min_lr:
            raise RuntimeError('init_lr < min_lr (%.10lf < %.10lf)' % (
                init_lr, min_lr))
        if init_lr < old_min_lr:
            raise RuntimeError('init_lr < old_min_lr (%.10lf < %.10lf)' % (
                init_lr, old_min_lr))

        return ((lr - old_min_lr) / (init_lr - old_min_lr) * (init_lr - min_lr)
                + min_lr)

    return scaled_continuous_eigencurve


def get_poly_remain_time_decay_curve(conf):
    """Parses the config for learning rate curve.

    eta_t = eta_0 (1 - t/T)^C

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    decay_rate = conf.getfloat('hyperparams', 'decay_rate')
    num_iter = conf.getint('hyperparams', 'num_iter')
    logging.info('lr curve: type = poly_remain_time_decay_curve')
    logging.info('lr curve: decay_rate = %.6lf', decay_rate)
    logging.info('lr curve: num_iter = %d', num_iter)

    # Handles errors
    if decay_rate < 0:
        raise ParseError('lr curve: decay_rate should in [0, +inf)'
                         ' (decay_rate = %.6lf)' % decay_rate)
    if num_iter <= 0:
        raise ParseError('lr curve: num_iter = %d should be positive'
                         % num_iter)

    # Generated learning rate curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def poly_remain_time_decay_curve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)
        return init_lr * ((1 - t / num_iter) ** decay_rate)

    return poly_remain_time_decay_curve


def get_elastic_step_decay_curve(conf):
    """Parses the config for learning rate curve.

    The decay rate is fixed to 2.0, i.e. halving the learning rate when
    proceeding to next interval. And the interval length is exponentially
    decreasing.

    eta_t = eta_0 / 2^k,    if t lies in interval k, where k = 0,1,2,...
    e.g [00000000)[1111)[22)[3) for interval_shrink_rate = 2
         ^                   ^
         t=0                 t=15

    The interval shrink rate is a real number larger than 1.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    interval_shrink_rate = conf.getfloat('hyperparams', 'interval_shrink_rate')
    num_iter = conf.getint('hyperparams', 'num_iter')
    cr_k = conf.getint('hyperparams', 'cr_k')
    logging.info('lr curve: type = elastic_step_decay')
    logging.info('lr curve: interval_shrink_rate = %.6lf',
                 interval_shrink_rate)
    logging.info('lr curve: cr_k = %d', cr_k)
    logging.info('lr curve: num_iter = %d', num_iter)

    # Handles errors
    if interval_shrink_rate <= 1:
        raise ParseError('lr curve: interval_shrink_rate should in (1, +inf)'
                         ' (interval_shrink_rate = %.6lf)'
                         % interval_shrink_rate)
    if num_iter <= 0:
        raise ParseError('lr curve: num_iter = %d should be positive'
                         % num_iter)

    if cr_k < 2:
        raise ParseError('lr curve: cr_k = %d should be >= 2' % cr_k)

    # Generated learning rate curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def elastic_step_decay_curve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)

        # Denotes
        #   Delta_k := portion of interval k in all iterations,
        #   a = 1 / interval_shrink_rate
        # We have
        #   1) sum_{i=0}^{inf} Delta_k = 1
        #   2) Delta_k = Delta_{k-1} * a,     forall k >= 1
        # Thus,
        #   1 = sum_{k=0}^{inf} Delta_k
        #     = sum_{k=0}^{inf} Delta_0 a^k
        #     = Delta_0 sum_{k=0}^{inf} a^k
        #     = Delta_0 * 1/(1-a)
        #
        # => Delta_0 = 1-a
        # => Delta_k = (1-a)a^k
        # => The accumulated length before interval K is,
        #     sum_{k'=0}^{K-1} Delta_k'
        #     = sum_{k'=0}^{K-1} (1-a)a^k'
        #     = 1-a^K
        #
        # We would like to find the largest interval id K that satisifes,
        #    t/T >= sum_{k'=0}^{K-1} Delta_k'
        # => t/T >= 1-a^K
        # => a^K >= 1-t/T
        # => K lna >= ln(1-t/T)
        # => K <= ln(1-t/T) / lna               # Notice lna < 0 since a < 1
        # => Largest K = floor(ln(1-t/T) / lna)

        if t == num_iter:
            return 0

        a = 1 / float(interval_shrink_rate)
        t_div_T = t / float(num_iter)
        k = int(math.floor(math.log(1 - t_div_T) / math.log(a)) + 0.5)
        return init_lr * (min(min(1, 0.5/a)**(k-cr_k+1), 1))       # Fixed decay rate

    return elastic_step_decay_curve


def get_step_decay_curve(conf):
    """Parses the config for learning rate curve.

    There are two tunable hyper-parameters: 'num_interval' and 'decay_rate',
    where 'num_interval' decides the number of piecewise constant intervals
    with length T/num_interval (T is the number of iterations), and
    'decay_rate' decides the learning rate decaying rate between two continuous
    intervals.

    For example, 'num_interval=3' and 'decay_rate=10' results in the common
    step decay curve (assume init_learning_rate = eta_0),
        learning rate = eta_0                       in [0, T/3)
                      = eta_0 / 10 = 0.1 eta_0      in [T/3, 2T/3)
                      = eta_0 / 100 = 0.01 eta_0    in [2T/3, T)

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    decay_rate = conf.getfloat('hyperparams', 'decay_rate')
    num_interval = conf.getint('hyperparams', 'num_interval')
    num_iter = conf.getint('hyperparams', 'num_iter')
    logging.info('lr curve: type = step_decay')
    logging.info('lr curve: decay_rate = %.6lf', decay_rate)
    logging.info('lr curve: num_interval = %d', num_interval)
    logging.info('lr curve: num_iter = %d', num_iter)

    # Handles errors
    if decay_rate <= 1:
        raise ParseError('lr curve: decay_rate should in (1, +inf)'
                         ' (decay_rate = %.6lf)'
                         % interval_shrink_rate)
    if num_interval <= 0:
        raise ParseError('lr curve: num_interval = %d should be positive'
                         % num_interval)
    if num_iter <= 0:
        raise ParseError('lr curve: num_iter = %d should be positive'
                         % num_iter)

    # Generated learning rate curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def step_decay_curve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)

        # t < T, thus k = [t / (T/#interval)] < [T / (T/#interval)] = #interval
        step_size = num_iter / float(num_interval)
        k = int(t / step_size)
        return init_lr * ((1/float(decay_rate))**k)

    return step_decay_curve


def get_linear_decay_curve(conf):
    """Parses the config for learning rate curve.

    From peak learning rate linearly goes down to 0.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
                learning rate curve.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_curve(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
    """
    # Parses raw info
    num_iter = conf.getint('hyperparams', 'num_iter')
    logging.info('lr curve: type = linear_decay')
    logging.info('lr curve: num_iter = %d', num_iter)

    # Handles errors
    if num_iter <= 0:
        raise ParseError('lr curve: num_iter = %d should be positive'
                         % num_iter)

    # Generated learning rate curve
    activation_curve = get_activation_curve(conf)
    restart_curve = get_restart_curve(conf)

    def linear_decay_curve(init_lr, t):
        t = activation_curve(t)
        t = restart_curve(t)

        return init_lr * ((num_iter - t)/float(num_iter))

    return linear_decay_curve
