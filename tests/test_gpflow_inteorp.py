#!/usr/bin/env python

"""Tests for `kernelbridge` package."""

import pytest

from kernelbridge.basekernel import Constant
from kernelbridge.basekernel import SquaredExponential
from kernelbridge.basekernel import RationalQuadratic
from kernelbridge.bridge.gpflow import GPflowKernel

import numpy as np
import gpflow

kernels = [
    Constant(1),
    SquaredExponential(1),
    RationalQuadratic(1, 1),
]


@pytest.mark.parametrize("k", kernels)
def test_compute(k):
    k = GPflowKernel(k)
    k(np.arange(0, 2, 0.2).reshape(-1, 1))
    k(np.arange(0, 2, 0.2).reshape(-1, 1), np.arange(1, 3, 0.2).reshape(-1, 1))

@pytest.mark.parametrize("k", kernels)
def test_regression(k):
    k = gpflow.kernels.Matern52()

    X = np.array([0.1, 0.3, 0.4, 0.7]).reshape(-1, 1)
    Y = np.array([3.3, 3.7, 4.0, 9.1]).reshape(-1, 1)

    m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxitr=100))

    print(m.predict_f(np.linspace(-0.1, 0.5, 4)))