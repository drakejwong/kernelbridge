#!/usr/bin/env python

"""Tests for `kernelbridge` package."""

import pytest

from kernelbridge.basekernel import Constant
from kernelbridge.basekernel import SquaredExponential
from kernelbridge.basekernel import RationalQuadratic
from kernelbridge.basekernel import RConvolution

import numpy as np
import random
# import copy

inf = float("inf")
nan = np.nan
kernels = [
    Constant(1),
    SquaredExponential(1),
    RationalQuadratic(1, 1),
]


@pytest.mark.parametrize("kernel", kernels)
def test_simple_kernel(kernel):
    ''' default behavior '''
    assert(kernel(0, 0) == 1)
    ''' corner cases '''
    assert(kernel(0, inf) <= 1)
    assert(kernel(0, -inf) <= 1)
    assert(kernel(inf, 0) <= 1)
    assert(kernel(-inf, 0) <= 1)
    ''' random input '''
    random.seed(0)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kernel(i, j) >= 0 and kernel(i, j) <= 1)
        assert(kernel(i, j) == kernel(j, i))  # check symmetry
    # ''' representation meaningness '''
    # assert(eval(repr(kernel)).theta == kernel.theta)
    # ''' hyperparameter retrieval '''
    # assert(isinstance(kernel.theta, tuple))
    # assert(len(kernel.theta) > 0)
    # kernel.theta = kernel.theta
    # another = copy.copy(kernel)
    # for t1, t2 in zip(kernel.theta, another.theta):
    #     assert(t1 == t2)

@pytest.mark.parametrize('kernel', kernels)
def test_kernel_add_constant(kernel):
    ''' check by definition '''
    for kadd in [kernel + 1, 1 + kernel]:
        random.seed(0)
        # mirror = eval(repr(kadd))  # representation meaningness test
        # assert(mirror.theta == kadd.theta)
        for _ in range(1000):
            i = random.paretovariate(0.1)
            j = random.paretovariate(0.1)
            assert(kadd(i, j) == kernel(i, j) + 1)
            assert(kadd(i, j) == kadd(j, i))
            # assert(kadd(i, j) == mirror(i, j))
            # assert(kadd(i, j) == mirror(j, i))
        # ''' representation generation '''
        # assert(len(str(kadd).split('+')) == 2)
        # assert(str(kernel) in str(kadd))
        # assert(len(repr(kadd).split('+')) == 2)
        # assert(repr(kernel) in repr(kadd))
        # ''' hyperparameter retrieval '''
        # assert(kernel.theta in kadd.theta)
        # kadd.theta = kadd.theta


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_kernel_add_kernel(k1, k2):
    ''' check by definition '''
    kadd = k1 + k2
    random.seed(0)
    # mirror = eval(repr(kadd))  # representation meaningness test
    # assert(mirror.theta == kadd.theta)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kadd(i, j) == k1(i, j) + k2(i, j))
        assert(kadd(i, j) == kadd(j, i))
        # assert(kadd(i, j) == mirror(i, j))
        # assert(kadd(i, j) == mirror(j, i))
    ''' representation generation '''
    # assert(len(str(kadd).split('+')) == 2)
    # assert(str(k1) in str(kadd))
    # assert(str(k2) in str(kadd))
    # assert(len(repr(kadd).split('+')) == 2)
    # assert(repr(k1) in repr(kadd))
    # assert(repr(k2) in repr(kadd))
    # ''' hyperparameter retrieval '''
    # assert(k1.theta in kadd.theta)
    # assert(k2.theta in kadd.theta)
    # kadd.theta = kadd.theta


@pytest.mark.parametrize('kernel', kernels)
def test_kernel_mul_constant(kernel):
    ''' check by definition '''
    for kmul in [kernel * 2, 2 * kernel]:
        random.seed(0)
        # mirror = eval(repr(kmul))  # representation meaningness test
        # assert(mirror.theta == kmul.theta)
        for _ in range(1000):
            i = random.paretovariate(0.1)
            j = random.paretovariate(0.1)
            assert(kmul(i, j) == kernel(i, j) * 2)
            assert(kmul(i, j) == kmul(j, i))
            # assert(kmul(i, j) == mirror(i, j))
            # assert(kmul(i, j) == mirror(j, i))
        # ''' representation generation '''
        # assert(len(str(kmul).split('*')) == 2)
        # assert(str(kernel) in str(kmul))
        # assert(len(repr(kmul).split('*')) == 2)
        # assert(repr(kernel) in repr(kmul))
        # ''' hyperparameter retrieval '''
        # assert(kernel.theta in kmul.theta)
        # kmul.theta = kmul.theta


@pytest.mark.parametrize('k1', kernels)
@pytest.mark.parametrize('k2', kernels)
def test_kernel_mul_kernel(k1, k2):
    ''' check by definition '''
    kmul = k1 * k2
    random.seed(0)
    # mirror = eval(repr(kmul))  # representation meaningness test
    # assert(mirror.theta == kmul.theta)
    for _ in range(1000):
        i = random.paretovariate(0.1)
        j = random.paretovariate(0.1)
        assert(kmul(i, j) == k1(i, j) * k2(i, j))
        assert(kmul(i, j) == kmul(j, i))
        # assert(kmul(i, j) == mirror(i, j))
        # assert(kmul(i, j) == mirror(j, i))
    # ''' representation generation '''
    # assert(len(str(kmul).split('*')) == 2)
    # assert(str(k1) in str(kmul))
    # assert(str(k2) in str(kmul))
    # assert(len(repr(kmul).split('*')) == 2)
    # assert(repr(k1) in repr(kmul))
    # assert(repr(k2) in repr(kmul))
    # ''' hyperparameter retrieval '''
    # assert(k1.theta in kmul.theta)
    # assert(k2.theta in kmul.theta)
    # kmul.theta = kmul.theta

@pytest.mark.parametrize('k', [RConvolution(r"GCA[AGCT]*ACT")])
def test_rconv_kernel(k):
    assert(k("AGGGCAACGTACGATCAACT", "AGGGCAACGTACGATCAACT") == 4)
