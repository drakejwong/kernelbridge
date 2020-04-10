#!/usr/bin/env python

"""Tests for `kernelbridge` package."""

import pytest

from click.testing import CliRunner

from kernelbridge import basekernel
from kernelbridge import cli

import numpy as np
# import gpflow

inf = float("inf")
nan = np.nan
kernels = [
    Constant(1),
    SquaredExponential(1),
    RationalQuadratic(1, 1),
]

