#!/usr/bin/env python

"""Tests for `kernelbridge` package."""

import pytest

from click.testing import CliRunner

from kernelbridge import basekernel
from kernelbridge import cli

import numpy as np


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_kernel_creation():
    """Test basekernel creation of RBF kernel with full hyperparameter specs."""
    basekernel.BaseKernel.create(
        "RBF",
        "Radial basis function kernel",
        "exp(-0.5 * (x1 - x2) ** 2 * length_scale**-2)",
        ('x1', 'x2'),
        ('length_scale', np.float32, 1e-6, float("inf"),
         "Determines rate of kernel decay to zero.")
    )

def test_kernel_addition():
    """Test basekernel addition of stationary kernel with constant kernel."""
    k1 = basekernel.BaseKernel.create(
        "RBF",
        "Radial basis function kernel",
        "exp(-0.5 * (x1 - x2) ** 2 * length_scale**-2)",
        ('x1', 'x2'),
        ('length_scale', np.float32, 1e-6, float("inf"),
        "Determines rate of kernel decay to zero.")
    )
    
    basekernel.Sum(k1, 2)


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'kernelbridge.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
