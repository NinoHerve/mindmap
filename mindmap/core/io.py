"""
Module Name: core/io.py
Description: This module provides functions to read/write simple datatypes.

Contains functions to read simple data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from pathlib import Path
from typing import Any
import os

import yaml
import numpy as np


def read_config(file: str | Path, env: bool = True) -> dict[str, Any]:
    """
    Read configuration file.

    Parameters
    ----------
    file: path-like
        Path to configuration file.

    env: bool
        If True, then the variables in the configuration file will be saved in the os 
        environment.

    Returns
    -------
    variables: dict[str, Any]
        If env is False, then the variables are returned in a dict object.
    """

    with open(file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


    for key, value in config.items():
        if env:
            os.environ[key] = str(value)
        if isinstance(value, str) and "/" in value:
            config[key] = Path(value)

    return config


def write_config(fname, config):
    """Write configuration file."""
    for key in config.keys():
        if isinstance(config[key], Path):
            config[key] = str(config[key])

    with open(fname, 'w', encoding="utf-8") as outfile:
        yaml.dump(config, outfile, sort_keys=False, default_flow_style=False)


def read_eig(file: str | Path):
    """
    Read eigendecomposition.
    """
    eig = np.load(file)

    eigvec = eig["eigvec"]
    eigval = eig["eigval"]

    return eigvec, eigval


def write_eig(fname, eigvec, eigval):
    """
    Write eigendecomposition.
    """
    np.savez(fname, eigvec=eigvec, eigval=eigval)