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
import importlib
from typing import Any
import os
import zipfile

import yaml
import numpy as np
import pandas as pd
from numpy.lib import format as npy_format


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


def read_npz_fast(path: str | Path) -> dict[str, np.ndarray]:
    """
    Reads every array in a `.npz` archive (as written by `np.savez`) into a
    dict, in O(n) time for n arrays -- unlike `np.load(path)`'s `NpzFile`,
    whose `__getitem__` does `if key in self._files` against a plain Python
    `list`, an O(n) scan repeated on *every* lookup. Looping `data[key]` for
    n keys is therefore O(n^2): confirmed via profiling on a real 88,000-
    entry `MetricSuite` save (`mindmap.metric.MetricSuite.load`, the caller
    this exists for) -- 32.7s of a 40s total load was that scan alone, and
    replacing it with this function dropped the same load to 2.25s.

    Reads each member directly via `zipfile.ZipFile` (whose member lookup is
    an O(1) dict, `NameToInfo`) and decodes it with `numpy.lib.format`'s own
    `.npy` reader -- the same low-level codec `np.load`/`np.savez` use
    internally, so this reproduces `np.load`'s exact per-array behavior
    (including `allow_pickle=False`, matching `np.load`'s own default) minus
    the quadratic blowup. Assumes every member is `.npy`-formatted, true for
    anything written by `np.savez` (the only writer used in this codebase) --
    unlike `NpzFile`, this does not fall back to raw bytes for non-`.npy`
    members.

    Parameters
    ----------
    path: str | Path

    Returns
    -------
    dict[str, np.ndarray], keyed by array name (the ".npy" suffix stripped,
    matching `NpzFile.files`' naming convention).
    """
    arrays = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            with zf.open(name) as f:
                arrays[key] = npy_format.read_array(f, allow_pickle=False)
    return arrays


def write_csv_row(filename, data, index_cols, write_new=False):
    """
    Saves or updates rows in a CSV file.

    Parameters
    ----------
    filename: str | Path
        Path to the CSV file.
    data: dict | pd.Series | pd.DataFrame
        Data to write.
    write_new: bool
        If True, creates the file if it doesn't exist.
    """
    # Standardize input to a DataFrame
    if isinstance(data, dict):
        new_df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        new_df = data.to_frame().T
    elif isinstance(data, pd.DataFrame):
        new_df = data
    else:
        raise TypeError("Data must be a dict, Series, or DataFrame")

    # Ensure index_cols is a list
    if isinstance(index_cols, str):
        index_cols = [index_cols]

    # Ensure filename is a Path
    filename = Path(filename)

    # Check file existence
    if not filename.exists():
        if write_new:
            new_df.to_csv(filename, index=False)
            print(f"Created new file: {filename}")
            return
        else:
            raise FileNotFoundError(f"The file '{filename}' does not exist and write_new is False.")

    # Load existing data
    existing_df = pd.read_csv(filename)
    new_df = new_df.reindex(columns=existing_df.columns, fill_value=None)

    # Merge
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df = combined_df.drop_duplicates(subset=index_cols, keep="last")

    # save
    final_df.to_csv(filename, index=False)


def get_config(config_input):
    """
    Handles types of input for the config argument.
    """
    # Handle "Keywords" for internal library configs
    if config_input == "charm":
        # accessing the file inside the package
        config_input = importlib.resources.files("mindmap.resources").joinpath("bem_charm_config.yml")

    # Handle File Paths (String or Path object)
    if isinstance(config_input, (str, Path)):
        return read_config(config_input)

    # Handle Dictionary
    if isinstance(config_input, dict):
        return config_input

    raise ValueError("Invalid config input. Provide a path, 'charm', or a dict")

