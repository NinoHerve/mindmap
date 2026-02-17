"""
Module Name: utils.py
Description: Miscellaneous functions.

This module contains miscellaneous functions that do not belong to any other submodules.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import os
from pathlib import Path


def _retrieve_env(*args) -> dict:
    try:
        if len(args)==1:
            variables = os.environ[args[0]]

            if isinstance(variables, str) and ("/" in variables):
                variables = Path(vars)

        else:
            variables = {}

            for key in args:
                v = os.environ[key]
                if isinstance(v, str):
                    if v in ["True", "False"]:
                        variables[key] = v == "True"
                    elif "/" in v:
                        variables[key] = Path(v)
                    else:
                        variables[key] = v

    except KeyError as e:
        raise KeyError(f"Environment variable {e} was not found. Read configuration file or check configuration file.") from e

    return variables
