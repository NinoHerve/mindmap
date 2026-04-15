"""
Module Name: analysis.functional.common.utils.py
Description: Common utilities for other sub-modules in this module.

This module provides utilities.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np

def is_binary_mne(obj):
    """
    Checks if mne object is binary.
    """
    data = obj.data

    unique_values = np.unique(data)

    # Check if there are only 2 values and if they are (0, 1)
    return len(unique_values) <= 2 and np.all(np.isin(unique_values, [0, 1]))
