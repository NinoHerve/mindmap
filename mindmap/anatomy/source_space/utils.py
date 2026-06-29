"""
Module Name: source_space/utils.py
Description: Helpers for source spaces.

This module provides helpers for source spaces.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np
import pandas as pd

import mne

from mindmap.anatomy.parcellation.base import Parcellation

def correct_labels(
    labels: list[str] | list[int],
    sources: mne.SourceSpaces,
    parcellation: Parcellation, 
):
    """
    Correct labels that misplace left and right hemisphere dipoles because the parcellation is not perfect.
    """
    if len(labels) == 0:
        return []
    
    # Detect the input type based on the first element
    is_int_type = isinstance(labels[0], (int, np.integer))

    # Build fast lookup tables from the parcellation
    clut = parcellation.get_clut()
    id_to_name = pd.Series(clut["Label name"].values, index=clut["#No."]).to_dict()
    name_to_id = {v:k for k, v in id_to_name.items()}
    
    # Standardize input to names for the hemisphere manipulation step
    if is_int_type:
        label_names = [id_to_name[label_id] for label_id in labels]
    else:
        label_names = list(labels)
    
    # Determine left/right hemisphere separation from MNE source space
    split = len(sources[0]["vertno"])

    # Switch 'lh' labels in right hemisphere to 'rh' homologue and vice virsa
    label_names[:split] = [l.replace("-rh-", "-lh-") for l in label_names[:split]]
    label_names[split:] = [l.replace("-lh-", "-rh-") for l in label_names[split:]]

    # Return the data matching the original input type
    if is_int_type:
        return np.array([name_to_id[name] for name in label_names])
    else:
        return np.array(label_names)