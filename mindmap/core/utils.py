"""
Module Name: core/utils.py
Description: This module provides functions to read/write simple datatypes.

Contains functions to read simple data.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from pathlib import Path
import json
import re

import numpy as np

from .io import write_config, read_config
from .metric import compute_distances

def create_config(root, fname="config.yml"):
    """
    Create configuration file from dataset folder.
    """
    root_path = Path(root).resolve()
    derivatives_path = root_path / "derivatives"

    # 1. Extract Subjects (remove 'sub-' prefix for the list)
    subjects = sorted([
        d.name.replace("sub-", "")
        for d in root_path.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])

    # 2. Extrac Datatypes (folders inside sub-xxx that don't start with 'ses-')
    datatypes = set()
    for sub in subjects:
        sub_dir = root_path/f"sub-{sub}"
        for d in sub_dir.iterdir():
            if d.is_dir() and not d.name.startswith("ses-"):
                datatypes.add(d.name)

            elif d.is_dir() and d.name.startswith("ses-"):
                for session_sub_dir in d.iterdir():
                    if session_sub_dir.is_dir():
                        datatypes.add(session_sub_dir.name)

    # 3. Handle Derivatives
    deriv_map = {}
    if derivatives_path.exists():
        for d in derivatives_path.iterdir():
            if not d.is_dir():
                continue

            name_lower = d.name.lower()

            # Mapping specific keas
            if "mindmap" in name_lower:
                key = "DIR_MM"
            elif "freesurfer" in name_lower:
                key = "DIR_FS"
            elif "charm" in name_lower:
                key = "DIR_CHARM"
            else:
                key = f"DIR_{d.name}"

            deriv_map[key] = str(d)

    # 4. Construct the dictionary structure
    config_data = {
        "NAME":root_path.name,
        "DIR_RAW": str(root_path),
        **deriv_map,
        "FS_HOME": None,
        "SUBS": subjects,
        "DATATYPES": sorted(list(datatypes))
    }

    # 5. Write to YAML
    write_config(fname, config_data)

    print(f"Configuration saved to {fname}")

def create_tmp_config(fname="config_template.yml"):
    """
    Create template configuration file to fill manually.
    """
    # Define empty structure
    placeholder = "#tofill"
    template = {
        "DIR_RAW": placeholder,
        "DIR_FS": placeholder,
        "DIR_MM": placeholder,
        "DIR_CHARM": placeholder,
        "FS_HOME": placeholder,
        "NAME": placeholder,
        "SUBS": [],
        "DATATYPES": [],
    }

    # write file
    write_config(fname, template)

    print(f"Empty template created at {fname}")

def add_to_config(config_file, key, value):
    """
    Add information to configuration file.
    """
    # 1. load existing configuration
    config = read_config(config_file)

    # 2. add or update the new key
    config[key] = value

    # 3. categorize keys for smart sorting
    name   = {"NAME":config["NAME"]}
    raw    = {"DIR_RAW":config["DIR_RAW"]}

    dirs   = {k: v for k, v in config.items()
            if k.startswith("DIR_") and k != "DIR_RAW"}

    stds   = {k: v for k, v in config.items() if "STD" in k}

    others = {k: v for k, v in config.items()
            if k not in dirs and k not in stds and k not in ["NAME", "DIR_RAW"]}

    # 4. rebuild ordered dict [DIRS] -> [STDS] -> [Others]
    sorted_config = {}
    sorted_config.update(name)
    sorted_config.update(raw)
    sorted_config.update(dict(sorted(dirs.items())))
    sorted_config.update(dict(sorted(stds.items())))
    sorted_config.update(others)

    write_config(config_file, sorted_config)

    print(f"Successfully added '{key}' to {config_file}.")



def get_spike_extent_mask(
    data : np.ndarray,
    perc_max_amp : float = 0.7
) -> np.ndarray:
    """
    Identifies the boolean mask of channels belonging to the 
    spatial extent of a peak event.

    Parameters
    ----------
    data: np.ndarray of shape (n_timepoints, n_variables)
        Input data.

    perc_max_amp: float
        Percentage of maximum amplitude for threshold. Value between 0 and 1.

    Returns
    -------
        np.ndarray of shape (n_channels)
            Boolean mask (True where channel is in extent)
    """
    # ensure inputs are at least 2D: (n_timepoints, n_variables)
    # if input is (M,), it becomes (1,M)
    data = np.atleast_2d(data)

    # 1. Find the gloabl peak in time
    # We find the peak row (time_idx) across the whole matrix
    peak_time_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)[0]

    # 2. Extract the spatial distribution at that specific peak time
    # Shape will be (n_variables,)
    spatial_vector = np.abs(data[peak_time_idx, :])

    # 3. Calculate the threshold for this specific spatial vector
    threshold = perc_max_amp * np.max(spatial_vector)

    # 4. Create boolean mask
    return spatial_vector >= threshold


def get_spurious_mask(
    extent_mask : np.ndarray,
    coord : np.ndarray,
    spurious_dist : float = 0.01,
) -> np.ndarray:
    """
    Identifies the boolean mask of channels belonging to the spurious area.

    Parameters
    ----------
    extent_mask: np.ndarray
        Boolean mask of channels in the reference set H.
    coord: np.ndarray of shape (n_points, 3)
        Coordinates of channels (if in meters, spurious_dist should be 0.01).
    spurious_dist: float
        Distance beyond which data is spurious (e.g., 10 for mm).
    
    Returns
    -------
    is_spurious: np.ndarray
        Boolean mask where True indicates a 'distant' channel.
    """
    # Distance between all channels and the channels in set H
    dist_matrix = compute_distances(coord, coord[extent_mask])

    # minimum distance to any channel in the extent
    dist_to_h   = np.min(dist_matrix, axis=1)

    return dist_to_h > spurious_dist


def create_dataset_description(
    bids_root,
    name = "Unknown dataset",
    bids_version = "1.8.0",
    dataset_type = "raw",
    authors = None,
    generated_by = None,
    how_to_acknowledge = None,
    overwrite = False,
):
    """
    Create a minimal dataset_descritpion.json in a BIDS root directory.
    
    Parameters
    ----------
    bids_root: str or Path
        Path to the root of the BIDS dataset.

    name: str
        Dataset name.

    bids_version: str
        BIDS version string.
    
    dataset_type: str
        Usually "raw" or "derivative".

    authors: list of str or None
        Optional list of authors.

    generated_by: list of dict or None
        For derivatives. List of dictionaries containing 'Name', 'Version', etc.

    how_to_acknowledge: str or None
        Optional text on how to cite the dataset.

    overwrite: bool
        If False, will not overwrite an existing file.
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    desc_path = bids_root / "dataset_description.json"

    if desc_path.exists() and not overwrite:
        print(f"{desc_path} already exists. Use overwrite=True to replace it.")
        return desc_path

    description = {
        "Name": name,
        "BIDSVersion": bids_version,
        "DatasetType": dataset_type,
    }

    if authors:
        description["Authors"] = authors if isinstance(authors, list) else [authors]
    else:
        description["Authors"] = ["Unknown"]

    if dataset_type == "derivative":
        if generated_by is None:
            description["GeneratedBy"] = [{
                "Name": "Custom Pipeline",
                "Version": "1.0.0",
            }]
        else:
            description["GeneratedBy"] = generated_by

    if how_to_acknowledge:
        description["HowToAcknowledge"] = generated_by

    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(description, f, indent=4, ensure_ascii=False)

    print(f"Created {desc_path}")
    return desc_path

