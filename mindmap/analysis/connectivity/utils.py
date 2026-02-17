"""
Module Name: analysis/connectivity/utils.py
Description: Useful functions for connectomes.

This module provides functions to handle connectomes such as 
building them from tractograms.

Author: Nino Herve
Created: 2025-10-24
Last Modified: 2025-10-24
License: MIT
"""

import numpy as np


def print_connectome_summary(connectome: np.ndarray) -> None:
    """
    Print various information about the connectome matrix.
    """
    # compute number of connections per node
    connections_per_node = (connectome > 0).sum(axis=1)

    n = len(connectome)
    e = np.sum(connections_per_node)/2

    print(f"Number of nodes: {n}")
    print(f"Total number of connections: {e} (sparsity = {2*e/(n*n-n):.3}%)")
    print(f"Minimum (non-zero) weight: {connectome[connectome>0].min()}")
    print(f"Maximum weight: {connectome.max()}")
    print(f"Number of isolated nodes: {np.sum(connections_per_node==0)} (ratio = {np.sum(connections_per_node==0)/n*100:.3})%")
