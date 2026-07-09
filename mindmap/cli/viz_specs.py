"""
Module Name: cli/viz_specs.py
Description: Declares, for each mindmap-viz target, which objects it needs.
Entities are NOT declared here — resolve.py discovers them from the BIDS
layout at run time and only asks the user when a subject genuinely has
several matching files.
"""

from mindmap.cli.resolve import ObjectSpec

TARGET_SPECS = {
    "src": [
        ObjectSpec(suffix="src"),
    ],
    "align": [
        ObjectSpec(suffix="bem"),
        ObjectSpec(suffix="ave"),
    ],
    "leadfield": [
        ObjectSpec(suffix="fwd"),
        ObjectSpec(suffix="ave"),
    ],
    "ave": [
        ObjectSpec(suffix="ave"),
    ],
    "iave": [
        ObjectSpec(suffix="i-ave"),
    ],
    "iave-pos": [
        ObjectSpec(suffix="i-ave"),
        ObjectSpec(suffix="src"),
    ],
    "vave": [
        ObjectSpec(suffix="i-ave"),
        ObjectSpec(suffix="v-ave", multi=True),
    ],
    "stc": [
        ObjectSpec(suffix="stc", multi=True),
        ObjectSpec(suffix="fwd"),
    ],
    "con": [
        ObjectSpec(suffix="con"),
    ],
    "eigvec": [
        ObjectSpec(suffix="eig"),
        ObjectSpec(suffix="fwd"),
    ],
    "eigval": [
        ObjectSpec(suffix="eig"),
    ],
}
