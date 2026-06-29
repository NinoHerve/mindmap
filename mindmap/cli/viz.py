"""
Command lines to generate plots for given subjects.
"""

import argparse
from mindmap.io.dataset import DatasetManager
from mindmap.core import get_spike_extent_mask, get_spurious_mask
from mindmap.analysis.functional.eeg import get_ch_positions, normalize_evoked
from mindmap.viz import (
    plot_brain, plot_source_estimate, plot_alignment, plot_leadfield,
    plot_eigvec, plot_ieeg, plot_bem, plot_con_summary, plot_eigval,
    plot_channel_grid, plot_segmentation,
)

def main():
    """
    mindmap-viz command line entry point.
    """

    #---------
    # Parsing
    # --------

    parser = argparse.ArgumentParser(prog="mindmap-viz")
    parser.add_argument(
        "target",
        choices = ["charm", "src", "bem", "align", "leadfield", "ave", "iave", "stc", "con", "eigvec", "eigval", "iave-pos", "vave"],
        help    = "What to visualize",
    )
    parser.add_argument(
        "--config",
        required = True,
        help     = "Path to configuration YAML file."
    )
    parser.add_argument(
        "--subject",
        required = True,
        help     = "Subject ID",
    )
    parser.add_argument(
        "--desc",
        required = False,
        default  = None,
        help     = "BIDS description identifier"
    )
    parser.add_argument(
        "--task",
        required = False,
        default  = None,
        help     = "BIDS task identifier."
    )

    args = parser.parse_args()

    # --------------
    # Configuration
    # --------------

    dm = DatasetManager(args.config)

    # -----
    # Plot
    # -----
    if args.target == "charm":
        ft1 = dm.get_path(args.subject, "T1w")
        fseg = dm.get_path(args.subject, "seg")
        flut = str(fseg).replace(".nii.gz", "_LUT.txt")
        plot_segmentation(ft1, fseg, flut)

    if args.target == "src":
        src = dm.load(args.subject, "src")
        plot_brain(src, mesh_params={"show_src":True}).show()

    elif args.target == "bem":
        plot_bem(dm.config["DIR_FS"], args.subject)

    elif args.target == "align":
        bem = dm.load(args.subject, "bem")
        ave = dm.load(args.subject, "ave", task=args.task)
        plot_alignment(bem, ave).show()

    elif args.target == "leadfield":
        fwd = dm.load(args.subject, "fwd")
        ave = dm.load(args.subject, "ave", task=args.task)
        plot_leadfield(ave, fwd).show()

    elif args.target == "ave":
        ave = dm.load(args.subject, "ave", task=args.task)
        ave.plot()

    elif args.target == "iave":
        iave = dm.load(args.subject, "i-ave", task=args.task)
        iave.plot()

    elif args.target == "iave-pos":
        iave = dm.load(args.subject, "i-ave", task = args.task)
        src  = dm.load(args.subject, "src")

        extent_mask = get_spike_extent_mask(iave.data.T, perc_max_amp=0.7)
        spurious_mask = get_spurious_mask(extent_mask, get_ch_positions(iave), spurious_dist=0.01)
        import numpy as np
        iave.data = np.abs(iave.data)
        plot_ieeg(src, iave, subject=args.subject, extent_mask=extent_mask, spurious_mask=spurious_mask).show()

    elif args.target == "vave":
        iave = dm.load(args.subject, "i-ave", task = args.task)
        vave = dm.load(args.subject, "v-ave", task = args.task, desc = args.desc)

        iave = normalize_evoked(iave, "global", "zscore")
        vave = normalize_evoked(vave, "global", "zscore")

        extent_mask = get_spike_extent_mask(iave.data.T, perc_max_amp=0.7)
        spurious_mask = get_spurious_mask(extent_mask, get_ch_positions(iave), spurious_dist=0.01)
        plot_channel_grid(iave, {args.desc:vave}, extent_mask=extent_mask, spurious_mask=spurious_mask)

    elif args.target == "stc":
        stc = dm.load(args.subject, "stc", task=args.task, desc=args.desc)
        fwd = dm.load(args.subject, "fwd")
        plot_source_estimate(fwd["src"], stc, title=f"sub-{args.subject}: {args.desc}", ts_params={"absolute":True}).show()

    elif args.target == "con":
        con = dm.load(args.subject, "con", desc=args.desc)
        plot_con_summary(con, args.subject)

    elif args.target == "eigvec":
        eig = dm.load(args.subject, "eig", desc = args.desc)
        src = dm.load(args.subject, "fwd", desc="sim")["src"]
        plot_eigvec(src, eig["vec"], eig["val"]).show()

    elif args.target == "eigval":
        eig = dm.load(args.subject, "eig", desc = args.desc)
        plot_eigval(eig["val"], args.subject)
