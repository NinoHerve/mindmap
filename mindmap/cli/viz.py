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
    plot_channel_grid,
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
        choices = ["src", "bem", "align", "leadfield", "ave", "iave", "stc", "con", "eigvec", "eigval", "iave-pos", "vave"],
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
    sr = dm.get_subject(args.subject)

    # -----
    # Plot
    # -----

    if args.target == "src":
        src = sr.load("src")
        plot_brain(src, mesh_params={"show_src":True}).show()

    elif args.target == "bem":
        plot_bem(dm.config["DIR_FS"], args.subject)

    elif args.target == "align":
        bem = sr.load("bem")
        ave = sr.load("ave", task=args.task)
        plot_alignment(bem, ave).show()

    elif args.target == "leadfield":
        fwd = sr.load("fwd")
        ave = sr.load("ave", task=args.task)
        plot_leadfield(ave, fwd).show()

    elif args.target == "ave":
        ave = sr.load("ave", task=args.task)
        ave.plot()

    elif args.target == "iave":
        iave = sr.load("i-ave", task=args.task)
        iave.plot()

    elif args.target == "iave-pos":
        iave = sr.load("i-ave", task = args.task)
        src  = sr.load("src")

        extent_mask = get_spike_extent_mask(iave.data.T, perc_max_amp=0.7)
        spurious_mask = get_spurious_mask(extent_mask, get_ch_positions(iave), spurious_dist=0.01)
        plot_ieeg(src, iave, subject=args.subject, extent_mask=extent_mask, spurious_mask=spurious_mask).show()

    elif args.target == "vave":
        iave = sr.load("i-ave", task = args.task)
        vave = sr.load("v-ave", task = args.task, desc = args.desc)

        iave = normalize_evoked(iave, "global", "zscore")
        vave = normalize_evoked(vave, "global", "zscore")

        extent_mask = get_spike_extent_mask(iave.data.T, perc_max_amp=0.7)
        spurious_mask = get_spurious_mask(extent_mask, get_ch_positions(iave), spurious_dist=0.01)
        plot_channel_grid(iave, {args.desc:vave}, extent_mask=extent_mask, spurious_mask=spurious_mask)

    elif args.target == "stc":
        stc = sr.load("stc", task=args.task, desc=args.desc)
        fwd = sr.load("fwd")
        plot_source_estimate(fwd["src"], stc, title=f"sub-{args.subject}: {args.desc}", ts_params={"absolute":True}).show()

    elif args.target == "con":
        con = sr.load("con", desc=args.desc)
        plot_con_summary(con, args.subject)

    elif args.target == "eigvec":
        eig = sr.load("eig", desc = args.desc)
        src = sr.load("src")
        plot_eigvec(src, eig["vec"], eig["val"]).show()

    elif args.target == "eigval":
        eig = sr.load("eig", desc = args.desc)
        plot_eigval(eig["val"], args.subject)
