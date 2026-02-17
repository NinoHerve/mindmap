"""
Command lines to generate plots for given subjects.
"""

import argparse
from mindmap.io.dataset import DatasetManager
from mindmap.viz import plot_brain, plot_bem, plot_alignment, plot_leadfield,\
    plot_source_estimate, plot_con_summary, plot_eigvec, plot_eigval, plot_ieeg

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
        choices = ["src", "bem", "align", "leadfield", "ave", "iave", "stc", "con", "eigvec", "eigval", "ieeg"],
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

    elif args.target == "stc":
        stc = sr.load("stc", task=args.task, desc=args.desc)
        fwd = sr.load("fwd")
        plot_source_estimate(fwd["src"], stc, ts_params={"absolute":True}).show()

    elif args.target == "con":
        con = sr.load("con", desc=args.desc)
        plot_con_summary(con, args.subject)

    elif args.target == "eigvec":
        eig = sr.load("eig", desc = args.desc)
        src = sr.load("src")
        plot_eigvec(src, eig["eigvec"], eig["eigval"]).show()

    elif args.target == "eigval":
        eig = sr.load("eig", desc = args.desc)
        plot_eigval(eig["eigval"], args.subject)

    elif args.target == "ieeg":
        iave = sr.load("i-ave", task = args.task)
        src  = sr.load("src")
        plot_ieeg(src, iave, subject=args.subject).show()
