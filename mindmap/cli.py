"""
Command lines to generate plots for given subjects.
"""

import argparse
import mne
from mindmap.io import read_config, get_bidspath
from mindmap.viz import plot_brain, plot_bem

def check_file(file):
    """Check file existense"""
    if not file.exists():
        raise FileNotFoundError(f"File does not exist: {file}")

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
        choices = ["src", "bem"],
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

    args = parser.parse_args()

    # --------------
    # Configuration
    # --------------

    config = read_config(args.config)

    # -----
    # Plot
    # -----

    if args.target == "src":
        file = get_bidspath(
            root     = config["MM_DIR"],
            datatype = args.target,
            subject  = args.subject,
        )
        check_file(file)

        src = mne.read_source_spaces(file)
        plot_brain(src, mesh_params={"show_src":True}).show()
    
    elif args.target == "bem":
        plot_bem(config["FS_DIR"], args.subject)
