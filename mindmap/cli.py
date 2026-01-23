"""
Command lines to generate plots for given subjects.
"""

import argparse
import mne
from mindmap.io import read_config, get_bidspath
from mindmap.viz import plot_brain, plot_bem, plot_alignment, plot_leadfield

def check_file(file):
    """Check file existense"""
    if not file.exists():
        raise FileNotFoundError(f"File does not exist: {file}")
    
def get_ave_info(config):
    """Retrieve first task, session and run from the metadata"""

    task = config["TASKS"][0] if config.get("TASKS") else None
    ses  = config["SES"][0] if config.get("SES") else None
    run  = config["RUNS"][0] if config.get("RUNS") else None

    return task, ses, run

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
        choices = ["src", "bem", "align", "leadfield"],
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

    elif args.target == "align":
        task, ses, run = get_ave_info(config)

        fbem = get_bidspath(
            root     = config["MM_DIR"],
            datatype = "bem",
            subject  = args.subject,
        )
        fave = get_bidspath(
            root     = config["MM_DIR"],
            datatype = "ave",
            task     = task,
            session  = ses,
            run      = run,
            subject  = args.subject,
        )

        bem = mne.read_bem_solution(fbem)
        ave = mne.Evoked(fave, verbose="warning")

        plot_alignment(bem, ave).show()

    elif args.target == "leadfield":
        task, ses, run = get_ave_info(config)

        ffwd = get_bidspath(
            root     = config["MM_DIR"],
            datatype = "fwd",
            subject  = args.subject,
        )
        fave = get_bidspath(
            root     = config["MM_DIR"],
            datatype = "ave",
            task     = task,
            session  = ses,
            run      = run,
            subject  = args.subject,
        )

        eeg = mne.Evoked(fave, verbose="warning")
        fwd = mne.read_forward_solution(ffwd)

        plot_leadfield(eeg, fwd).show()
