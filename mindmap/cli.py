"""
Command lines to generate plots for given subjects.
"""

import argparse
import numpy as np
import mne
from mindmap.io import read_config, get_bidspath
from mindmap.viz import plot_brain, plot_bem, plot_alignment, plot_leadfield,\
    plot_source_estimate, plot_con_summary, plot_eigvec, plot_eigval, plot_ieeg

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


    elif args.target == "ave":
        fave = get_bidspath(
            root        = config["MM_DIR"],
            datatype    = "ave",
            task        = args.task,
            description = args.desc,
            subject     = args.subject,
        )

        eeg = mne.Evoked(fave, verbose="warning")

        eeg.plot()

    elif args.target == "iave":
        fiave = get_bidspath(
            root        = config["MM_DIR"],
            datatype    = "iave",
            task        = args.task,
            description = args.desc,
            subject     = args.subject,
        )

        ieeg = mne.Evoked(fiave, verbose="warning")

        ieeg.plot()

    elif args.target == "stc":
        ffwd = get_bidspath(
            root     = config["MM_DIR"],
            datatype = "fwd",
            subject  = args.subject,
        )

        fstc = get_bidspath(
            root        = config["MM_DIR"],
            datatype    = "stc",
            description = args.desc,
            subject     = args.subject,
        )

        fwd = mne.read_forward_solution(ffwd, verbose="warning")
        stc = mne.read_source_estimate(fstc)

        plot_source_estimate(fwd["src"], stc, ts_params={"absolute":True}).show()

    elif args.target == "con":
        fcon = get_bidspath(
            root        = config["MM_DIR"],
            datatype    = "con",
            subject     = args.subject,
            description = args.desc,
        )

        con = np.load(fcon)

        plot_con_summary(con, args.subject)

    elif args.target == "eigvec":
        feig = get_bidspath(
            root        = config["MM_DIR"],
            datatype    = "eig",
            subject     = args.subject,
            description = args.desc
        )
        fsrc = get_bidspath(
            root = config["MM_DIR"],
            datatype = "src",
            subject = args.subject,
        )

        src = mne.read_source_spaces(fsrc)
        eig = np.load(feig)
        vec = eig["eigvec"]
        val = eig["eigval"]

        plot_eigvec(src, vec, val).show()

    elif args.target == "eigval":
        feig = get_bidspath(
            root        = config["MM_DIR"],
            datatype    = "eig",
            subject     = args.subject,
            description = args.desc,
        )

        eig = np.load(feig)
        val = eig["eigval"]

        plot_eigval(val, args.subject)


    elif args.target == "ieeg":
        fieeg = get_bidspath(
            root     = config["MM_DIR"],
            datatype = "iave",
            subject  = args.subject,
            task     = args.task,
        )
        fsrc = get_bidspath(
            root     = config["MM_DIR"],
            datatype = "src",
            subject  = args.subject,
        )

        src = mne.read_source_spaces(fsrc)
        ieeg = mne.Evoked(fieeg, verbose="warning")

        plot_ieeg(src, ieeg, subject=args.subject).show()
