"""
Command lines to generate plots of mindmap generated objects for given subjects.
"""

import argparse
import numpy as np

from mindmap.io.dataset import DatasetManager
from mindmap.core import get_spike_extent_mask, get_spurious_mask
from mindmap.analysis.functional.eeg import get_ch_positions, normalize_evoked
from mindmap.cli.resolve import resolve_bundle, resolve_path, parse_entity_flags
from mindmap.cli.viz_specs import TARGET_SPECS
from mindmap.viz import (
    plot_brain, plot_brain_activity, plot_alignment, plot_leadfield,
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
        choices = ["charm", "parc", "src", "bem", "align", "leadfield", "ave", "iave", "stc", "con", "eigvec", "eigval", "iave-pos", "vave"],
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
        "--entity", "-e",
        action   = "append",
        default  = [],
        metavar  = "SUFFIX:KEY=VALUE[,KEY=VALUE...]",
        help     = "Request one object by its (possibly partial) BIDS entities, e.g. "
                    "-e stc:desc=cset,res=ico5. Repeat -e for the same suffix to request "
                    "several objects for side-by-side comparison, e.g. -e stc:desc=cset "
                    "-e stc:desc=sloreta (only for targets/suffixes that support it). "
                    "Anything left ambiguous is resolved interactively, one entity at a "
                    "time; answering blank means that entity must be absent from the file."
    )

    args = parser.parse_args()
    cli_entities = parse_entity_flags(args.entity)

    # --------------
    # Configuration
    # --------------

    dm = DatasetManager(args.config)

    # ----------------------------
    # Resolve the objects needed
    # ----------------------------

    bundle = {}
    if args.target in TARGET_SPECS:
        bundle = resolve_bundle(dm, args.subject, TARGET_SPECS[args.target], cli_entities)

    # -----
    # Plot
    # -----
    if args.target == "charm":
        ft1 = resolve_path(dm, args.subject, "t1-charm", cli_entities.get("t1-charm", []))
        fseg = resolve_path(dm, args.subject, "seg", cli_entities.get("seg", []))
        flut = str(fseg).replace(".nii.gz", "_LUT.txt")
        plot_segmentation(ft1, fseg, flut)

    elif args.target == "parc":
        ft1 = resolve_path(dm, args.subject, "t1.mgz", cli_entities.get("t1.mgz", []))
        fseg = resolve_path(dm, args.subject, "dseg", cli_entities.get("dseg", []))
        flut = str(fseg).replace(".nii.gz", ".txt")  # dseg's LUT sidecar convention (see _load_parc)
        plot_segmentation(ft1, fseg, flut)

    elif args.target == "src":
        plot_brain(bundle["src"], mesh_params={"show_src":True}).show()

    elif args.target == "bem":
        plot_bem(dm.config["DIR_FS"], args.subject)

    elif args.target == "align":
        plot_alignment(bundle["bem"], bundle["ave"]).show()

    elif args.target == "leadfield":
        plot_leadfield(bundle["ave"], bundle["fwd"]).show()

    elif args.target == "ave":
        bundle["ave"].plot()

    elif args.target == "iave":
        bundle["i-ave"].plot()

    elif args.target == "iave-pos":
        iave = bundle["i-ave"]
        src  = bundle["src"]

        extent_mask = get_spike_extent_mask(iave.data.T, perc_max_amp=0.7)
        spurious_mask = get_spurious_mask(extent_mask, get_ch_positions(iave), spurious_dist=0.01)
        iave.data = np.abs(iave.data)
        plot_ieeg(src, iave, subject=args.subject, extent_mask=extent_mask, spurious_mask=spurious_mask).show()

    elif args.target == "vave":
        iave  = bundle["i-ave"]
        vaves = bundle["v-ave"]  # {desc: v-ave object}, one or many

        iave = normalize_evoked(iave, "global", "zscore")
        vaves = {desc: normalize_evoked(vave, "global", "zscore") for desc, vave in vaves.items()}

        extent_mask = get_spike_extent_mask(iave.data.T, perc_max_amp=0.7)
        spurious_mask = get_spurious_mask(extent_mask, get_ch_positions(iave), spurious_dist=0.01)
        plot_channel_grid(iave, vaves, extent_mask=extent_mask, spurious_mask=spurious_mask)

    elif args.target == "stc":
        stcs = bundle["stc"]  # {label: stc}, one or many
        fwd = bundle["fwd"]
        plot_brain_activity(
            fwd["src"], list(stcs.values()),
            title=[f"sub-{args.subject}: {label}" for label in stcs],
            func_params={"absolute": True, "normalize": "max"},
        ).show()

    elif args.target == "con":
        plot_con_summary(bundle["con"], args.subject)

    elif args.target == "eigvec":
        eig = bundle["eig"]
        src = bundle["fwd"]["src"]
        plot_eigvec(src, eig["vec"], eig["val"]).show()

    elif args.target == "eigval":
        plot_eigval(bundle["eig"]["val"], args.subject)
