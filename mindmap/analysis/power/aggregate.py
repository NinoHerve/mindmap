"""
Module Name: analysis/power/aggregate.py
Description: Aggregation of per-run pilot checkpoint files into a long-format
dataframe for variance-component fitting.

Author: Nino Herve
"""

import ast
import json
from pathlib import Path

import pandas as pd


def load_pilot_checkpoints(
    index_df: pd.DataFrame,
    metrics_dir: str | Path,
    index_cols: tuple = ("region", "repetition"),
) -> pd.DataFrame:
    """
    Stitches per-run JSON checkpoints (one `run-{run:03d}.json` per run, a
    list of per-method score dicts, e.g. as written from
    `mindmap.analysis.functional.inverse.score_source_reconstruction`'s
    output) into one long-format dataframe, tagging every row with that
    run's `index_cols` from `index_df`.

    A run whose checkpoint file doesn't exist yet (solve step still running,
    or crashed before writing it) is silently skipped -- callers can re-run
    the solve step and call this again once every checkpoint lands.

    Parameters
    ----------
    index_df: pd.DataFrame
        A pilot's run index (e.g. `simulation_index.tsv`), one row per run,
        with a `run` column and every name in `index_cols`. If `region` is
        requested but not already a plain column, it's parsed from
        `source_index_list` (a stringified single-element list, e.g.
        `"['ctx-rh-...']"`) -- the convention `generate_simulation_dataset`
        writes.

    metrics_dir: str | Path
        Directory containing `run-{run:03d}.json` checkpoint files.

    index_cols: tuple[str, ...]
        Columns from `index_df` to attach to every row of that run's
        checkpoint (e.g. `("region", "repetition")` for the base pilot,
        `("region", "repetition", "snr")` for a noise sweep, `("region",
        "repetition", "connectome_draw", "severity")` for a connectome-
        degradation sweep).

    Returns
    -------
    df: pd.DataFrame
        One row per (run, method), with the checkpoint's scalar outcome
        columns plus `index_cols`.
    """
    metrics_dir = Path(metrics_dir)
    index_df = index_df.copy()

    if "region" in index_cols and "region" not in index_df.columns:
        index_df["region"] = index_df["source_index_list"].apply(lambda s: ast.literal_eval(s)[0])

    rows = []
    for _, idx_row in index_df.iterrows():
        run = int(idx_row["run"])
        fjson = metrics_dir / f"run-{run:03d}.json"
        if not fjson.exists():
            continue
        with open(fjson) as f:
            run_rows = json.load(f)
        for r in run_rows:
            r = dict(r)
            for col in index_cols:
                r[col] = idx_row[col]
            rows.append(r)

    return pd.DataFrame(rows)
