"""
Module Name: metric/base.py
Description: Contains MetricSuite class.

The MetricSuite class allows managing all metrics computed within a same study.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import mne

from mindmap.core.io import read_npz_fast

from mindmap.viz.core import (
    metric_bar, metric_box, metric_heatmap, metric_violin, metric_relational
)
from mindmap.viz.dispatch import plot_brain_activity
from mindmap.viz.stats import metric_box_significance, metric_relational_clusters

from mindmap.analysis.stats import (
    compare_methods as _compare_methods_df,
    cluster_permutation_mne as _cluster_permutation_mne,
)


def _atomic_replace(path: Path, write_func):
    """
    Writes `path` by first calling `write_func(tmp_path)` to fully write the
    content to a temporary sibling file, then atomically replacing `path`
    with it via `os.replace` (atomic on both POSIX and Windows).

    Guarantees `path` is always either its previous complete content or the
    new complete content -- never a truncated/partial file -- even if the
    process is killed mid-write. Plain `open(path, "w")` / `np.savez(path)`
    truncate the destination immediately, so a crash mid-write would
    otherwise destroy the previous good save, not just fail to add the new
    one.

    `tmp_path` keeps `path`'s full original filename as its tail (only a
    unique token is prefixed), since some writers (numpy, MNE) validate or
    auto-append a specific suffix -- e.g. MNE's source-space writer checks
    for a "-src.fif" ending, which a token spliced in before the extension
    would break, but a prefix never touches.
    """
    tmp_path = path.with_name(f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}-{path.name}")
    try:
        write_func(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _is_arraylike(val):
    """Whether val should be treated as a per-point coordinate rather than a scalar tag."""
    return isinstance(val, (list, tuple, np.ndarray, pd.Series))


def _json_safe(val):
    """Converts numpy/pandas scalars into plain JSON-serializable Python types."""
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return val


def _entry_to_frame(entry, group_name):
    """
    Materializes a single registered entry into a long-format DataFrame slice.

    entry["value"] is an array of shape (n_points,) for a single curve, or
    (n_reps, n_points) for a stack of repeated observations (e.g. one curve
    per simulation run) sharing the same point-wise coordinates.
    """
    value = np.atleast_1d(np.asarray(entry["value"]))
    label, metric, coords = entry["label"], entry["metric"], entry["coords"]

    base = {group_name: label, "Metric": metric}

    if value.ndim == 1:
        data = {**base, "Value": value}
        for name, coord in coords.items():
            data[name] = coord
        return pd.DataFrame(data)

    if value.ndim != 2:
        raise ValueError(
            f"Registered values must be 1D or 2D (got shape {value.shape} for "
            f"label='{label}', metric='{metric}')."
        )

    # (n_reps, n_points): tile point-wise coordinates across reps, broadcast scalar
    # coordinates as-is, and tag each row with its repetition index so that repeated
    # curves (e.g. one per simulation run) stay distinguishable for CI estimation.
    n_reps, n_points = value.shape
    data = {**base, "Value": value.ravel(), "rep": np.repeat(np.arange(n_reps), n_points)}

    for name, coord in coords.items():
        coord_arr = np.asarray(coord) if _is_arraylike(coord) else None
        if coord_arr is not None and coord_arr.shape[0] == n_points:
            data[name] = np.tile(coord_arr, n_reps)
        else:
            data[name] = coord

    return pd.DataFrame(data)


class MetricSuite:
    """Manages metrics from a same study."""
    def __init__(self, name="Evaluation", group_name="Label", subject=None, metadata=None, context=None):
        """
        Docstring for __init__

        Parameters
        ----------
        name: str
            Study name

        subject: str, optional
            Identifier for the subject/patient this suite's entries belong
            to (e.g. "sub-01"). Automatically tagged as a "Subject" column
            on every entry registered via `register()`, so per-subject
            suites can later be pooled with `MetricSuite.concat` for
            group-level analysis. Can be overridden per-entry via
            `register(..., subject=...)`.

        metadata: dict, optional
            Free-form provenance to carry alongside the results (e.g.
            solver parameters, dataset path, a git commit, a one-line
            description) -- saved and reloaded as-is, so a folder written
            by `save()` documents its own context rather than being just
            numbers. Must be JSON-serializable.

        context: dict
            Context for 3D plotting.
            - 'src':mne.SourceSpaces
            - 'ave':mne.Evoked
        """
        self.name = name
        self._entries = []
        self.group_name = group_name
        self.subject = subject

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        self.metadata = metadata or {}

        self.context = context or {}
        self._validate_context()

    def _validate_context(self):
        """Validates that the context dictionary only contains allowed types."""
        if not isinstance(self.context, dict):
            raise TypeError("Context must be a dictionary.")
        
        allowed_keys = {"src", "ave"}
        invalid_keys = set(self.context.keys()) - allowed_keys
        if invalid_keys:
            raise KeyError(
                f"Invalid keys(s) found in context: {invalid_keys}."
                f"Only 'src' and 'ave' are allowed."
            )
        
        # Check 'ave' type consistency
        if "ave" in self.context and self.context["ave"] is not None:
            if not isinstance(self.context["ave"], mne.Evoked):
                raise TypeError(
                    f"Expected context['ave'] to be an instance of mne.Evoked, "
                    f"got {type(self.context['ave']).__name__} instead."
                )
            
        # Check 'src' type consistency
        if "src" in self.context and self.context["src"] is not None:
            if not isinstance(self.context["src"], mne.SourceSpaces):
                raise TypeError(
                    f"Expected context['src'] to be an instance of mne.SourceSpaces, "
                    f"got {type(self.context['src']).__name__} instead."
                )

    def register(self, label, metric, values, subject=None, **coords):
        """
        Adds a metric's results to the study.

        'values' can be a scalar, a 1D array of shape (n_points,) for a single
        curve/observation, or a 2D array of shape (n_reps, n_points) for a
        stack of repeated observations (e.g. one curve per simulation run)
        sharing the same point-wise coordinates. Coordinates whose length
        matches the last axis of 'values' are treated as point-wise (e.g.
        'times', 'recall'); anything else (scalar, or length-mismatched) is
        kept as a constant tag broadcast to every row.

        Data is kept as arrays and only expanded into a long-format table
        lazily, when `.dataframe`/`.plot()`/etc. are accessed - this keeps
        registering large stacks (e.g. thousands of per-run curves) cheap.

        Parameters
        ----------
        subject: str, optional
            Overrides this suite's `self.subject` for this entry only.
            Stored as a "Subject" tag, the same constant-tag mechanism as
            any other coordinate. Omit to use the suite's default (set via
            `MetricSuite(subject=...)`), which is None (no "Subject"
            column at all) unless configured.
        """
        effective_subject = subject if subject is not None else self.subject
        if effective_subject is not None:
            coords = {**coords, "Subject": effective_subject}

        value = np.asarray(values) if _is_arraylike(values) else np.asarray([values])

        n_points = value.shape[-1]
        for name, val in coords.items():
            if _is_arraylike(val) and len(val) != n_points:
                raise ValueError(
                    "Coordinate lengths must match the last axis of 'values'."
                    f"Instead, coordinate '{name}' has length {len(val)} for {n_points} points."
                )

        stored_coords = {
            name: (np.asarray(val) if _is_arraylike(val) else val)
            for name, val in coords.items()
        }

        self._entries.append({
            "label": label,
            "metric": metric,
            "value": value,
            "coords": stored_coords,
        })

        return self

    def add_from_function(self, method_name, metric_name, func, *args, **kwargs):
        """Register a new metric given the function to compute it."""
        result = func(*args, **kwargs)
        return self.register(method_name, metric_name, result)

    def reduce(self, func=np.mean):
        """
        Collapses every stacked (n_reps, n_points) entry (e.g. one curve per
        simulation run) along the reps axis via `func`, leaving single curves
        untouched. Returns a new, lightweight MetricSuite.

        Materializing the full table for a study with thousands of raw per-run
        curves and letting seaborn aggregate it (via `.plot()`) is correct but
        slow (tens of seconds). Reducing first is a vectorized O(n) numpy op,
        so `suite.reduce().plot(...)` is the fast path for a mean/summary
        curve; keep using the un-reduced suite's `.plot()` when you actually
        want per-run resolution (e.g. a bootstrapped CI band).

        Parameters
        ----------
        func: callable(array, axis=0) -> array
            Reduction applied along the reps axis, e.g. np.mean, np.median.
        """
        reduced = MetricSuite(
            name=self.name, group_name=self.group_name, subject=self.subject,
            metadata=self.metadata, context=self.context,
        )
        for e in self._entries:
            value = func(e["value"], axis=0) if e["value"].ndim == 2 else e["value"]
            reduced.register(e["label"], e["metric"], value, **e["coords"])
        return reduced

    @property
    def dataframe(self):
        """Dataframe property. Materializes the long-format table on demand."""
        if not self._entries:
            return pd.DataFrame()
        frames = [_entry_to_frame(e, self.group_name) for e in self._entries]
        return pd.concat(frames, ignore_index=True)

    def get_filtered_df(self, metrics=None, labels=None, subjects=None):
        """Filter data"""
        df = self.dataframe
        if df.empty:
            return df

        if metrics:
            metrics = [metrics] if isinstance(metrics, str) else metrics
            df = df[df["Metric"].isin(metrics)]
        if labels:
            labels = [labels] if isinstance(labels, str) else labels
            df = df[df[self.group_name].isin(labels)]
        if subjects:
            if "Subject" not in df.columns:
                raise ValueError("No subjects have been registered in this MetricSuite.")
            subjects = [subjects] if isinstance(subjects, str) else subjects
            df = df[df["Subject"].isin(subjects)]

        return df

    def summary(self):
        """Quick statistical overview."""
        return self.dataframe.groupby(["Metric", self.group_name])["Value"].agg(["mean", "std", "count"])

    def save(self, folder: str | Path, overwrite: bool = False):
        """
        Saves the MetricSuite to disk as a compact array store (data.npz) plus
        a small metadata file (meta.json). Keeping arrays as arrays (rather
        than exploding everything into a long-format CSV) is what makes this
        cheap even for studies with thousands of registered per-run curves.

        Parameters
        ----------
        folder: str | Path

        overwrite: bool
            If False (default), refuses to write into a folder that already
            holds a previous save (i.e. has a meta.json), so a typo'd or
            reused path can't silently destroy an earlier study's results.
        """
        folder = Path(folder)

        if (folder/"meta.json").exists() and not overwrite:
            raise FileExistsError(
                f"'{folder}' already contains a saved MetricSuite. Use overwrite=True to replace it."
            )

        folder.mkdir(parents=True, exist_ok=True)

        # 1. Save raw arrays + per-entry metadata
        arrays = {}
        entries_meta = []
        for i, e in enumerate(self._entries):
            arrays[f"entry{i}__value"] = e["value"]

            scalar_coords = {}
            array_coord_names = []
            for c_name, c_val in e["coords"].items():
                if isinstance(c_val, np.ndarray):
                    arrays[f"entry{i}__coord__{c_name}"] = c_val
                    array_coord_names.append(c_name)
                else:
                    scalar_coords[c_name] = _json_safe(c_val)

            entries_meta.append({
                "label": e["label"],
                "metric": e["metric"],
                "scalar_coords": scalar_coords,
                "array_coord_names": array_coord_names,
            })

        _atomic_replace(folder/"data.npz", lambda p: np.savez(p, **arrays))

        # 2. Save MNE Context Objects if they exist -- and remove any stale
        # one left over from a previous save into this same folder that no
        # longer applies (e.g. this suite's context dropped 'ave'/'src').
        ave_path = folder/"context-ave.fif"
        if self.context.get("ave") is not None:
            _atomic_replace(ave_path, lambda p: self.context["ave"].save(p, overwrite=True, verbose="warning"))
        elif ave_path.exists():
            ave_path.unlink()

        src_path = folder/"context-src.fif"
        if self.context.get("src") is not None:
            _atomic_replace(
                src_path,
                lambda p: mne.write_source_spaces(p, self.context["src"], overwrite=True, verbose="warning"),
            )
        elif src_path.exists():
            src_path.unlink()

        # 3. Write meta.json last, atomically, and only once the data/context
        # it describes is already safely on disk -- so a crash can never
        # leave a meta.json that references entries data.npz doesn't have
        # yet (the reverse -- data.npz holding entries meta.json hasn't
        # caught up to describing yet -- is harmless: load() only reads what
        # meta.json points to).
        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": self.name,
                        "group_name": self.group_name,
                        "subject": self.subject,
                        "metadata": self.metadata,
                        "saved_at": datetime.now().isoformat(),
                        "entries": entries_meta,
                    },
                    f, indent=2,
                )

        _atomic_replace(folder/"meta.json", _write_meta)

        print(f"MetricSuite and context successfully saved to folder: {folder}")

    @staticmethod
    def peek(folder: str | Path) -> dict:
        """
        Lightweight inspection of a folder written by `save()`, without
        loading the array data or MNE context files.

        Reads only meta.json -- cheap even when the study has thousands of
        registered entries or a large `mne.SourceSpaces` context. Useful for
        deciding whether a saved folder is worth a full `load()`, e.g. when
        scanning many per-subject folders before combining them.

        Parameters
        ----------
        folder: str | Path

        Returns
        -------
        dict
            'name', 'group_name', 'subject', 'metadata', 'saved_at',
            'n_entries', 'metrics' (sorted set of registered metric names),
            'labels' (sorted set of registered group_name values).
        """
        folder = Path(folder)
        meta_path = folder/"meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Could not find file '{meta_path.name}' inside metric folder.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        entries = meta.get("entries", [])
        return {
            "name": meta["name"],
            "group_name": meta["group_name"],
            "subject": meta.get("subject"),
            "metadata": meta.get("metadata", {}),
            "saved_at": meta.get("saved_at"),
            "n_entries": len(entries),
            "metrics": sorted({e["metric"] for e in entries}),
            "labels": sorted({e["label"] for e in entries}),
        }

    @staticmethod
    def load(folder: str | Path):
        """
        Loads a MetricSuite from a folder written by `save()` (data.npz + meta.json).
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Could not find folder '{folder}'.")

        meta_path = folder/"meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Could not find file '{meta_path.name}' inside metric folder.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 1. Re-load MNE context components if files exist
        context = {}

        ave_path = folder/"context-ave.fif"
        if ave_path.exists():
            context["ave"] = mne.read_evokeds(ave_path, verbose=False)[0]

        src_path = folder/"context-src.fif"
        if src_path.exists():
            context["src"] = mne.read_source_spaces(src_path, verbose=False)

        # 2. Reconstruct the base MetricSuite object
        suite = MetricSuite(
            name=meta["name"], group_name=meta["group_name"], subject=meta.get("subject"),
            metadata=meta.get("metadata"), context=context,
        )

        # 3. Repopulate entries from the array store. Uses read_npz_fast
        # instead of np.load/NpzFile: NpzFile.__getitem__ does an O(n) list
        # scan on every lookup, making a loop of n lookups O(n^2) -- 36.6s
        # for a real 88,000-entry study, of which 32.7s was that scan alone
        # (see read_npz_fast's docstring). read_npz_fast reads every member
        # once, O(n) total, dropping the same load to ~2s.
        data_path = folder/"data.npz"
        arrays = read_npz_fast(data_path)
        for i, entry_meta in enumerate(meta["entries"]):
            value = arrays[f"entry{i}__value"]
            coords = dict(entry_meta["scalar_coords"])
            for c_name in entry_meta["array_coord_names"]:
                coords[c_name] = arrays[f"entry{i}__coord__{c_name}"]
            suite.register(entry_meta["label"], entry_meta["metric"], value, **coords)

        return suite

    @classmethod
    def concat(cls, suites, name="Combined"):
        """
        Combines multiple MetricSuites -- typically one per subject, each
        loaded via `load()` -- into a single suite for group-level analysis.

        Each input suite's entries are pooled as-is: any "Subject" tag
        already on them (from `MetricSuite(subject=...)` or a per-call
        `register(..., subject=...)`) is preserved, since this only merges
        already-registered entries -- it doesn't add subject tags itself.

        `context` (used for `.plot(kind="vista")` spatial/brain plots) is
        not carried over: a merged multi-subject suite generally can't
        share one `src`/`ave` context across subjects, so spatial plotting
        on the result isn't supported unless you set `.context` afterwards
        to something valid for whichever subject(s) you're plotting.

        Parameters
        ----------
        suites: list of MetricSuite
            Suites to combine. Must all share the same `group_name` --
            construct each per-subject suite with the same one (the
            default, "Label", already matches unless overridden).

        name: str
            Name for the combined suite.

        Returns
        -------
        MetricSuite
        """
        if not suites:
            raise ValueError("`suites` must contain at least one MetricSuite.")

        group_name = suites[0].group_name
        for s in suites:
            if s.group_name != group_name:
                raise ValueError(
                    "Cannot combine suites with different group_name "
                    f"('{group_name}' vs '{s.group_name}')."
                )

        combined = cls(name=name, group_name=group_name)
        for s in suites:
            combined._entries.extend(s._entries)

        return combined

    @classmethod
    def load_many(cls, folders, name="Combined"):
        """
        Loads multiple folders written by `save()` and combines them into a
        single suite -- the common "one folder per subject, analyze as a
        group" workflow in one call instead of a hand-written load+concat loop.

        Parameters
        ----------
        folders: list of str | Path

        name: str
            Name for the combined suite.

        Returns
        -------
        MetricSuite
        """
        suites = [cls.load(folder) for folder in folders]
        return cls.concat(suites, name=name)

    @classmethod
    def load_study(cls, root: str | Path, name: str = None):
        """
        Loads every subject subfolder under `root` and combines them into a
        single suite.

        Expects the directory layout produced by saving one MetricSuite per
        subject into `root/<subject>/` -- i.e. every immediate subdirectory
        of `root` that contains a meta.json (its own `save()` output).
        Subfolders without one (stray non-MetricSuite directories) are
        skipped silently.

        Parameters
        ----------
        root: str | Path

        name: str, optional
            Name for the combined suite. Defaults to the root folder's name.

        Returns
        -------
        MetricSuite
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Could not find folder '{root}'.")

        folders = sorted(p for p in root.iterdir() if p.is_dir() and (p/"meta.json").exists())
        if not folders:
            raise FileNotFoundError(f"No saved MetricSuite folders (with a meta.json) found under '{root}'.")

        return cls.load_many(folders, name=name or root.name)


    def plot(self, metric, labels=None, subjects=None, x=None, kind=None, ax=None, **kwargs):
        """
        Flexible plotting interface. Plots a single metric onto a single
        `Axes`. For multiple metrics side by side, call this once per
        metric onto your own `plt.subplots(1, n)` grid, e.g.
        `fig, axes = plt.subplots(1, 2); suite.plot(metric="a", ax=axes[0]); suite.plot(metric="b", ax=axes[1])`.

        Parameters:
        ----------
        metric: str
            Which registered metric to plot.

        labels: list | str
            Filter for specific labels.

        subjects: list | str, optional
            Filter for specific subjects (the "Subject" column registered
            via `MetricSuite(subject=...)` / `register(..., subject=...)`).

        x: str, optional
            The coordinate column name to use as the X-axis. If provided, triggers
            a relational plot.

        kind: str ['auto', 'box', 'violin', 'bar', or 'heatmap']
            Type of plot. Defaults to 'box' for statistical and 'line'
            for relational layouts.

        ax: matplotlib.axes.Axes, optional
            Axes to plot into. A new figure/axes is created if not given.

        **kwargs:
            Passed to the underlying Seaborn/Matplotlib functions. Pass
            `group_name="Subject"` to color by subject instead of by
            the suite's default grouping column.
        """
        if not isinstance(metric, str):
            raise TypeError(
                f"MetricSuite.plot() accepts a single metric (str), got {type(metric).__name__}. "
                "For multiple metrics, call .plot(metric=..., ax=axes[i]) once per metric onto "
                "your own plt.subplots(1, n) grid."
            )

        # get data
        df = self.get_filtered_df(metric, labels, subjects)

        if df.empty:
            print("Query returned no data.")
            return

        # 1. If x is provided, validate it and route to relational plotting.
        if x is not None:
            if x not in df.columns:
                raise ValueError(
                    f"Coordinate column '{x}' does not exist in the registered data."
                )
            if df[x].isna().all():
                raise ValueError(
                    f"Coordinate column '{x}' contains only NaN values for the selected metric/labels."
                )

            # Default relational kind to 'line' if not explicitly stated.
            plot_kind = kind if kind in ["line", "scatter"] else "line"
            return self._plot_relational(df, x=x, kind=plot_kind, ax=ax, **kwargs)

        # 2. If no x is provided, check if fallback to spatial is requested/needed
        spatial_kinds = ["vista"]
        meta_cols = [c for c in df.columns if c not in [self.group_name, "Metric", "Value"]]
        has_spatial = any(c in meta_cols for c in ["channel", "vertex", "region"])

        if kind in spatial_kinds or (kind is None and has_spatial):
            return self._plot_spatial(df, **kwargs)

        # 3. Default fallback: Statistical plots
        plot_kind = kind if kind in ["box", "violin", "bar", "heatmap"] else "box"
        return self._plot_statistical(df, kind=plot_kind, ax=ax, **kwargs)


    def _plot_statistical(self, df, kind="box", ax=None, **kwargs):
        """
        Plot statistical plots.
        """
        plot_kind = {
            "box": metric_box,
            "violin": metric_violin,
            "bar": metric_bar,
            "heatmap": metric_heatmap,
        }

        if kind not in plot_kind:
            raise ValueError(f"Kind '{kind}' not supported. Choose from {list(plot_kind.keys())}")

        kwargs.setdefault("group_name", self.group_name)
        return plot_kind[kind](df, ax=ax, **kwargs)

    def _plot_relational(self, df, x, kind="line", ax=None, **kwargs):
        """
        Plot relational plot.
        """
        kwargs.setdefault("group_name", self.group_name)
        return metric_relational(df, x=x, kind=kind, ax=ax, **kwargs)


    def _plot_spatial(self, df, **kwargs):
        if "channel" in df.columns and df["channel"].notna().any():
            target_col = "channel"
        elif "vertex" in df.columns and df["vertex"].notna().any():
            target_col = "vertex"
        else:
            raise ValueError("No spatial coordinates ('channel' or 'vertex') found in data.")
    
        df_spatial = df.dropna(subset=[target_col])

        pivoted = df_spatial.pivot(index=target_col, columns=self.group_name, values="Value")

        params = {
            "source_space":self.context.get("src"),
            "title": pivoted.columns.tolist(),
        }

        if target_col == "channel":
            params["sensors"] = [self._sync_sensor_data(pivoted[label])
                                 for label in pivoted.columns]
            params["functionals"] = None
        else:
            params["functionals"] = [val.values for _, val in pivoted.items()]
            params["sensors"] = self.context.get("ave")

        params.update(kwargs)
        return plot_brain_activity(**params)


    def _sync_sensor_data(self, series):
        """Aligns a column of data to the MNE Evoked object order."""
        if "ave" not in self.context:
            raise ValueError("Spatial channel plots require an 'ave' (mne.Evoked) in context.")

        evoked = self.context["ave"].copy()

        data_aligned = series.reindex(evoked.ch_names).values

        n_times = evoked.data.shape[1]
        evoked.data = np.tile(data_aligned[:, np.newaxis], (1, n_times))

        return evoked

    def compare_methods(
        self, metric, reference, competitors=None, test="wilcoxon",
        correction="fdr_bh", alpha=0.05, subject_col="region",
        labels=None, subjects=None, plot=False, ax=None, **plot_kwargs,
    ):
        """
        Paired comparison of `reference` against every other (or listed
        `competitors`) method registered under `metric`, matched by
        `subject_col` (default "region" -- the repeated-measures unit used
        throughout this codebase).

        Follows the same get_filtered_df -> transform -> delegate pattern
        as `.plot()`: filters via `get_filtered_df(metrics=metric,
        labels=labels, subjects=subjects)`, then delegates to
        `mindmap.analysis.stats.compare_methods`.

        `reference` is required (unlike `.plot()`'s optional filters) --
        picking a reference automatically would silently determine what
        "significant" even means, which should be an explicit choice.

        Parameters
        ----------
        metric: str
            Which registered "Metric" to compare.

        reference: str
            Required -- the `self.group_name` level every competitor is
            compared against.

        competitors: list[str], optional
            Defaults to every other registered label for this metric.

        test, correction, alpha
            Forwarded to `mindmap.analysis.stats.compare_methods`.

        subject_col: str
            Repeated-measures grouping column, default "region".

        labels, subjects
            Forwarded to `get_filtered_df` (in addition to `metrics=metric`).

        plot: bool
            If True, also builds an annotated box plot via
            `mindmap.viz.stats.metric_box_significance` and returns
            `(results, fig, ax)` instead of just `results`.

        ax: matplotlib.axes.Axes, optional
            Forwarded to `metric_box_significance` when `plot=True`. Axes to
            plot into; a new figure/axes is created if not given.

        **plot_kwargs
            Forwarded to `metric_box_significance` when `plot=True`.

        Returns
        -------
        pd.DataFrame, or (pd.DataFrame, matplotlib.figure.Figure,
        matplotlib.axes.Axes) if `plot=True`.
        """
        df = self.get_filtered_df(metrics=metric, labels=labels, subjects=subjects)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")

        results = _compare_methods_df(
            df, group_col=self.group_name, reference=reference, subject_col=subject_col,
            competitors=competitors, test=test, correction=correction, alpha=alpha,
        )

        if not plot:
            return results

        fig, ax = metric_box_significance(
            df, results, group_name=self.group_name, reference=reference, alpha=alpha, ax=ax, **plot_kwargs,
        )
        return results, fig, ax

    def cluster_permutation(
        self, metric, coord, reference, competitor, subject_col="region",
        higher_is_better=True, coord_order=None, cluster_alpha=0.05,
        n_permutations=3000, seed=0, labels=None, subjects=None,
        plot=False, ax=None, **plot_kwargs,
    ):
        """
        Region-as-subject cluster-permutation test of `reference` vs
        `competitor` along `coord` (e.g. "snr", "percentage"), delegating
        to `mindmap.analysis.stats.cluster_permutation_mne` (which wraps
        `mne.stats.permutation_cluster_1samp_test`).

        Parameters
        ----------
        metric: str
            Which registered "Metric" to compare.

        coord: str
            Coordinate column to sweep along (e.g. "snr").

        reference, competitor: str
            The two `self.group_name` levels being compared.

        subject_col, higher_is_better, coord_order, cluster_alpha,
        n_permutations, seed
            Forwarded to `mindmap.analysis.stats.cluster_permutation_mne`.

        labels, subjects
            Forwarded to `get_filtered_df` in addition to `metrics=metric`
            (`labels` defaults to `[reference, competitor]` when not given,
            so unrelated methods registered under the same metric don't
            leak into the comparison).

        plot: bool
            If True, also builds a shaded curve plot via
            `mindmap.viz.stats.metric_relational_clusters` and returns
            `(result, fig, ax)` instead of just `result`.

        ax: matplotlib.axes.Axes, optional
            Forwarded to `metric_relational_clusters` when `plot=True`. Axes
            to plot into; a new figure/axes is created if not given.

        **plot_kwargs
            Forwarded to `metric_relational_clusters` when `plot=True`.

        Returns
        -------
        dict (see `cluster_permutation_mne`'s docstring for keys:
        "clusters", "t_obs", "coord_values", "diff"), or (dict,
        matplotlib.figure.Figure, matplotlib.axes.Axes) if `plot=True`.
        """
        effective_labels = labels if labels is not None else [reference, competitor]
        df = self.get_filtered_df(metrics=metric, labels=effective_labels, subjects=subjects)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")

        result = _cluster_permutation_mne(
            df, group_col=self.group_name, coord_col=coord, reference=reference, competitor=competitor,
            subject_col=subject_col, higher_is_better=higher_is_better, coord_order=coord_order,
            cluster_alpha=cluster_alpha, n_permutations=n_permutations, seed=seed,
        )

        if not plot:
            return result

        fig, ax = metric_relational_clusters(
            df, result["clusters"], x=coord, group_name=self.group_name,
            alpha=cluster_alpha, ax=ax, **plot_kwargs,
        )
        return result, fig, ax
