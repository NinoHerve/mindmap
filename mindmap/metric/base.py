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
from pathlib import Path

import numpy as np
import pandas as pd

import mne

from mindmap.viz.core import (
    metric_bar, metric_box, metric_heatmap, metric_violin, metric_relational
)
from mindmap.viz.dispatch import plot_brain_activity


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
    def __init__(self, name="Evaluation", group_name="Label", context=None):
        """
        Docstring for __init__
        
        Parameters
        ----------
        name: str
            Study name

        context: dict
            Context for 3D plotting.
            - 'src':mne.SourceSpaces
            - 'ave':mne.Evoked
        """
        self.name = name
        self._entries = []
        self.group_name = group_name
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

    def register(self, label, metric, values, **coords):
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
        """
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
        reduced = MetricSuite(name=self.name, group_name=self.group_name, context=self.context)
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

    def get_filtered_df(self, metrics=None, labels=None):
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

        return df

    def summary(self):
        """Quick statistical overview."""
        return self.dataframe.groupby(["Metric", self.group_name])["Value"].agg(["mean", "std", "count"])

    def save(self, folder: str | Path):
        """
        Saves the MetricSuite to disk as a compact array store (data.npz) plus
        a small metadata file (meta.json). Keeping arrays as arrays (rather
        than exploding everything into a long-format CSV) is what makes this
        cheap even for studies with thousands of registered per-run curves.
        """
        folder = Path(folder)
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

        np.savez(folder/"data.npz", **arrays)

        with open(folder/"meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {"name": self.name, "group_name": self.group_name, "entries": entries_meta},
                f, indent=2,
            )

        # 2. Save MNE Context Objects if they exist
        if "ave" in self.context and self.context["ave"] is not None:
            ave_path = folder/"context-ave.fif"
            self.context["ave"].save(ave_path, overwrite=True, verbose="warning")

        if "src" in self.context and self.context["src"] is not None:
            src_path = folder/"context-src.fif"
            mne.write_source_spaces(src_path, self.context["src"], overwrite=True, verbose="warning")

        print(f"MetricSuite and context successfully saved to folder: {folder}")

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
        suite = MetricSuite(name=meta["name"], group_name=meta["group_name"], context=context)

        # 3. Repopulate entries from the array store
        data_path = folder/"data.npz"
        with np.load(data_path) as data:
            for i, entry_meta in enumerate(meta["entries"]):
                value = data[f"entry{i}__value"]
                coords = dict(entry_meta["scalar_coords"])
                for c_name in entry_meta["array_coord_names"]:
                    coords[c_name] = data[f"entry{i}__coord__{c_name}"]
                suite.register(entry_meta["label"], entry_meta["metric"], value, **coords)

        return suite


    def plot(self, metrics=None, labels=None, coord=None, kind=None, **kwargs):
        """
        Flexible plotting interface.

        Parameters:
        ----------
        metrics: list | str
            Filter for specific metrics.

        labels: list | str
            Filter for specific labels.
        
        coord: str, optional
            The coordinate column name to use as the X-axis. If provided, triggers
            a relational plot.

        kind: str ['auto', 'box', 'violin', 'bar', or 'heatmap']
            Type of plot. Defaults to 'bo' for statistical and 'line'
            for relational layouts.

        **kwargs:
            Passed to the underlying Seaborn/Matplotlib functions.
        """
        # get data
        df = self.get_filtered_df(metrics, labels)

        if df.empty:
            print("Query returned no data.")
            return
        
        # 1. If coord is provided, validate it and route to relational plotting.
        if coord is not None:
            if coord not in df.columns:
                raise ValueError(
                    f"Coordinate column '{coord}' does not exist in the registered data."
                )
            if df[coord].isna().all():
                raise ValueError(
                    f"Coordinate column '{coord}' contains only NaN values for the selected metrics/labels."
                )
            
            # Default relational kind to 'line' if not explicitly stated.
            plot_kind = kind if kind in ["line", "scatter"] else "line"
            return self._plot_relational(df, x_coord=coord, kind=plot_kind, **kwargs)
        
        # 2. If no coord is provided, check if fallback to spatial is requested/needed
        spatial_kinds = ["vista"]
        meta_cols = [c for c in df.columns if c not in [self.group_name, "Metric", "Value"]]
        has_spatial = any(c in meta_cols for c in ["channel", "vertex", "region"])

        if kind in spatial_kinds or (kind is None and has_spatial):
            return self._plot_spatial(df, **kwargs)
        
        # 3. Default fallback: Statistical plots
        plot_kind = kind if kind in ["box", "violin", "bar", "heatmap"] else "box"
        return self._plot_statistical(df, kind=plot_kind, **kwargs) 


    def _plot_statistical(self, df, kind="box", **kwargs):
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

        return plot_kind[kind](df, **kwargs)

    def _plot_relational(self, df, x_coord, kind="line", **kwargs):
        """
        Plot relational plot.
        """
        return metric_relational(
            df,
            coords = [x_coord],
            hue = self.group_name,
            kind = kind,
            **kwargs,
        )


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
