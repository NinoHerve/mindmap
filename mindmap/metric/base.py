"""
Module Name: metric/base.py
Description: Contains MetricSuite class.

The MetricSuite class allows managing all metrics computed within a same study.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import numpy as np
import pandas as pd
from mindmap.viz.core import (
    metric_bar, metric_box, metric_heatmap, metric_violin, metric_relational
)
from mindmap.viz.dispatch import plot_brain_activity

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
            - 'src':mne.SourceSpace
            - 'ave':mne.Evoked
        """
        self.name = name
        self._data = []
        self.group_name = group_name
        self.context = context or {}

    def register(self, label, metric, values, **coords):
        """Adds results. 'values' can be a single number or a list/array."""
        if not isinstance(values, (list, tuple, np.ndarray, pd.Series,)):
            values = [values]

        n_obs = len(values)

        for i, val in enumerate(values):
            entry = {
                self.group_name: label,
                "Metric": metric,
                "Value": val,
            }

            for c_name, c_val in coords.items():
                if isinstance(c_val, (list, tuple, np.ndarray, pd.Series)) and len(c_val) == n_obs:
                    entry[c_name] = c_val[i] if not isinstance(c_val, pd.Series) else c_val.iloc[i]
                else:
                    entry[c_name] = c_val

            self._data.append(entry)

        return self

    def add_from_function(self, method_name, metric_name, func, *args, **kwargs):
        """Register a new metric given the function to compute it."""
        result = func(*args, **kwargs)
        return self.register(method_name, metric_name, result)

    @property
    def dataframe(self):
        """Dataframe property."""
        return pd.DataFrame(self._data)

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

    def plot(self, metrics=None, labels=None, kind=None, **kwargs):
        """
        Flexible plotting interface.

        Parameters:
        ----------
        metrics: list | str
            Filter for specific metrics.

        labels: list | str
            Filter for specific labels.

        kind: str ['auto', 'box', 'violin', 'bar', or 'heatmap']
            Type of plot.

        **kwargs:
            Passed to the underlying Seaborn/Matplotlib functions.
        """
        # get data
        df = self.get_filtered_df(metrics, labels)

        if df.empty:
            print("Query returned no data.")
            return

        # Indentify plot families
        meta_cols = [c for c in df.columns if c not in [self.group_name, "Metric", "Value"]]
        has_spatial = any(c in meta_cols for c in ["channel", "vertex", "region"])
        has_relational = any(c in meta_cols for c in ["time", "freq", "ms"])

        # map 'kind' to a family
        stat_kinds = ["box", "violin", "bar"]
        rel_kinds = ["line", "scatter"]
        spatial_kinds = ["vista"]

        # decision tree
        if kind in stat_kinds:
            return self._plot_statistical(df, kind, **kwargs)

        if kind in rel_kinds:
            return self._plot_relational(df, kind, **kwargs)

        if kind in spatial_kinds:
            return self._plot_spatial(df, **kwargs)

        if kind is None:
            if has_spatial:
                return self._plot_spatial(df, **kwargs)
            if has_relational:
                return self._plot_relational(df, "line", **kwargs)
            return self._plot_statistical(df, "box", **kwargs)


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

    def _plot_relational(self, df, kind="line", **kwargs):
        """
        Plot relational plot.
        """
        reserved = [self.group_name, "Metric", "Value"]
        all_coords = [c for c in df.columns if c not in reserved]

        return metric_relational(
            df,
            coords = all_coords,
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
