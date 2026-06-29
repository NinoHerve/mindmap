"""
Module Name: metric/base.py
Description: Contains MetricSuite class.

The MetricSuite class allows managing all metrics computed within a same study.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

from pathlib import Path

import numpy as np
import pandas as pd

import mne

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
            - 'src':mne.SourceSpaces
            - 'ave':mne.Evoked
        """
        self.name = name
        self._data = []
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
                if isinstance(c_val, (list, tuple, np.ndarray, pd.Series)):
                    if not len(c_val) == n_obs:
                        raise ValueError(
                            "Coordinate lengths must be of the same length as the new metric."
                            f"Instead, coordinate '{c_name}' has length {len(c_val)}."
                        )
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

    def save(self, folder: str | Path):
        """
        Saves the MetricSuite to a CSV file.
        Embeds object metadata safely as commented headers at the top.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata & metrics dataframe
        csv_path = folder/"metrics.csv"
        with open(folder, "w", encoding="utf-8") as f:
            f.write(f"# metric_suite_name: {self.name}\n")
            f.write(f"# metric_suite_group_name: {self.group_name}\n")
        self.dataframe.to_csv(csv_path, mode="a", index=False, header=True)
        
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
        Loads a MetricSuite from a custom CSV file, parsing out the embedded metadata headers.
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Could not find folder '{folder}'.")
        
        csv_path = folder/"metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find file '{csv_path.name}' inside metric folder.")

        name = "Evaluation"
        group_name = "Label"

        # 1. Parse the metadata headers from the top of the file
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# ").strip():
                    # Clean up the comment and split key/value
                    clean_line = line.lstrip("# ").strip()
                    if ":" in clean_line:
                        key, val = clean_line.split(":", 1)
                        key, val = key.strip(), val.strip()
                        if key == "metric_suite_name":
                            name = val
                        elif key == "metric_suite_group_name":
                            group_name = val

                else:
                    # Stop reading once we hit actual data rows
                    break

        # 2. Re-load MNE context components if files exist
        context = {}

        ave_path = folder/"context-ave.fif"
        if ave_path.exists():
            context["ave"] = mne.read_evokeds(ave_path, verbose=False)[0]

        src_path = folder/"context-src.fif"
        if src_path.exists():
            context["src"] = mne.read_source_spaces(src_path, verbose=False)
        
        # 3. Reconstruct the base MetricSuite object
        suite = MetricSuite(name=name, group_name=group_name, context=context)

        # 3. Populate DaraFrame data
        df = pd.read_csv(csv_path, comment="#")
        suite._data = df.to_dict(orient="records")

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
