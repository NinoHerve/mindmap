"""
Module Name: metric/base.py
Description: Contains MetricSuite class.

The MetricSuite class allows managing all metrics computed within a same study.

Author: Nino Herve
Created: 2025-10-27
Last Modified: 2025-10-27
License: MIT
"""

import pandas as pd
from mindmap.viz.core import metric_bar, metric_box, metric_heatmap, metric_violin

class MetricSuite:
    """Manages metrics from a same study."""
    def __init__(self, name="Evaluation"):
        self.name = name
        self._data = []

    def register(self, method, metric, values):
        """Adds results. 'values' can be a single number or a list/array."""
        if not isinstance(values, (list, tuple, pd.Series)):
            values = [values]

        for val in values:
            self._data.append({
                "Method": method,
                "Metric": metric,
                "Value": val,
            })
        return self

    def add_from_function(self, method_name, metric_name, func, *args, **kwargs):
        """Register a new metric given the function to compute it."""
        result = func(*args, **kwargs)
        return self.register(method_name, metric_name, result)

    @property
    def dataframe(self):
        """Dataframe property."""
        return pd.DataFrame(self._data)

    def summary(self):
        """Quick statistical overview."""
        return self.dataframe.groupby(["Metric", "Method"])["Value"].agg("mean", "std", "count")

    def plot(self, metrics=None, methods=None, kind="auto", **kwargs):
        """
        Flexible plotting interface.

        Parameters:
        ----------
        metrics: list | str
            Filter for specific metrics.

        methods: list | str
            Filter for specific methods.

        kind: str ['auto', 'box', 'violin', 'bar', or 'heatmap']
            Type of plot.

        **kwargs:
            Passed to the underlying Seaborn/Matplotlib functions.
        """

        df = self.dataframe

        # flexible filtering
        if metrics:
            metrics = [metrics] if isinstance(metrics, str) else metrics
            df = df[df["Metric"].isin(metrics)]
        if methods:
            methods = [methods] if isinstance(methods, str) else methods
            df = df[df["Method"].isin(methods)]

        if df.empty:
            print("Query returned no data.")
            return

        plot_methods = {
            "box": metric_box,
            "violin": metric_violin,
            "bar": metric_bar,
            "heatmap": metric_heatmap,
        }

        if kind not in plot_methods:
            raise ValueError(f"Kind '{kind}' not supported. Choose from {list(plot_methods.keys())}")

        return plot_methods[kind](df, **kwargs)


