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
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mindmap.core.io import read_npz_fast, atomic_replace, json_safe

from mindmap.metric.results import PairwiseResult, ClusterResult, NormalityResult, AnovaResult
from mindmap.metric.plot import PlotAccessor
from mindmap.metric.stat import StatAccessor


def _is_arraylike(val):
    """Whether val should be treated as a per-point coordinate rather than a scalar tag."""
    return isinstance(val, (list, tuple, np.ndarray, pd.Series))


def _entry_to_frame(entry):
    """
    Materializes a single registered entry into a long-format DataFrame slice.

    entry["value"] is an array of shape (n_points,) for a single curve, or
    (n_reps, n_points) for a stack of repeated observations (e.g. one curve
    per simulation run) sharing the same point-wise coordinates.
    """
    value = np.atleast_1d(np.asarray(entry["value"]))
    metric, coords = entry["metric"], entry["coords"]

    base = {"Metric": metric}

    if value.ndim == 1:
        data = {**base, "Value": value}
        for name, coord in coords.items():
            data[name] = coord
        return pd.DataFrame(data)

    if value.ndim != 2:
        raise ValueError(
            f"Registered values must be 1D or 2D (got shape {value.shape} for metric='{metric}')."
        )

    # (n_reps, n_points): tile point-wise coordinates across reps, repeat
    # rep-wise coordinates across points, broadcast scalar coordinates
    # as-is, and always tag each row with its repetition index (in addition
    # to any rep-wise coordinate given) so repeated curves (e.g. one per
    # simulation run) stay distinguishable for CI estimation even when no
    # rep-wise coordinate was registered.
    n_reps, n_points = value.shape
    data = {**base, "Value": value.ravel(), "rep": np.repeat(np.arange(n_reps), n_points)}

    for name, coord in coords.items():
        coord_arr = np.asarray(coord) if _is_arraylike(coord) else None
        if coord_arr is not None and coord_arr.shape[0] == n_points:
            data[name] = np.tile(coord_arr, n_reps)
        elif coord_arr is not None and coord_arr.shape[0] == n_reps:
            data[name] = np.repeat(coord_arr, n_points)
        else:
            data[name] = coord

    return pd.DataFrame(data)


class MetricSuite:
    """
    Collects every metric from one study -- register results as you compute
    them, then filter, plot, and run statistics on the whole collection
    without juggling separate arrays or DataFrames yourself.
    """
    def __init__(self, name="Evaluation", metadata=None):
        """
        Creates an empty suite, ready for `register()` calls.

        !!! note "Technical details"
            `metadata` never becomes a per-row column and never survives
            `concat`/pooling -- if a tag needs to differentiate rows (now,
            or after a future pooling step across several suites), register
            it via `register(..., **coords)` instead; `metadata` is only
            for context that will never need that (must be
            JSON-serializable).

        Parameters
        ----------
        name: str, optional
            A name for this study, used only for your own reference (e.g.
            shown when saving).

        metadata: dict, optional
            Free-form notes to keep alongside the results (e.g. solver
            settings, a dataset path, a one-line description) -- saved and
            reloaded as-is.
        """
        self.name = name
        self._entries = []

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        self.metadata = metadata or {}

        # suite.plot.box(...)/.violin(...)/.bar(...)/.line(...)/.scatter(...) for raw views (metric
        # name in, no test); suite.plot.group_comparison(result)/.cluster_permutation(result) for
        # annotated views (an already-computed PairwiseResult/ClusterResult in, never a metric name)
        # -- see mindmap.metric.plot.PlotAccessor.
        self.plot = PlotAccessor(self)

        # suite.stat.wilcoxon(...)/.ttest(...)/.mannwhitneyu(...)/.paired_ttest(...)/.ttest_ess(...)/
        # .cluster_permutation(...)/.normality_test(...)/.anova(...) -- pure computation, cached,
        # typed results -- see mindmap.metric.stat.StatAccessor.
        self.stat = StatAccessor(self)

    def register(self, metric, values, **coords):
        """
        Adds one result to the study: a value (or a curve, or a batch of
        repeated curves) for one metric, tagged with whatever labels you
        want to filter, group, or plot by later -- there's no dedicated
        "group" argument here; pick any tag name you like (e.g. `Method=`,
        `Subject=`) and point `group_by=` at it when you run a comparison
        or plot.

        !!! note "Technical details"
            'values' can be a scalar, a 1D array of shape (n_points,) for a
            single curve/observation, or a 2D array of shape (n_reps,
            n_points) for a stack of repeated observations (e.g. one curve
            per simulation run) sharing the same point-wise coordinates.

            Every keyword coordinate in '**coords' must be one of:

            - a genuine scalar (anything that isn't a
              list/tuple/np.ndarray/pd.Series -- e.g. a str, int, float,
              bool): broadcast as a constant tag to every row (e.g.
              `Method="wCSET"` or `Subject="sub-01"` -- there is no
              suite-level shortcut for a tag you want constant across many
              calls, register it explicitly every time, e.g. via a loop
              variable or `functools.partial(suite.register, Subject=...)`).
            - an array-like whose length exactly matches the last axis of
              'values' ("n_points"): point-wise (e.g. 'times', 'recall'),
              tiled across every repetition if 'values' is 2D.
            - only when 'values' is 2D, an array-like whose length matches
              the *first* axis instead ("n_reps"): tags each repetition
              itself (e.g. a per-run 'seed'), repeated across every point
              within that repetition. This rides alongside the automatic,
              always-present "rep" index column (0..n_reps-1) -- it
              doesn't replace it.

            An array-like matching neither length raises `ValueError` --
            there is no silent "broadcast a mismatched array" fallback.

            This still infers a coordinate's role from length alone,
            though: an array that happens to have exactly n_points (or,
            for a 2D stack, n_reps) elements *by coincidence*, without
            actually being meant to line up position-for-position with
            'values', will still be (silently, incorrectly) treated as
            point-wise/rep-wise -- n_points is checked first, so a
            coincidental n_points == n_reps match resolves as point-wise.
            There is currently no way to force a same-length array to be
            treated as a whole-entry scalar tag instead -- if that
            matters, wrap it so its outer length differs from both, or
            avoid the coincidence some other way.

            Data is kept as arrays and only expanded into a long-format
            table lazily, when `to_df()`/`suite.stat.wilcoxon`/etc.
            are accessed - this keeps registering large stacks (e.g.
            thousands of per-run curves) cheap.

            Adding an entry always clears every cached stats result (see
            `suite.stat.wilcoxon`/`.cluster_permutation`/
            `.list_cached_stats`) -- cheaper and safer than figuring out
            which cached results a given new entry could affect, and it
            guarantees a suite never silently returns stats computed
            before some of its current data existed.

        Parameters
        ----------
        metric: str
            Name of the outcome being recorded (e.g. "correlation"). A
            suite can hold several metrics at once; most methods look at
            one at a time via `metrics=...`.

        values: scalar | array-like of shape (n_points,) | array-like of shape (n_reps, n_points)
            The measurement(s) themselves.

        **coords
            Any labels you want attached (e.g. `Method=...`, `region=...`,
            `times=...`, `condition=...`, `Subject=...`) -- each keyword
            becomes its own column. At least one is usually needed for
            `suite.stat.wilcoxon`/`.cluster_permutation`'s
            `group_by` to have something to point at.
        """
        value = np.asarray(values) if _is_arraylike(values) else np.asarray([values])

        n_points = value.shape[-1]
        n_reps = value.shape[0] if value.ndim == 2 else None
        for name, val in coords.items():
            if not _is_arraylike(val):
                continue
            length = len(val)
            if length == n_points or (n_reps is not None and length == n_reps):
                continue
            expected = f"{n_points} (n_points)" + (f" or {n_reps} (n_reps)" if n_reps is not None else "")
            raise ValueError(
                f"Coordinate lengths must match {expected}. "
                f"Instead, coordinate '{name}' has length {length}."
            )

        stored_coords = {
            name: (np.asarray(val) if _is_arraylike(val) else val)
            for name, val in coords.items()
        }

        self._entries.append({
            "metric": metric,
            "value": value,
            "coords": stored_coords,
        })

        self.stat.clear_stats_cache()

        return self

    def to_df(self, metrics=None, filters=None):
        """
        Returns the study's data (or a filtered subset of it) as a plain,
        long-format DataFrame -- the building block behind
        `suite.stat.wilcoxon`/`.cluster_permutation`/etc., but also
        the main way to get your numbers out for anything this class
        doesn't cover directly.

        !!! note "Technical details"
            Built on demand from every registered entry (`pd.concat`, one
            row per point/repetition) -- entries registered with different
            coords simply end up with NaN in whichever columns they didn't
            provide, same as any long-format table pooling heterogeneous
            sources.

            Each of `metrics`/`filters`' values accepts a single value or
            a list, matched via `.isin()`. `filters` matches any
            registered column (raises `ValueError` if that column was
            never registered) -- always a `dict`, never bare keyword
            arguments, matching the calling convention every other method
            that forwards to this one (`suite.stat.wilcoxon`,
            `.cluster_permutation`, ...) uses for `filters` too.

        Parameters
        ----------
        metrics: list | str, optional
            Only keep these "Metric" values.

        filters: dict, optional
            Only keep rows matching these `{column: value(s)}` pairs, e.g.
            `{"Method": "wCSET"}` or `{"Subject": ["sub-01", "sub-02"]}`.
        """
        if not self._entries:
            return pd.DataFrame()
        df = pd.concat([_entry_to_frame(e) for e in self._entries], ignore_index=True)

        if metrics:
            metrics = [metrics] if not _is_arraylike(metrics) else metrics
            df = df[df["Metric"].isin(metrics)]
        for name, val in (filters or {}).items():
            if val is None:
                continue
            if name not in df.columns:
                raise ValueError(f"No column '{name}' has been registered in this MetricSuite.")
            val = [val] if not _is_arraylike(val) else val
            df = df[df[name].isin(val)]

        return df

    def summary(self, metric, group_by=None, filters=None):
        """
        Mean/std/count for one metric -- either over the whole population,
        or per group if `group_by` is given. A quick sanity check before
        diving into plots or tests.

        Parameters
        ----------
        metric: str
            Which registered metric to summarize.

        group_by: str, optional
            Which registered column to group by. Omit to summarize the
            metric's whole population in one row.

        filters: dict, optional
            Only include rows matching these `{column: value(s)}` pairs.
        """
        df = self.to_df(metrics=metric, filters=filters)
        if df.empty:
            raise ValueError(f"No data registered for metric '{metric}'.")
        if group_by is None:
            return df["Value"].agg(["mean", "std", "count"]).to_frame(name=metric).T
        return df.groupby(group_by)["Value"].agg(["mean", "std", "count"]).reset_index()

    def describe(self, metric=None):
        """
        Describes the suite's structure -- which metrics are registered
        and, for each, which coords they carry: "categorical" (a count of
        distinct values) or "numerical" (a `[min, max]` range). A quick
        "what's actually in here" overview before reaching for `summary()`
        (the numbers) or `to_df()` (the raw table).

        !!! note "Technical details"
            A coord only appears under a metric if at least one of that
            metric's rows actually has a non-null value for it -- metrics
            registered with different coord sets don't cross-contaminate
            each other's rows here (unlike `to_df()`'s single wide table,
            where an unrelated metric's coord shows up as all-NaN).
            Categorical/numerical is decided the same way
            `mindmap.viz.core.metric_line`/`metric_scatter` decide their
            x-axis (`pd.api.types.is_numeric_dtype`). Categorical values
            are listed up to 8; beyond that, truncated with a count of how
            many more.

        Parameters
        ----------
        metric: str, optional
            Only describe this metric. Omit to describe every registered
            metric.

        Returns
        -------
        pd.DataFrame
            Columns: `"Metric"`, `"Coord"`, `"Kind"` (`"categorical"` |
            `"numerical"`), `"N unique"`, `"Detail"` (a range for
            numerical coords, a value preview for categorical ones).
        """
        df = self.to_df(metrics=metric)
        columns = ["Metric", "Coord", "Kind", "N unique", "Detail"]
        if df.empty:
            return pd.DataFrame(columns=columns)

        rows = []
        for m, sub in df.groupby("Metric"):
            coord_cols = [c for c in sub.columns if c not in ("Metric", "Value")]
            for c in coord_cols:
                series = sub[c].dropna()
                if series.empty:
                    continue
                n_unique = series.nunique()
                if pd.api.types.is_numeric_dtype(series):
                    kind = "numerical"
                    detail = f"[{series.min():g}, {series.max():g}]"
                else:
                    kind = "categorical"
                    values = sorted(series.unique().astype(str))
                    detail = ", ".join(values) if len(values) <= 8 else f"{', '.join(values[:8])}, ... (+{len(values) - 8} more)"
                rows.append({"Metric": m, "Coord": c, "Kind": kind, "N unique": n_unique, "Detail": detail})

        return pd.DataFrame(rows, columns=columns)

    def save(self, folder: str | Path, overwrite: bool = False):
        """
        Saves the whole study to a folder -- raw data and any cached
        statistics -- so it can be reloaded later with `load()`.

        !!! note "Technical details"
            Data is stored as a compact array store (data.npz) plus a
            small metadata file (meta.json), rather than exploding
            everything into a long-format CSV -- this keeps saving cheap
            even for studies with thousands of registered per-run curves.
            Every currently cached stats result (see
            `suite.stat.wilcoxon`/`.cluster_permutation`/
            `.list_cached_stats`) is also persisted, into a `stats/`
            subfolder, so an expensive cluster-permutation test never
            needs to rerun just because a notebook was reopened -- `stats/`
            is entirely rewritten from `self.stat._stats_cache` on every
            save (removed outright if the cache is empty), never left
            holding stale entries from an earlier save.

        Parameters
        ----------
        folder: str | Path
            Where to save.

        overwrite: bool, optional
            If False (default), refuses to write into a folder that
            already holds a previous save, so a typo'd or reused path
            can't silently destroy an earlier study's results.
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
                    scalar_coords[c_name] = json_safe(c_val)

            entries_meta.append({
                "metric": e["metric"],
                "scalar_coords": scalar_coords,
                "array_coord_names": array_coord_names,
            })

        atomic_replace(folder/"data.npz", lambda p: np.savez(p, **arrays))

        # 2. Save/refresh cached stats -- always mirrors self.stat._stats_cache
        # exactly (see docstring above).
        stats_dir = folder/"stats"
        if stats_dir.exists():
            shutil.rmtree(stats_dir)
        if self.stat._stats_cache:
            stats_dir.mkdir(parents=True, exist_ok=True)
            index = {}
            for i, result in enumerate(self.stat._stats_cache.values()):
                sub_name = f"{i:04d}"
                result.save(stats_dir/sub_name)
                index[sub_name] = result.KIND

            def _write_stats_index(p):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2)
            atomic_replace(stats_dir/"index.json", _write_stats_index)

        # 3. Write meta.json last, atomically, and only once the data/stats
        # it describes are already safely on disk -- so a crash can never
        # leave a meta.json that references entries data.npz doesn't have
        # yet (the reverse -- data.npz/stats holding entries meta.json
        # hasn't caught up to describing yet -- is harmless: load() only
        # reads what meta.json points to).
        def _write_meta(p):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": self.name,
                        "metadata": self.metadata,
                        "saved_at": datetime.now().isoformat(),
                        "entries": entries_meta,
                    },
                    f, indent=2,
                )

        atomic_replace(folder/"meta.json", _write_meta)

        print(f"MetricSuite successfully saved to folder: {folder}")

    @staticmethod
    def peek(folder: str | Path) -> dict:
        """
        Peeks at a saved folder's basic info -- name, metrics -- without
        actually loading the data. Handy for deciding whether a folder is
        worth a full `load()`, e.g. when scanning many candidates before
        combining them.

        !!! note "Technical details"
            Reads only meta.json, never the array data -- cheap even for a
            study with thousands of entries.

        Parameters
        ----------
        folder: str | Path
            A folder previously written by `save()`.

        Returns
        -------
        dict
            `"name"`, `"metadata"`, `"saved_at"`, `"n_entries"`,
            `"metrics"` (sorted set of registered metric names).
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
            "metadata": meta.get("metadata", {}),
            "saved_at": meta.get("saved_at"),
            "n_entries": len(entries),
            "metrics": sorted({e["metric"] for e in entries}),
        }

    @staticmethod
    def load(folder: str | Path):
        """
        Loads a study back from a folder written by `save()` -- data and
        cached statistics both included.

        !!! note "Technical details"
            Reconstructs entries from `data.npz`/`meta.json` and, if
            present, repopulates the stats cache from `stats/`, so
            previously computed (and potentially expensive, e.g.
            cluster-permutation) results don't need to be recomputed
            after reloading.

            Folders saved before this class's grouping axis became a
            per-call `group_by` choice stored a suite-level `"group_name"`
            plus a dedicated per-entry `"label"` field, distinct from
            `scalar_coords`. Those are transparently folded into an
            ordinary coord (`coords[group_name] = label`) so old saved
            studies keep loading correctly -- nothing to do differently
            when loading a folder saved by the current code, which never
            writes those fields.

        Parameters
        ----------
        folder: str | Path
            A folder previously written by `save()`.

        Returns
        -------
        MetricSuite
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Could not find folder '{folder}'.")

        meta_path = folder/"meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Could not find file '{meta_path.name}' inside metric folder.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 1. Reconstruct the base MetricSuite object
        suite = MetricSuite(name=meta["name"], metadata=meta.get("metadata"))

        # Legacy saves (pre `group_by` generalization) stored a suite-level
        # "group_name" plus a dedicated per-entry "label" field -- fold that
        # back into an ordinary coord under the same column name.
        legacy_group_name = meta.get("group_name")

        # 2. Repopulate entries from the array store. Uses read_npz_fast
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
            if legacy_group_name is not None and "label" in entry_meta:
                coords[legacy_group_name] = entry_meta["label"]
            suite.register(entry_meta["metric"], value, **coords)

        # 3. Repopulate cached stats, if this save included any. register()
        # above already cleared _stats_cache -- this runs after, so the
        # reloaded cache is the last word.
        stats_dir = folder/"stats"
        index_path = stats_dir/"index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            for sub_name, kind in index.items():
                if kind == PairwiseResult.KIND:
                    result = PairwiseResult.load(stats_dir/sub_name)
                    pair = (result.table.loc[0, "group_a"], result.table.loc[0, "group_b"])
                    key = suite.stat._pairwise_key(
                        result.metric, result.group_by, result.test, result.unit_col, result.pairing_col,
                        result.subject_order, result.filters, pair,
                    )
                elif kind == ClusterResult.KIND:
                    result = ClusterResult.load(stats_dir/sub_name)
                    pair = next(iter(result.clusters_by_pair.keys()))
                    key = suite.stat._cluster_key(
                        result.metric, result.coord, result.pairing_col, result.unit_col, result.group_by,
                        result.higher_is_better, result.coord_order, result.cluster_alpha,
                        result.n_permutations, result.seed, result.effect_size_summary,
                        result.filters, pair,
                    )
                elif kind == NormalityResult.KIND:
                    result = NormalityResult.load(stats_dir/sub_name)
                    group = result.table.loc[0, "group"]
                    key = suite.stat._normality_key(result.metric, result.group_by, result.test, result.filters, group)
                elif kind == AnovaResult.KIND:
                    result = AnovaResult.load(stats_dir/sub_name)
                    key = suite.stat._anova_key(
                        result.metric, result.group_by, result.pairing_col, result.unit_col,
                        result.test, result.filters, result.groups,
                    )
                else:
                    continue
                suite.stat._stats_cache[key] = result

        return suite

    @classmethod
    def concat(cls, suites, name="Combined"):
        """
        Combines several suites into one -- e.g. one per subject, each
        loaded via `load()` -- for group-level analysis.

        !!! note "Technical details"
            Each input suite's entries are pooled exactly as-is: whatever
            tags were already registered on them (via `register(...,
            **coords)`) survive into the combined suite untouched --
            `concat` doesn't inspect or require any particular tag name,
            it just extends the entry lists. If you want to be able to
            tell which input suite a pooled row came from, register a
            distinguishing tag (e.g. `Subject=...`) on every entry of each
            input suite *before* calling this -- there's no suite-level
            mechanism for that, plain explicit `register()` calls are the
            whole mechanism.

            Cached stats are *not* pooled (each input suite's cache was
            computed against its own, narrower data) -- the combined
            suite starts with an empty stats cache, same as any freshly
            constructed `MetricSuite`.

            Works just as well on suites you built by hand as on ones
            reloaded via `load()` -- `concat` doesn't care where a suite's
            entries came from, it just extends the entry lists.

        Parameters
        ----------
        suites: list of MetricSuite
            Suites to combine.

        name: str, optional
            Name for the combined suite.

        Returns
        -------
        MetricSuite
        """
        if not suites:
            raise ValueError("`suites` must contain at least one MetricSuite.")

        combined = cls(name=name)
        for s in suites:
            combined._entries.extend(s._entries)

        return combined

