"""
This module performs unit tests for core/io.py.
"""

import numpy as np

from mindmap.core.io import read_npz_fast


def test_read_npz_fast_matches_np_load(tmp_path):
    """
    read_npz_fast must return exactly what np.load's NpzFile would, for
    every array shape/dtype it stores -- it's a drop-in replacement for the
    NpzFile[key] loop in MetricSuite.load(), just without NpzFile's O(n)
    per-lookup list scan (see read_npz_fast's docstring).
    """
    rng = np.random.default_rng(0)
    arrays = {
        "scalar": np.array([1.5]),
        "vector": rng.normal(size=10),
        "matrix": rng.normal(size=(4, 6)),
        "ints": np.arange(20, dtype=np.int64),
    }
    path = tmp_path / "data.npz"
    np.savez(path, **arrays)

    fast = read_npz_fast(path)
    reference = np.load(path)

    assert set(fast.keys()) == set(reference.files)
    for key in reference.files:
        np.testing.assert_array_equal(fast[key], reference[key])


def test_read_npz_fast_many_small_arrays(tmp_path):
    """
    Regression guard for the actual bug this exists to fix: a large number
    of small arrays (the MetricSuite.save() shape -- one array per
    registered entry) must all round-trip correctly, not just a handful.
    """
    n = 500
    arrays = {f"entry{i}__value": np.array([float(i)]) for i in range(n)}
    path = tmp_path / "data.npz"
    np.savez(path, **arrays)

    fast = read_npz_fast(path)

    assert len(fast) == n
    for i in range(n):
        np.testing.assert_array_equal(fast[f"entry{i}__value"], [float(i)])
