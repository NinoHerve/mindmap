"""
Module Name: analysis/power/variance.py
Description: Mixed-effects variance-component fitting for pilot sample-size
studies.

Author: Nino Herve
"""

import numpy as np
import pandas as pd

from statsmodels.regression.mixed_linear_model import MixedLM


def fit_variance_components(
    df: pd.DataFrame,
    outcome: str,
    group_col: str,
    fixed_col: str,
    vc_formula: dict,
    fixed_levels: list = None,
) -> dict:
    """
    Fits `outcome ~ C(fixed_col)` with a `group_col` random intercept plus
    named variance components (`vc_formula`), and extracts each variance
    component plus the fixed-effect condition means (contrast-coded,
    mean-zeroed).

    This is the per-outcome model the original 6-method pilot used --
    `outcome ~ C(method) + (1|region) + (0+C(method)|region, homoscedastic)
    + (1|region:repetition)`, fit as `groups="region"`,
    `vc_formula={"method_region": "0 + C(method)", "rep": "0 +
    C(repetition)"}` -- generalized so a noise-level or connectome-
    degradation pilot can pass its own `fixed_col`/`group_col`/`vc_formula`
    without duplicating the fitting logic.

    A full unstructured `re_formula="~C(fixed_col)"` random-slope covariance
    is deliberately not used -- it's badly underidentified at pilot scale
    (confirmed singular/non-convergent on the original 208-row, 26-group
    pilot). This variance-components form (`re_formula="~1"` + named
    `vc_formula` terms) is a homoscedasticity simplification that stays
    fittable at that scale.

    Parameters
    ----------
    df: pd.DataFrame
        Long-format outcome table (e.g. from `load_pilot_checkpoints`), with
        at least `[group_col, fixed_col, outcome]` plus every column
        referenced by `vc_formula`.

    outcome: str
        Column to fit as the response.

    group_col: str
        Column defining the top-level random-intercept grouping (`region`
        in the original pilot).

    fixed_col: str
        Column defining the fixed factor of interest (`method` in the
        original pilot -- the thing sample size is being sized to detect
        differences between).

    vc_formula: dict[str, str]
        Named variance components, passed straight through to
        `MixedLM.from_formula(vc_formula=...)` (e.g. `{"method_region": "0 +
        C(method)", "rep": "0 + C(repetition)"}`).

    fixed_levels: list[str], optional
        Fixed ordering for the returned `condition_means`. Defaults to the
        sorted unique values of `df[fixed_col]`.

    Returns
    -------
    result: dict
        `sigma2_{group_col}`, `sigma2_{name}` for every `vc_formula` key,
        `sigma2_resid`, and `condition_means` (list, in `fixed_levels`
        order, mean-zeroed).
    """
    data = df.dropna(subset=[group_col, fixed_col, outcome]).copy()
    data[group_col] = data[group_col].astype("category")
    data[fixed_col] = data[fixed_col].astype("category")

    model = MixedLM.from_formula(
        f"{outcome} ~ C({fixed_col})",
        groups=group_col,
        re_formula="~1",
        vc_formula=vc_formula,
        data=data,
    )
    fit = model.fit(reml=True)

    vc_names = list(model.exog_vc.names)
    result = {f"sigma2_{group_col}": float(fit.cov_re.values[0, 0])}
    for name in vc_formula:
        result[f"sigma2_{name}"] = float(fit.vcomp[vc_names.index(name)])
    result["sigma2_resid"] = float(fit.scale)

    # patsy's C(fixed_col) reference level sorts alphabetically among the
    # string values actually present, not necessarily fixed_levels[0] --
    # recover it from fe_params' index rather than assuming an order.
    levels = fixed_levels if fixed_levels is not None else sorted(df[fixed_col].unique())
    fe_keys = [k for k in fit.fe_params.index if k.startswith(f"C({fixed_col})[T.")]
    included = [k.split("[T.")[1].rstrip("]") for k in fe_keys]
    reference = (set(levels) - set(included)).pop()
    effect = {reference: 0.0}
    for k, lvl in zip(fe_keys, included):
        effect[lvl] = float(fit.fe_params[k])
    means = np.array([effect[lvl] for lvl in levels])
    result["condition_means"] = (means - means.mean()).tolist()

    return result
