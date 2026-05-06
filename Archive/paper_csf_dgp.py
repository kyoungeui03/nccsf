from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _direct_grf_imports import import_grf_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_synthetic_grf = import_grf_module(PROJECT_ROOT, "grf.synthetic.grf")
generate_causal_survival_data = _synthetic_grf.generate_causal_survival_data


# The paper-style four scenarios correspond to the CSF synthetic settings in
# grf.synthetic.grf type1-type4.
#
# We keep both horizon conventions:
# - paper-exact: the original survival-probability thresholds from the CSF paper
# - archive-aligned: survival.probability evaluated at the same time point as RMST
SETTINGS = {
    1: {"dgp": "type1", "rmst_horizon": 1.5, "paper_survival_horizon": 0.8, "aligned_survival_horizon": 1.5},
    2: {"dgp": "type2", "rmst_horizon": 2.0, "paper_survival_horizon": 1.2, "aligned_survival_horizon": 2.0},
    3: {"dgp": "type3", "rmst_horizon": 15.0, "paper_survival_horizon": 10.0, "aligned_survival_horizon": 15.0},
    4: {"dgp": "type4", "rmst_horizon": 3.0, "paper_survival_horizon": 2.0, "aligned_survival_horizon": 3.0},
}


def _setting_spec(setting_id: int) -> dict[str, float | str]:
    try:
        return SETTINGS[int(setting_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported paper CSF setting: {setting_id}") from exc


def _sample_dataset(
    setting_id: int,
    n: int,
    p: int,
    rng: np.random.Generator,
    *,
    protocol: str = "paper-exact",
):
    spec = _setting_spec(setting_id)
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    if protocol == "paper-exact":
        survival_horizon = float(spec["paper_survival_horizon"])
    elif protocol == "archive-aligned":
        survival_horizon = float(spec["aligned_survival_horizon"])
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    data = generate_causal_survival_data(
        n=int(n),
        p=int(p),
        dgp=str(spec["dgp"]),
        Y_max=float(spec["rmst_horizon"]),
        y0=survival_horizon,
        seed=seed,
    )
    return (
        np.asarray(data["X"], dtype=float),
        np.asarray(data["W"], dtype=float),
        np.asarray(data["Y"], dtype=float),
        np.asarray(data["D"], dtype=float),
    )


def _target_horizon(setting_id: int, target: str, *, protocol: str = "paper-exact") -> float:
    spec = _setting_spec(setting_id)
    if target == "RMST":
        return float(spec["rmst_horizon"])
    if target == "survival.probability":
        if protocol == "paper-exact":
            return float(spec["paper_survival_horizon"])
        if protocol == "archive-aligned":
            return float(spec["aligned_survival_horizon"])
        raise ValueError(f"Unsupported protocol: {protocol}")
    raise ValueError(f"Unsupported target: {target}")


def _true_cate(setting_id: int, x: np.ndarray, *, target: str, horizon: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    spec = _setting_spec(setting_id)
    data = generate_causal_survival_data(
        n=x.shape[0],
        p=x.shape[1],
        X=x,
        dgp=str(spec["dgp"]),
        Y_max=float(spec["rmst_horizon"]),
        y0=float(horizon),
        n_mc=20000,
        seed=10_000 + int(setting_id),
    )
    if target == "RMST":
        return np.asarray(data["cate"], dtype=float)
    if target == "survival.probability":
        return np.asarray(data["cate.prob"], dtype=float)
    raise ValueError(f"Unsupported target: {target}")


def _classification_mask(setting_id: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if int(setting_id) == 4:
        return np.asarray(x[:, 0] >= 0.3, dtype=bool)
    return np.ones(x.shape[0], dtype=bool)


def _make_obs_df(x: np.ndarray, a: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    x = np.asarray(x, dtype=float)
    feature_cols = [f"X{i}" for i in range(x.shape[1])]
    obs_df = pd.DataFrame(x, columns=feature_cols)
    obs_df.insert(0, "A", np.asarray(a, dtype=float).ravel())
    obs_df.insert(0, "event", np.asarray(d, dtype=float).ravel())
    obs_df.insert(0, "time", np.asarray(y, dtype=float).ravel())
    return obs_df, feature_cols
