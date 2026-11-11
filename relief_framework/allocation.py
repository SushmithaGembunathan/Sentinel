from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from . import config


GROUP_ORDER = ("base", "elderly", "disabled", "maternal", "infant")


def _camp_group_values(camps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H = camps["H"].to_numpy(dtype=float)
    if {"e", "d", "m", "f"}.issubset(camps.columns):
        e = camps["e"].to_numpy(dtype=float)
        d = camps["d"].to_numpy(dtype=float)
        m = camps["m"].to_numpy(dtype=float)
        f = camps["f"].to_numpy(dtype=float)
    else:
        e = np.zeros_like(H)
        d = np.zeros_like(H)
        m = np.zeros_like(H)
        f = np.zeros_like(H)
    return H, e, d, m, f


def priority_scores(camps: pd.DataFrame, resource: str) -> np.ndarray:
    """Compute the priority score P_i(r) for each camp."""
    resource_spec = config.RESOURCE_SPECS[resource]
    H, e, d, m, f = _camp_group_values(camps)
    b = np.maximum(H - (e + d + m + f), 0.0)
    weights = config.GROUP_WEIGHTS
    omega = {
        "base": weights["base"] * resource_spec.rho["base"],
        "elderly": weights["elderly"] * resource_spec.rho["elderly"],
        "disabled": weights["disabled"] * resource_spec.rho["disabled"],
        "maternal": weights["maternal"] * resource_spec.rho["maternal"],
        "infant": weights["infant"] * resource_spec.rho["infant"],
    }
    numerator = (
        omega["base"] * b
        + omega["elderly"] * e
        + omega["disabled"] * d
        + omega["maternal"] * m
        + omega["infant"] * f
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.divide(numerator, H)
    return np.where(H > 0, score, 0.0)


def demand(camps: pd.DataFrame, resource: str) -> np.ndarray:
    """Compute the total demand D_i(r) for each camp."""
    resource_spec = config.RESOURCE_SPECS[resource]
    H, e, d, m, f = _camp_group_values(camps)
    stock_col = f"stock_{resource}"
    if stock_col in camps.columns:
        stock = camps[stock_col].to_numpy(dtype=float)
    else:
        stock = np.zeros_like(H)
    requirement = config.HORIZON_DAYS * (
        resource_spec.nu0 * H
        + resource_spec.nu["elderly"] * e
        + resource_spec.nu["disabled"] * d
        + resource_spec.nu["maternal"] * m
        + resource_spec.nu["infant"] * f
    )
    return np.maximum(requirement - stock, 0.0)


def _round_allocation(x: np.ndarray, cap: float, D: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    D = np.asarray(D, dtype=float)
    x = np.clip(x, 0.0, D)

    if cap <= 0.0:
        return np.zeros_like(x)

    if x.sum() <= cap:
        return x

    floored = np.floor(x)
    floored = np.clip(floored, 0.0, D)
    remaining = cap - floored.sum()
    while remaining > 0:
        order = np.argsort(x - floored)[::-1]
        for idx in order:
            if remaining <= 0:
                break
            if floored[idx] < D[idx]:
                floored[idx] += 1.0
                remaining -= 1.0
        if np.all(floored >= D):
            break
    while floored.sum() > cap:
        idx = int(np.argmax(floored))
        if floored[idx] <= 0:
            break
        floored[idx] -= 1.0
    return np.clip(floored, 0.0, D)


def _rank_camps(camps: pd.DataFrame, P: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, int]:
    """Rank camps by priority, then vulnerable headcount, then camp id.

    The ranking is defined only over camps with positive shortfall (D > 0): zero-demand
    camps are excluded from the denominator and keep rank 0. This matches the allocation
    screen's requirement to show the number of camps that are actually in contention for the
    selected resource.
    """
    positive = np.asarray(D, dtype=float) > 0.0
    idx = np.arange(len(camps))
    ranked_idx = idx[positive]
    if ranked_idx.size == 0:
        return np.zeros(len(camps), dtype=int), 0

    vulnerable = np.zeros(len(camps), dtype=float)
    for group_name in ["e", "d", "m", "f"]:
        if group_name in camps.columns:
            vulnerable += camps[group_name].to_numpy(dtype=float)

    ordered = sorted(
        ranked_idx.tolist(),
        key=lambda i: (
            -float(P[i]),
            -float(vulnerable[i]),
            str(camps["camp_id"].iloc[i]),
        ),
    )
    ranks = np.zeros(len(camps), dtype=int)
    for rank_value, camp_index in enumerate(ordered, start=1):
        ranks[camp_index] = rank_value
    return ranks, int(ranked_idx.size)


def _build_result(camps: pd.DataFrame, resource: str, P: np.ndarray, D: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    # Deterministic tie-breaking: equal priority scores are ordered by higher vulnerable
    # headcount first and then by camp_id to keep the ranking reproducible across runs.
    ranks, rank_denominator = _rank_camps(camps, P, D)
    result = pd.DataFrame(
        {
            "camp_id": camps["camp_id"].tolist(),
            "H": camps["H"].to_numpy(dtype=float),
            "P": P,
            "D": D,
            "x": x,
            "coverage": np.divide(x, D, out=np.zeros_like(D, dtype=float), where=D > 0),
            "rank": ranks,
            "rank_denominator": rank_denominator,
        }
    )

    for group_name in ["e", "d", "m", "f", "b"]:
        if group_name in camps.columns:
            result[group_name] = camps[group_name].to_numpy(dtype=float)

    for key in ["base", "elderly", "disabled", "maternal", "infant"]:
        result[f"w_{key}"] = config.GROUP_WEIGHTS[key]
        result[f"rho_{key}"] = config.RESOURCE_SPECS[resource].rho[key]

    return result


def allocate_equity(camps: pd.DataFrame, resource: str, S: float, floor: float | None = None) -> pd.DataFrame:
    """Equity-weighted allocation solved as a linear program with a minimum floor.

    The default floor follows the configured PHI value, but callers can override it to
    evaluate the no-floor variant used in experiments.
    """
    P = priority_scores(camps, resource)
    D = demand(camps, resource)
    total_demand = float(np.sum(D))

    if total_demand <= 0.0 or S <= 0.0:
        result = _build_result(camps, resource, P, D, np.zeros_like(D))
        return result

    if S >= total_demand:
        x = D.copy()
        return _build_result(camps, resource, P, D, x)

    floor_value = config.PHI if floor is None else float(floor)
    phi_hat = min(floor_value, S / total_demand)
    bounds = [(max(phi_hat * demand_i, 0.0), demand_i) for demand_i in D]
    objective = -P.copy()
    constraints = np.ones((1, len(D)))
    lp = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=np.array([S], dtype=float),
        bounds=bounds,
        method="highs",
    )
    if lp.success:
        x = np.clip(lp.x, 0.0, D)
    else:
        x = np.clip(np.maximum(phi_hat * D, 0.0), 0.0, D)

    x = _round_allocation(x, S, D)
    return _build_result(camps, resource, P, D, x)


def allocate_weighted_proportional(camps: pd.DataFrame, resource: str, S: float) -> pd.DataFrame:
    """Allocate in proportion to P_i * D_i, capped by the camp's demand."""
    P = priority_scores(camps, resource)
    D = demand(camps, resource)
    total_demand = float(np.sum(D))
    total_weighted = float(np.sum(P * D))

    if total_demand <= 0.0 or S <= 0.0:
        result = _build_result(camps, resource, P, D, np.zeros_like(D))
        return result

    if S >= total_demand:
        x = D.copy()
        return _build_result(camps, resource, P, D, x)

    if total_weighted <= 0.0:
        raw = S * D / total_demand
    else:
        raw = S * (P * D) / total_weighted
    x = _round_allocation(np.clip(raw, 0.0, D), S, D)
    return _build_result(camps, resource, P, D, x)


def allocate_baseline(camps: pd.DataFrame, resource: str, S: float) -> pd.DataFrame:
    """Proportional allocation baseline capped by each camp's demand."""
    P = priority_scores(camps, resource)
    D = demand(camps, resource)
    total_demand = float(np.sum(D))

    if total_demand <= 0.0 or S <= 0.0:
        result = _build_result(camps, resource, P, D, np.zeros_like(D))
        return result

    if S >= total_demand:
        x = D.copy()
        return _build_result(camps, resource, P, D, x)

    raw = S * D / total_demand
    x = _round_allocation(np.clip(raw, 0.0, D), S, D)
    return _build_result(camps, resource, P, D, x)
