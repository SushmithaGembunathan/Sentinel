from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .allocation import priority_scores


def coverage_from_allocation(allocation: pd.DataFrame) -> np.ndarray:
    """Compute the camp-level coverage ratio for a given allocation result."""
    D = allocation["D"].to_numpy(dtype=float)
    x = allocation["x"].to_numpy(dtype=float)
    return np.divide(x, D, out=np.zeros_like(D, dtype=float), where=D > 0)


def vwc(camps: pd.DataFrame, allocation: pd.DataFrame) -> float:
    """Weighted coverage value (VWC) from the paper's definition."""
    c = coverage_from_allocation(allocation)
    priority_v = camps["e"].to_numpy(dtype=float) + camps["d"].to_numpy(dtype=float) + camps["m"].to_numpy(dtype=float) + camps["f"].to_numpy(dtype=float)
    denom = np.sum(priority_v)
    if denom <= 0:
        return 0.0
    return float(np.sum(priority_v * c) / denom)


def base_group_coverage(camps: pd.DataFrame, allocation: pd.DataFrame) -> float:
    """Coverage of non-priority residents (the base group b = H - (e+d+m+f))."""
    c = coverage_from_allocation(allocation)
    if "b" in camps.columns:
        base = camps["b"].to_numpy(dtype=float)
    else:
        base = camps["H"].to_numpy(dtype=float) - (
            camps["e"].to_numpy(dtype=float)
            + camps["d"].to_numpy(dtype=float)
            + camps["m"].to_numpy(dtype=float)
            + camps["f"].to_numpy(dtype=float)
        )
    denom = float(np.sum(base))
    if denom <= 0:
        return 0.0
    return float(np.sum(base * c) / denom)


def min_coverage(allocation: pd.DataFrame) -> float:
    """Minimum positive coverage across camps with positive demand."""
    c = coverage_from_allocation(allocation)
    positive = c[c > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.min(positive))


def sensitivity(camps: pd.DataFrame, resource: str, S: float) -> Tuple[float, float]:
    """Perturb each weight by ±25% and summarize Spearman sensitivity."""
    default_order = np.argsort(-priority_scores(camps, resource))
    default_ranks = np.empty_like(default_order)
    default_ranks[default_order] = np.arange(len(default_order))

    rho_values = []
    for weight_name in ["elderly", "disabled", "maternal", "infant", "base"]:
        for factor in (1.25, 0.75):
            weights = config.GROUP_WEIGHTS.copy()
            weights[weight_name] = config.GROUP_WEIGHTS[weight_name] * factor
            old = config.GROUP_WEIGHTS.copy()
            config.GROUP_WEIGHTS.update(weights)
            try:
                perturbed_order = np.argsort(-priority_scores(camps, resource))
                perturbed_ranks = np.empty_like(perturbed_order)
                perturbed_ranks[perturbed_order] = np.arange(len(perturbed_order))
                rho = spearmanr(default_ranks, perturbed_ranks).statistic
                rho_values.append(float(rho))
            finally:
                config.GROUP_WEIGHTS.clear()
                config.GROUP_WEIGHTS.update(old)

    if not rho_values:
        return 0.0, 0.0
    return float(np.mean(rho_values)), float(np.min(rho_values))


__all__ = ["coverage_from_allocation", "vwc", "base_group_coverage", "min_coverage", "sensitivity"]
