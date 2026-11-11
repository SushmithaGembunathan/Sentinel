from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from relief_framework import allocation, config
from relief_framework.config import ResourceSpec


@pytest.fixture
def worked_example() -> tuple[pd.DataFrame, str, float]:
    config.GROUP_WEIGHTS.clear()
    config.GROUP_WEIGHTS.update({"base": 1.0, "elderly": 1.5, "disabled": 2.0, "maternal": 2.0, "infant": 1.5})
    config.HORIZON_DAYS = 1
    config.PHI = 0.20
    resource = "food"
    config.RESOURCE_SPECS[resource] = ResourceSpec(
        name="food",
        nu0=1.0,
        nu={"base": 0.0, "elderly": 0.0, "disabled": 0.0, "maternal": 0.0, "infant": 0.0},
        rho={"base": 1.0, "elderly": 1.0, "disabled": 1.0, "maternal": 1.0, "infant": 1.0},
    )
    camps = pd.DataFrame(
        {
            "camp_id": ["A", "B"],
            "H": [150, 600],
            "e": [30, 30],
            "d": [20, 10],
            "m": [10, 5],
            "f": [8, 5],
            "stock_food": [0.0, 0.0],
        }
    )
    camps["b"] = camps["H"] - (camps["e"] + camps["d"] + camps["m"] + camps["f"])
    S = 300
    return camps, resource, S


def test_worked_example(worked_example):
    camps, resource, S = worked_example
    P = allocation.priority_scores(camps, resource)
    D = allocation.demand(camps, resource)

    assert P[0] == pytest.approx(1.3266666667, rel=1e-6)
    assert P[1] == pytest.approx(1.0541666667, rel=1e-6)
    assert D.tolist() == pytest.approx([150.0, 600.0], rel=1e-6)

    equity = allocation.allocate_equity(camps, resource, S)
    baseline = allocation.allocate_baseline(camps, resource, S)

    assert equity["x"].tolist() == pytest.approx([150.0, 150.0], rel=1e-6)
    assert baseline["x"].tolist() == pytest.approx([60.0, 240.0], rel=1e-6)
    assert equity["coverage"].tolist() == pytest.approx([1.0, 0.25], rel=1e-6)

    vwc_eq = allocation._build_result(camps, resource, P, D, equity["x"].to_numpy())
    from relief_framework.metrics import vwc

    assert vwc(camps, equity) == pytest.approx(0.682, abs=1e-3)
    assert vwc(camps, baseline) == pytest.approx(0.4000000000, rel=1e-6)

    # Every camp receives at least the fair floor, rounded by the LP's cap.
    assert equity["x"].min() >= 30.0
    assert equity["x"].max() <= 150.0
    assert baseline["x"].sum() <= S
    assert equity["x"].sum() <= S


def test_feasibility_and_bounds(worked_example):
    camps, resource, _ = worked_example
    D = allocation.demand(camps, resource)
    for S in (0.0, 100.0, 400.0, 2000.0):
        equity = allocation.allocate_equity(camps, resource, S)
        baseline = allocation.allocate_baseline(camps, resource, S)

        assert equity["x"].sum() <= S
        assert baseline["x"].sum() <= S
        D_arr = np.asarray(D)
        assert (equity["x"].to_numpy() <= D_arr).all()
        assert (baseline["x"].to_numpy() <= D_arr).all()
        assert ((equity["coverage"].to_numpy() >= 0.0) & (equity["coverage"].to_numpy() <= 1.0)).all()
        assert ((baseline["coverage"].to_numpy() >= 0.0) & (baseline["coverage"].to_numpy() <= 1.0)).all()

    if D.sum() > 0:
        phi_hat = min(config.PHI, 1.0 / (D.sum() / 1.0))
        assert phi_hat <= config.PHI

    # Very small S should not break feasibility.
    tiny = allocation.allocate_equity(camps, resource, 1.0)
    assert tiny["x"].sum() <= 1.0
    assert tiny["x"].min() >= 0.0


def test_rank_tie_breaking_and_denomination():
    config.GROUP_WEIGHTS.clear()
    config.GROUP_WEIGHTS.update({"base": 1.0, "elderly": 1.5, "disabled": 2.0, "maternal": 2.0, "infant": 1.5})
    config.RESOURCE_SPECS["food"] = config.ResourceSpec(
        name="food",
        nu0=1.0,
        nu={"base": 0.0, "elderly": 0.0, "disabled": 0.0, "maternal": 0.0, "infant": 0.0},
        rho={"base": 1.0, "elderly": 1.0, "disabled": 1.0, "maternal": 1.0, "infant": 1.0},
    )
    camps = pd.DataFrame(
        {
            "camp_id": ["B", "A", "C"],
            "H": [100, 100, 100],
            "e": [0, 10, 0],
            "d": [0, 0, 0],
            "m": [0, 0, 0],
            "f": [10, 0, 0],
            "stock_food": [0.0, 0.0, 0.0],
        }
    )
    camps["b"] = camps["H"] - (camps["e"] + camps["d"] + camps["m"] + camps["f"])
    result = allocation.allocate_equity(camps, "food", 300)
    assert result["rank_denominator"].iloc[0] == 3
    assert result["rank"].tolist() == [2, 1, 3]
    assert result.loc[result["camp_id"] == "A", "rank"].iat[0] == 1
    assert result.loc[result["camp_id"] == "B", "rank"].iat[0] == 2
    assert result.loc[result["camp_id"] == "C", "rank"].iat[0] == 3


def test_zero_demand_and_floor_feasibility_and_total_supply_limit(worked_example):
    camps, resource, _ = worked_example
    zero_demand = pd.DataFrame(
        {
            "camp_id": ["C1", "C2"],
            "H": [100, 100],
            "e": [0, 0],
            "d": [0, 0],
            "m": [0, 0],
            "f": [0, 0],
            "stock_food": [1000.0, 1000.0],
        }
    )
    zero_demand["b"] = zero_demand["H"] - (zero_demand["e"] + zero_demand["d"] + zero_demand["m"] + zero_demand["f"])
    result = allocation.allocate_equity(zero_demand, "food", 5.0)
    assert (result["D"] == 0).all()
    assert (result["rank"] == 0).all()
    assert result["coverage"].sum() == 0.0

    tiny = allocation.allocate_equity(camps, resource, 1.0)
    assert tiny["x"].sum() <= 1.0
    assert (tiny["x"].to_numpy() <= tiny["D"].to_numpy()).all()
    assert tiny["x"].min() >= 0.0

    total = allocation.demand(camps, resource).sum()
    for supply in (total * 0.5, total, total * 1.5):
        allocation_result = allocation.allocate_equity(camps, resource, supply)
        assert allocation_result["x"].sum() <= supply
        assert allocation_result["coverage"].between(0.0, 1.0).all()
