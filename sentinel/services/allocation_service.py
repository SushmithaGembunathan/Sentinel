from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from relief_framework import config
from relief_framework.allocation import allocate_baseline, allocate_equity, demand
from relief_framework.metrics import min_coverage, vwc

from sentinel.models import Camp


def camp_to_df(camp: Camp) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "camp_id": camp.camp_id,
            "H": camp.headcount,
            "e": camp.elderly,
            "d": camp.disabled,
            "m": camp.maternal,
            "f": camp.infant,
            "stock_food": camp.stock_food,
            "stock_water": camp.stock_water,
            "stock_medical": camp.stock_medical,
            "stock_infant_supplies": camp.stock_infant_supplies,
        }
    ])


def camps_to_df(camps: Iterable[Camp]) -> pd.DataFrame:
    rows = []
    for camp in camps:
        rows.append({
            "camp_id": camp.camp_id,
            "H": camp.headcount,
            "e": camp.elderly,
            "d": camp.disabled,
            "m": camp.maternal,
            "f": camp.infant,
            "stock_food": camp.stock_food,
            "stock_water": camp.stock_water,
            "stock_medical": camp.stock_medical,
            "stock_infant_supplies": camp.stock_infant_supplies,
        })
    return pd.DataFrame(rows)


def build_reason(camp: Camp, resource: str, allocation_row: dict) -> str:
    total = camp.elderly + camp.disabled + camp.maternal + camp.infant
    headcount = max(camp.headcount, 1)
    elderly_pct = (camp.elderly / headcount) * 100
    disabled_pct = (camp.disabled / headcount) * 100
    return (
        f"{elderly_pct:.0f}% elderly, {disabled_pct:.0f}% disabled; "
        f"ranked {int(allocation_row['rank'])} of {max(len(allocation_row), 1)}"
    )


def run_allocation_for_resource(camps: Iterable[Camp], resource: str, supply_pct: float) -> list[dict]:
    df = camps_to_df(camps)
    total_demand = float(demand(df, resource).sum())
    supply = (float(supply_pct) / 100.0) * total_demand
    baseline = allocate_baseline(df, resource, supply)
    equity = allocate_equity(df, resource, supply)
    results = []
    for _, row in equity.iterrows():
        camp = next((c for c in camps if c.camp_id == row["camp_id"]), None)
        if camp is None:
            continue
        reason = build_reason(camp, resource, row.to_dict())
        results.append({
            "camp_id": row["camp_id"],
            "H": row["H"],
            "priority_score": float(row["P"]),
            "demand": float(row["D"]),
            "recommended_qty": float(row["x"]),
            "coverage": float(row["coverage"]),
            "baseline_qty": float(baseline.loc[baseline["camp_id"] == row["camp_id"], "x"].iat[0]),
            "baseline_coverage": float(baseline.loc[baseline["camp_id"] == row["camp_id"], "coverage"].iat[0]),
            "rank": int(row["rank"]),
            "reason": reason,
            "vwc_equity": vwc(df, equity),
            "vwc_baseline": vwc(df, baseline),
            "min_coverage_equity": min_coverage(equity),
            "min_coverage_baseline": min_coverage(baseline),
        })
    return results
