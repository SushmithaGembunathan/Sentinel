from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from . import config


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    if rng is None:
        return config.RNG
    return rng


def generate_camps(n: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Create synthetic camp-level population and stock information."""
    rng = _ensure_rng(rng)
    headcount = rng.lognormal(mean=5.2, sigma=0.6, size=n).clip(50, 2000).astype(int)

    priority_total = rng.uniform(0.05, 0.30, size=n)
    e = np.zeros(n, dtype=int)
    d = np.zeros(n, dtype=int)
    m = np.zeros(n, dtype=int)
    f = np.zeros(n, dtype=int)

    for i, share in enumerate(priority_total):
        weights = rng.dirichlet(np.full(4, 2.0))
        counts = np.rint(headcount[i] * share * weights).astype(int)
        excess = counts.sum() - headcount[i]
        if excess > 0:
            for idx in np.argsort(counts)[::-1]:
                take = min(excess, counts[idx])
                counts[idx] -= take
                excess -= take
                if excess <= 0:
                    break
        while counts.sum() > headcount[i]:
            idx = int(np.argmax(counts))
            counts[idx] -= 1
        e[i], d[i], m[i], f[i] = counts

    camp_df = pd.DataFrame(
        {
            "camp_id": [f"C{i:03d}" for i in range(1, n + 1)],
            "H": headcount,
            "e": e,
            "d": d,
            "m": m,
            "f": f,
        }
    )
    camp_df["b"] = camp_df["H"] - (camp_df["e"] + camp_df["d"] + camp_df["m"] + camp_df["f"])

    for resource in config.RESOURCE_SPECS:
        spec = config.RESOURCE_SPECS[resource]
        requirement = config.HORIZON_DAYS * (
            spec.nu0 * camp_df["H"]
            + spec.nu["elderly"] * camp_df["e"]
            + spec.nu["disabled"] * camp_df["d"]
            + spec.nu["maternal"] * camp_df["m"]
            + spec.nu["infant"] * camp_df["f"]
        )
        camp_df[f"stock_{resource}"] = requirement * rng.uniform(0.0, 0.30, size=n)

    return camp_df


def generate_disbursals(
    camps: pd.DataFrame, rng: np.random.Generator | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate payment and assessment tables for a camp roster."""
    rng = _ensure_rng(rng)
    payment_rows = []
    assessment_rows = []

    for _, row in camps.iterrows():
        camp_id = row["camp_id"]
        payments_for_camp = int(rng.integers(1, max(3, int(row["H"] / 15))))
        for p_idx in range(payments_for_camp):
            beneficiary_id = f"{camp_id}-B{p_idx + 1}"
            damage_category = rng.choice(list(config.RESOURCE_SPECS.keys()))
            amount = max(20.0, float(rng.lognormal(mean=np.log(180.0), sigma=0.75)))
            assessment_id = f"A{len(assessment_rows) + 1:05d}"
            assessment_rows.append(
                {
                    "assessment_id": assessment_id,
                    "camp_id": camp_id,
                    "damage_category": damage_category,
                    "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 45))),
                    "risk_score": float(rng.uniform(0.1, 0.9)),
                }
            )
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
            payment_rows.append(
                {
                    "payment_id": f"P{len(payment_rows) + 1:05d}",
                    "beneficiary_id": beneficiary_id,
                    "camp_id": camp_id,
                    "amount": amount,
                    "damage_category": damage_category,
                    "assessment_id": assessment_id,
                    "date": date,
                }
            )

    payments = pd.DataFrame(payment_rows)
    assessments = pd.DataFrame(assessment_rows)
    if not assessments.empty:
        assessments = assessments[["assessment_id", "camp_id", "damage_category", "date", "risk_score"]]
    return payments, assessments
