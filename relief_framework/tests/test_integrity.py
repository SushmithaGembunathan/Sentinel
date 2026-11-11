from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from relief_framework.integrity import (
    IncrementalIntegrityChecker,
    inject_anomalies,
)


def _payment(
    payment_id: str,
    beneficiary_id: str,
    amount: float,
    date: str,
    assessment_id: str = "A1",
    bank_hash: str | None = None,
) -> dict[str, object]:
    return {
        "payment_id": payment_id,
        "beneficiary_id": beneficiary_id,
        "camp_id": "C1",
        "amount": amount,
        "damage_category": "food",
        "assessment_id": assessment_id,
        "date": pd.Timestamp(date),
        "bank_hash": bank_hash,
    }


def test_incremental_checker_flags_pair_missing_assessment_and_outlier() -> None:
    checker = IncrementalIntegrityChecker({"A1"})
    first = _payment("P1", "B1", 100.0, "2025-01-01")
    checker.add_existing(first)

    duplicate = _payment("P2", "B1", 100.0, "2025-01-02")
    assert checker.check_and_add(duplicate)["duplicate"] == ["P1", "P2"]

    missing = _payment("P3", "B2", 101.0, "2025-01-03", assessment_id="")
    assert checker.check_and_add(missing)["missing_assessment"] == ["P3"]

    for payment in (
        _payment("P4", "B4", 99.0, "2025-01-04"),
        _payment("P5", "B5", 100.0, "2025-01-05"),
    ):
        checker.add_existing(payment)
    outlier = _payment("P6", "B6", 1000.0, "2025-01-06")
    assert checker.check_and_add(outlier)["inflated_amount"] == ["P6"]


def test_incremental_checker_uses_bank_hash_lookup() -> None:
    checker = IncrementalIntegrityChecker({"A1"})
    checker.add_existing(
        _payment("P1", "B1", 75.0, "2025-01-01", bank_hash="same-bank")
    )
    result = checker.check_and_add(
        _payment("P2", "B2", 75.0, "2025-01-02", bank_hash="same-bank")
    )
    assert result["duplicate"] == ["P1", "P2"]


def test_inject_anomalies_does_not_concatenate_an_empty_frame() -> None:
    clean = pd.DataFrame(
        {
            "payment_id": [f"P{i}" for i in range(20)],
            "beneficiary_id": [f"B{i}" for i in range(20)],
            "camp_id": ["C1"] * 20,
            "amount": [100.0] * 20,
            "damage_category": ["food"] * 20,
            "assessment_id": [f"A{i}" for i in range(20)],
            "date": pd.date_range("2025-01-01", periods=20),
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        result = inject_anomalies(
            clean,
            rng=np.random.default_rng(42),
            rates={"duplicate": 0.1, "missing_assessment": 0.1, "inflated_amount": 0.1},
        )
    assert len(result) > len(clean)
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
