from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from . import config

ANOMALY_TYPES = ("duplicate", "missing_assessment", "inflated_amount")
INFLATION_MULTIPLIER = 2.5
MODIFIED_ZSCORE_THRESHOLD = 3.5


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def detect_duplicate_payments(
    payments: pd.DataFrame, window_days: int = config.PAYMENT_DUPLICATE_WINDOW_DAYS
) -> Tuple[List[str], str]:
    """Flag duplicate payments for the same beneficiary in a short time window."""
    flagged: set[str] = set()
    reasons: List[str] = []
    for beneficiary_id, group in payments.groupby("beneficiary_id"):
        group = group.sort_values("date").reset_index(drop=True)
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            current = group.iloc[idx]
            delta_days = (pd.Timestamp(current["date"]) - pd.Timestamp(prev["date"])).days
            same_amount = np.isclose(float(current["amount"]), float(prev["amount"]))
            if same_amount and delta_days <= window_days:
                flagged.add(str(prev["payment_id"]))
                flagged.add(str(current["payment_id"]))
                reasons.append(
                    f"Duplicate payment for beneficiary {beneficiary_id}: {prev['payment_id']} and {current['payment_id']} within {window_days} days."
                )
    reason = "; ".join(reasons) if reasons else "No duplicate payment patterns detected."
    return sorted(flagged), reason


def detect_missing_assessment(payments: pd.DataFrame, assessments: pd.DataFrame) -> Tuple[List[str], str]:
    """Flag payments missing an assessment reference or using a nonexistent assessment ID."""
    valid_ids = set(assessments["assessment_id"].dropna().astype(str))
    missing = payments[
        (payments["assessment_id"].isna()) | (~payments["assessment_id"].astype(str).isin(valid_ids))
    ]
    flagged = sorted(missing["payment_id"].astype(str).tolist())
    reason = (
        "Missing or invalid assessment references detected."
        if flagged
        else "All payments have valid assessment references."
    )
    return flagged, reason


def detect_zscore_outliers(payments: pd.DataFrame, threshold: float = 3.0) -> Tuple[List[str], str]:
    """Flag payment amounts that are z-score outliers within each damage category."""
    flagged: set[str] = set()
    reasons: List[str] = []
    for category, group in payments.groupby("damage_category"):
        amounts = group["amount"].astype(float).to_numpy()
        if amounts.size < 3:
            continue
        mean = float(np.mean(amounts))
        std = float(np.std(amounts, ddof=0))
        if np.isclose(std, 0.0):
            continue
        z_scores = np.abs((amounts - mean) / std)
        for payment_id, z_score in zip(group["payment_id"].astype(str), z_scores):
            if z_score > threshold:
                flagged.add(payment_id)
                reasons.append(f"Outlier amount in category {category}: payment {payment_id} has |z|={z_score:.2f}.")
    reason = "; ".join(reasons) if reasons else "No outlier payments detected."
    return sorted(flagged), reason


def detect_modified_zscore_outliers(
    payments: pd.DataFrame, threshold: float = MODIFIED_ZSCORE_THRESHOLD
) -> Tuple[List[str], str]:
    """Evaluate inflated amounts with the median/MAD modified z-score."""
    flagged: set[str] = set()
    reasons: List[str] = []
    for category, group in payments.groupby("damage_category"):
        amounts = group["amount"].astype(float).to_numpy()
        if amounts.size < 3:
            continue
        median = float(np.median(amounts))
        mad = float(np.median(np.abs(amounts - median)))
        if np.isclose(mad, 0.0):
            continue
        scores = np.abs(0.6745 * (amounts - median) / mad)
        for payment_id, score in zip(group["payment_id"].astype(str), scores):
            if score > threshold:
                flagged.add(payment_id)
                reasons.append(
                    f"Modified-z outlier in category {category}: "
                    f"payment {payment_id} has |z|={score:.2f}."
                )
    reason = "; ".join(reasons) if reasons else "No modified-z outliers detected."
    return sorted(flagged), reason


def inject_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator | None = None,
    rates: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Inject labelled duplicates, missing-assessment records and inflated amounts."""
    rng = config.RNG if rng is None else rng
    if rates is None:
        rates = {"duplicate": 0.03, "missing_assessment": 0.03, "inflated_amount": 0.03}

    df = df.copy()
    df["ground_truth_duplicate"] = False
    df["ground_truth_missing_assessment"] = False
    df["ground_truth_inflated_amount"] = False

    if not df.empty:
        duplicate_idx = rng.choice(df.index, size=max(1, int(len(df) * rates["duplicate"])), replace=False)
        duplicate_rows: list[pd.Series] = []
        for idx in duplicate_idx:
            row = df.loc[idx].copy()
            row["payment_id"] = f"{row['payment_id']}_dup"
            row["beneficiary_id"] = row["beneficiary_id"]
            row["date"] = pd.Timestamp(row["date"]) + pd.Timedelta(days=1)
            row["ground_truth_duplicate"] = True
            duplicate_rows.append(row)
        if duplicate_rows:
            df = pd.concat([df, pd.DataFrame(duplicate_rows)], ignore_index=True)

        for idx in rng.choice(df.index, size=max(1, int(len(df) * rates["missing_assessment"])), replace=False):
            df.loc[idx, "assessment_id"] = None
            df.loc[idx, "ground_truth_missing_assessment"] = True

        inflated_idx = rng.choice(df.index, size=max(1, int(len(df) * rates["inflated_amount"])), replace=False)
        for idx in inflated_idx:
            df.loc[idx, "amount"] = float(df.loc[idx, "amount"]) * INFLATION_MULTIPLIER
            df.loc[idx, "ground_truth_inflated_amount"] = True

    return df


def evaluate_detection(df: pd.DataFrame, flags: Dict[str, Iterable[str]]) -> Dict[str, Dict[str, float]]:
    """Return recall and precision metrics for each anomaly type."""
    summary: Dict[str, Dict[str, float]] = {}
    overall_true: set[str] = set()
    overall_flagged: set[str] = set()

    for anomaly_name in ANOMALY_TYPES:
        true_label = f"ground_truth_{anomaly_name}"
        true_ids = set(df.loc[df[true_label] == True, "payment_id"].astype(str))
        flagged_ids = set(str(payment_id) for payment_id in flags.get(anomaly_name, []))
        tp = len(true_ids & flagged_ids)
        fp = len(flagged_ids - true_ids)
        fn = len(true_ids - flagged_ids)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        summary[anomaly_name] = {"precision": precision, "recall": recall}
        overall_true |= true_ids
        overall_flagged |= flagged_ids

    overall_tp = len(overall_true & overall_flagged)
    overall_fp = len(overall_flagged - overall_true)
    overall_fn = len(overall_true - overall_flagged)
    summary["overall"] = {
        "precision": overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0,
        "recall": overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0,
    }
    return summary


@dataclass(frozen=True)
class _IndexedPayment:
    payment_id: str
    amount: float
    date: pd.Timestamp


class IncrementalIntegrityChecker:
    """Maintain duplicate and amount statistics for constant-time insert checks."""

    def __init__(
        self,
        valid_assessment_ids: Iterable[str] = (),
        window_days: int = config.PAYMENT_DUPLICATE_WINDOW_DAYS,
        zscore_threshold: float = 3.0,
    ) -> None:
        self.valid_assessment_ids = {str(value) for value in valid_assessment_ids}
        self.window_days = window_days
        self.zscore_threshold = zscore_threshold
        self._beneficiary_disaster: dict[str, list[_IndexedPayment]] = defaultdict(list)
        self._bank_hash: dict[str, list[_IndexedPayment]] = defaultdict(list)
        self._moments: dict[str, tuple[int, float, float]] = {}

    @staticmethod
    def _duplicate_keys(payment: dict[str, object]) -> tuple[str, str | None]:
        beneficiary = str(payment.get("beneficiary_id", ""))
        disaster = str(payment.get("disaster_id", payment.get("camp_id", "")))
        beneficiary_key = _stable_hash(f"{beneficiary}\x1f{disaster}")
        bank_hash = payment.get("bank_hash")
        bank_key = _stable_hash(str(bank_hash)) if bank_hash not in (None, "") else None
        return beneficiary_key, bank_key

    def add_existing(self, payment: dict[str, object]) -> None:
        """Index a historical record without returning detector flags."""
        self._update(payment)

    def check_and_add(self, payment: dict[str, object]) -> dict[str, list[str]]:
        """Check one record against prior state, then add it to the running index."""
        beneficiary_key, bank_key = self._duplicate_keys(payment)
        candidates = list(self._beneficiary_disaster.get(beneficiary_key, ()))
        if bank_key is not None:
            candidates.extend(self._bank_hash.get(bank_key, ()))

        amount = float(payment["amount"])
        date = pd.Timestamp(payment["date"])
        duplicate_ids = sorted({
            candidate.payment_id
            for candidate in candidates
            if np.isclose(candidate.amount, amount)
            and abs((date - candidate.date).days) <= self.window_days
        })

        assessment_id = payment.get("assessment_id")
        missing_assessment = (
            assessment_id is None
            or pd.isna(assessment_id)
            or (self.valid_assessment_ids and str(assessment_id) not in self.valid_assessment_ids)
        )

        category = str(payment["damage_category"])
        count, mean, m2 = self._moments.get(category, (0, 0.0, 0.0))
        outlier = False
        if count >= 3:
            std = float(np.sqrt(m2 / count))
            if not np.isclose(std, 0.0):
                outlier = abs((amount - mean) / std) > self.zscore_threshold

        self._update(payment)
        current_id = str(payment["payment_id"])
        return {
            "duplicate": sorted(set(duplicate_ids + ([current_id] if duplicate_ids else []))),
            "missing_assessment": [current_id] if missing_assessment else [],
            "inflated_amount": [current_id] if outlier else [],
        }

    def _update(self, payment: dict[str, object]) -> None:
        beneficiary_key, bank_key = self._duplicate_keys(payment)
        entry = _IndexedPayment(
            payment_id=str(payment["payment_id"]),
            amount=float(payment["amount"]),
            date=pd.Timestamp(payment["date"]),
        )
        self._beneficiary_disaster[beneficiary_key].append(entry)
        if bank_key is not None:
            self._bank_hash[bank_key].append(entry)

        category = str(payment["damage_category"])
        count, mean, m2 = self._moments.get(category, (0, 0.0, 0.0))
        count += 1
        delta = entry.amount - mean
        mean += delta / count
        m2 += delta * (entry.amount - mean)
        self._moments[category] = (count, mean, m2)


__all__ = [
    "ANOMALY_TYPES",
    "INFLATION_MULTIPLIER",
    "detect_duplicate_payments",
    "detect_missing_assessment",
    "detect_zscore_outliers",
    "detect_modified_zscore_outliers",
    "inject_anomalies",
    "evaluate_detection",
    "IncrementalIntegrityChecker",
]
