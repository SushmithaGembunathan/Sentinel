from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

from . import config
from .allocation import (
    allocate_baseline,
    allocate_equity,
    allocate_weighted_proportional,
    demand,
    priority_scores,
)
from .integrity import (
    ANOMALY_TYPES,
    INFLATION_MULTIPLIER,
    MODIFIED_ZSCORE_THRESHOLD,
    IncrementalIntegrityChecker,
    detect_duplicate_payments,
    detect_missing_assessment,
    detect_modified_zscore_outliers,
    detect_zscore_outliers,
    inject_anomalies,
)
from .metrics import base_group_coverage, min_coverage, vwc
from .simulate import generate_camps

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SEEDS = tuple(range(1, 31))
CAMPS_PER_SEED = 50
SUPPLY_LEVELS = config.SUPPLY_RATIOS
ANOMALY_RATES = dict.fromkeys(ANOMALY_TYPES, 0.05)
INTEGRITY_CLEAN_RECORDS = 1000


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _record_metrics(camps: pd.DataFrame, resource: str, supply: float) -> dict[str, dict[str, Any]]:
    allocations = {
        "baseline": allocate_baseline(camps, resource, supply),
        "proposed": allocate_equity(camps, resource, supply),
        "weighted_proportional": allocate_weighted_proportional(camps, resource, supply),
        "proposed_no_floor": allocate_equity(camps, resource, supply, floor=0.0),
    }
    return {
        method: {
            "vwc": vwc(camps, allocation),
            "min_coverage": min_coverage(allocation),
            "base_coverage": base_group_coverage(camps, allocation),
            "zero_coverage_camps": int(
                ((allocation["D"] > 0) & (allocation["coverage"] == 0)).sum()
            ),
            "allocation": allocation,
        }
        for method, allocation in allocations.items()
    }


def _sensitivity_by_resource(camps: pd.DataFrame, resource: str) -> list[dict[str, Any]]:
    default_order = np.argsort(-priority_scores(camps, resource))
    default_ranks = np.empty_like(default_order)
    default_ranks[default_order] = np.arange(len(default_order))
    rows: list[dict[str, Any]] = []
    for weight_name in config.GROUP_NAMES:
        for change, factor in (("-25%", 0.75), ("+25%", 1.25)):
            previous = config.GROUP_WEIGHTS.copy()
            config.GROUP_WEIGHTS[weight_name] = previous[weight_name] * factor
            try:
                perturbed_order = np.argsort(-priority_scores(camps, resource))
                perturbed_ranks = np.empty_like(perturbed_order)
                perturbed_ranks[perturbed_order] = np.arange(len(perturbed_order))
                rho = float(spearmanr(default_ranks, perturbed_ranks).statistic)
            finally:
                config.GROUP_WEIGHTS.clear()
                config.GROUP_WEIGHTS.update(previous)
            rows.append({"weight": weight_name, "change": change, "spearman": rho})
    return rows


def _seed_summary(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    """Summarize seed-level values with two-sided 95% Student-t intervals."""
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = dict(zip(group_columns, keys))
        count = int(group["seed"].nunique())
        critical = float(t.ppf(0.975, count - 1)) if count > 1 else 0.0
        result["n_seeds"] = count
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if count > 1 else 0.0
            half_width = critical * sd / np.sqrt(count)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
            result[f"{metric}_ci95_low"] = mean - half_width
            result[f"{metric}_ci95_high"] = mean + half_width
            result[f"{metric}_worst"] = float(np.min(values))
        rows.append(result)
    return pd.DataFrame(rows)


def _build_integrity_dataset(
    seed: int, n: int = INTEGRITY_CLEAN_RECORDS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create clean, fully assessed synthetic records for integrity evaluation."""
    rng = _seeded_rng(seed)
    categories = tuple(config.RESOURCE_SPECS)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D")
    payments = pd.DataFrame(
        {
            "payment_id": [f"S{seed:02d}-P{i:05d}" for i in range(n)],
            "beneficiary_id": [f"S{seed:02d}-B{i:05d}" for i in range(n)],
            "camp_id": [f"S{seed:02d}-C{i % 50:03d}" for i in range(n)],
            "amount": np.maximum(
                20.0,
                rng.lognormal(mean=np.log(180.0), sigma=0.75, size=n),
            ),
            "damage_category": rng.choice(categories, size=n),
            "assessment_id": [f"S{seed:02d}-A{i:05d}" for i in range(n)],
            "date": dates,
        }
    )
    assessments = pd.DataFrame(
        {
            "assessment_id": payments["assessment_id"],
            "camp_id": payments["camp_id"],
            "damage_category": payments["damage_category"],
            "date": payments["date"],
            "risk_score": rng.uniform(0.1, 0.9, n),
        }
    )
    return payments, assessments


def _duplicate_pair_metrics(
    payments: pd.DataFrame, flagged_ids: set[str]
) -> dict[str, int | float]:
    """Score injected duplicate pairs as a unit, requiring both records flagged."""
    by_id = payments.set_index(payments["payment_id"].astype(str), drop=False)
    true_pairs: set[frozenset[str]] = set()
    for duplicate_id in payments.loc[
        payments["ground_truth_duplicate"], "payment_id"
    ].astype(str):
        original_id = duplicate_id.removesuffix("_dup")
        if original_id in by_id.index:
            true_pairs.add(frozenset((original_id, duplicate_id)))

    predicted_pairs: set[frozenset[str]] = set()
    for _, group in payments.groupby("beneficiary_id", sort=False):
        ordered = group.sort_values("date")
        rows = list(ordered.to_dict("records"))
        for previous, current in zip(rows, rows[1:]):
            previous_id = str(previous["payment_id"])
            current_id = str(current["payment_id"])
            delta = abs((pd.Timestamp(current["date"]) - pd.Timestamp(previous["date"])).days)
            if (
                previous_id in flagged_ids
                and current_id in flagged_ids
                and np.isclose(float(previous["amount"]), float(current["amount"]))
                and delta <= config.PAYMENT_DUPLICATE_WINDOW_DAYS
            ):
                predicted_pairs.add(frozenset((previous_id, current_id)))

    tp = len(true_pairs & predicted_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)
    return {
        "injected": len(true_pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def run_integrity_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the reproducible 30-seed anomaly and latency experiments."""
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    records: list[dict[str, Any]] = []
    amount_distribution_rows: list[dict[str, Any]] = []
    outlier_comparison_rows: list[dict[str, Any]] = []
    overlap_total = 0
    overlap_excess_total = 0
    clean_false_positives = 0
    clean_record_count = 0
    pair_totals = {"injected": 0, "tp": 0, "fp": 0, "fn": 0}
    rule_counts = {
        rule: {"injected": 0, "tp": 0, "fp": 0, "fn": 0}
        for rule in ANOMALY_TYPES
    }
    union_counts = {"injected": 0, "tp": 0, "fp": 0, "fn": 0}
    outlier_totals = {
        method: {"injected": 0, "tp": 0, "fp": 0, "fn": 0}
        for method in ("z_score", "modified_z_score")
    }

    for seed in SEEDS:
        clean, assessments = _build_integrity_dataset(seed)
        for category, group in clean.groupby("damage_category", sort=True):
            mean_amount = float(group["amount"].mean())
            std_amount = float(group["amount"].std(ddof=1))
            amount_distribution_rows.append(
                {
                    "seed": seed,
                    "damage_category": category,
                    "records": len(group),
                    "amount_mean": mean_amount,
                    "amount_sd": std_amount,
                    "coefficient_of_variation": std_amount / mean_amount
                    if mean_amount
                    else 0.0,
                }
            )
        payments = inject_anomalies(clean, rng=_seeded_rng(seed + 1000), rates=ANOMALY_RATES)
        payments = payments.reset_index(drop=True)
        flags = {
            "duplicate": set(detect_duplicate_payments(payments)[0]),
            "missing_assessment": set(
                detect_missing_assessment(payments, assessments)[0]
            ),
            "inflated_amount": set(detect_zscore_outliers(payments)[0]),
        }
        modified_flags = set(detect_modified_zscore_outliers(payments)[0])
        truth_by_rule = {
            rule: set(
                payments.loc[
                    payments[f"ground_truth_{rule}"], "payment_id"
                ].astype(str)
            )
            for rule in ANOMALY_TYPES
        }
        labels_per_record = payments[
            [f"ground_truth_{rule}" for rule in ANOMALY_TYPES]
        ].sum(axis=1)
        seed_overlap = int((labels_per_record > 1).sum())
        overlap_total += seed_overlap
        overlap_excess_total += int((labels_per_record - 1).clip(lower=0).sum())

        for rule in ANOMALY_TYPES:
            true_ids = truth_by_rule[rule]
            flagged_ids = flags[rule]
            counts = {
                "injected": len(true_ids),
                "tp": len(true_ids & flagged_ids),
                "fp": len(flagged_ids - true_ids),
                "fn": len(true_ids - flagged_ids),
            }
            for key, value in counts.items():
                rule_counts[rule][key] += value
            precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
            recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
            records.append(
                {
                    "seed": seed,
                    "rule": rule,
                    **counts,
                    "precision": precision,
                    "recall": recall,
                    "pair_injected": np.nan,
                    "pair_tp": np.nan,
                    "pair_fp": np.nan,
                    "pair_fn": np.nan,
                    "pair_precision": np.nan,
                    "pair_recall": np.nan,
                }
            )

        inflated_truth = truth_by_rule["inflated_amount"]
        for method, flagged in (
            ("z_score", flags["inflated_amount"]),
            ("modified_z_score", modified_flags),
        ):
            counts = {
                "injected": len(inflated_truth),
                "tp": len(inflated_truth & flagged),
                "fp": len(flagged - inflated_truth),
                "fn": len(inflated_truth - flagged),
            }
            for key, value in counts.items():
                outlier_totals[method][key] += value
            outlier_comparison_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    **counts,
                    "threshold": 3.0
                    if method == "z_score"
                    else MODIFIED_ZSCORE_THRESHOLD,
                    "precision": counts["tp"] / (counts["tp"] + counts["fp"])
                    if counts["tp"] + counts["fp"]
                    else 0.0,
                    "recall": counts["tp"] / (counts["tp"] + counts["fn"])
                    if counts["tp"] + counts["fn"]
                    else 0.0,
                }
            )

        pair = _duplicate_pair_metrics(payments, flags["duplicate"])
        for key in pair_totals:
            pair_totals[key] += int(pair[key])
        duplicate_row = records[-len(ANOMALY_TYPES)]
        duplicate_row.update(
            {
                "pair_injected": pair["injected"],
                "pair_tp": pair["tp"],
                "pair_fp": pair["fp"],
                "pair_fn": pair["fn"],
                "pair_precision": pair["precision"],
                "pair_recall": pair["recall"],
            }
        )

        true_union = set().union(*truth_by_rule.values())
        flagged_union = set().union(*flags.values())
        union = {
            "injected": len(true_union),
            "tp": len(true_union & flagged_union),
            "fp": len(flagged_union - true_union),
            "fn": len(true_union - flagged_union),
        }
        for key, value in union.items():
            union_counts[key] += value

        duplicate_originals = {
            payment_id.removesuffix("_dup")
            for payment_id in truth_by_rule["duplicate"]
        }
        excluded = set().union(*truth_by_rule.values()) | duplicate_originals
        clean_ids = set(payments["payment_id"].astype(str)) - excluded
        clean_flagged = clean_ids & flagged_union
        clean_false_positives += len(clean_flagged)
        clean_record_count += len(clean_ids)
        records.append(
            {
                "seed": seed,
                "rule": "overall_unique_records",
                **union,
                "precision": union["tp"] / (union["tp"] + union["fp"])
                if union["tp"] + union["fp"]
                else 0.0,
                "recall": union["tp"] / (union["tp"] + union["fn"])
                if union["tp"] + union["fn"]
                else 0.0,
                "pair_injected": np.nan,
                "pair_tp": np.nan,
                "pair_fp": np.nan,
                "pair_fn": np.nan,
                "pair_precision": np.nan,
                "pair_recall": np.nan,
                "clean_record_false_positive_rate": len(clean_flagged) / len(clean_ids)
                if clean_ids
                else 0.0,
                "multi_label_records": seed_overlap,
                "overlap_excess_labels": int(
                    (labels_per_record - 1).clip(lower=0).sum()
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    for rule in ANOMALY_TYPES:
        counts = rule_counts[rule]
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        row: dict[str, Any] = {
            "rule": rule,
            **counts,
            "precision": precision,
            "recall": recall,
            "pair_injected": np.nan,
            "pair_tp": np.nan,
            "pair_fp": np.nan,
            "pair_fn": np.nan,
            "pair_precision": np.nan,
            "pair_recall": np.nan,
        }
        if rule == "duplicate":
            row.update(
                {
                    "pair_injected": pair_totals["injected"],
                    "pair_tp": pair_totals["tp"],
                    "pair_fp": pair_totals["fp"],
                    "pair_fn": pair_totals["fn"],
                    "pair_precision": pair_totals["tp"] / (pair_totals["tp"] + pair_totals["fp"])
                    if pair_totals["tp"] + pair_totals["fp"]
                    else 0.0,
                    "pair_recall": pair_totals["tp"] / (pair_totals["tp"] + pair_totals["fn"])
                    if pair_totals["tp"] + pair_totals["fn"]
                    else 0.0,
                }
            )
        rows.append(row)

    overall_precision = union_counts["tp"] / (union_counts["tp"] + union_counts["fp"]) if union_counts["tp"] + union_counts["fp"] else 0.0
    overall_recall = union_counts["tp"] / (union_counts["tp"] + union_counts["fn"]) if union_counts["tp"] + union_counts["fn"] else 0.0
    rows.append(
        {
            "rule": "overall_unique_records",
            **union_counts,
            "precision": overall_precision,
            "recall": overall_recall,
            "pair_injected": np.nan,
            "pair_tp": np.nan,
            "pair_fp": np.nan,
            "pair_fn": np.nan,
            "pair_precision": np.nan,
            "pair_recall": np.nan,
            "clean_record_false_positive_rate": clean_false_positives / clean_record_count
            if clean_record_count
            else 0.0,
            "multi_label_records": overlap_total,
            "overlap_excess_labels": overlap_excess_total,
        }
    )
    integrity_df = pd.DataFrame(rows)
    integrity_df["clean_record_false_positive_rate"] = integrity_df.get(
        "clean_record_false_positive_rate", pd.Series(dtype=float)
    )
    integrity_df["multi_label_records"] = integrity_df.get(
        "multi_label_records", pd.Series(dtype=float)
    )
    integrity_df.to_csv(RESULTS_DIR / "integrity_by_rule.csv", index=False)
    pd.DataFrame(records).to_csv(RESULTS_DIR / "integrity_by_seed.csv", index=False)
    amount_distribution = pd.DataFrame(amount_distribution_rows)
    amount_distribution_summary = (
        amount_distribution.groupby("damage_category", as_index=False)
        .agg(
            records_mean=("records", "mean"),
            amount_mean=("amount_mean", "mean"),
            amount_sd_mean=("amount_sd", "mean"),
            cv_mean=("coefficient_of_variation", "mean"),
            cv_sd=("coefficient_of_variation", "std"),
        )
    )
    amount_distribution_summary.to_csv(
        RESULTS_DIR / "integrity_amount_distribution.csv", index=False
    )
    outlier_summary_rows: list[dict[str, Any]] = []
    for method, counts in outlier_totals.items():
        outlier_summary_rows.append(
            {
                "method": method,
                **counts,
                "threshold": 3.0
                if method == "z_score"
                else MODIFIED_ZSCORE_THRESHOLD,
                "precision": counts["tp"] / (counts["tp"] + counts["fp"])
                if counts["tp"] + counts["fp"]
                else 0.0,
                "recall": counts["tp"] / (counts["tp"] + counts["fn"])
                if counts["tp"] + counts["fn"]
                else 0.0,
            }
        )
    outlier_comparison = pd.DataFrame(outlier_summary_rows)
    outlier_comparison.to_csv(
        RESULTS_DIR / "integrity_outlier_comparison.csv", index=False
    )
    metadata = pd.DataFrame(
        [
            ("seed_count", len(SEEDS)),
            ("clean_records_per_seed", INTEGRITY_CLEAN_RECORDS),
            ("duplicate_injection_rate", ANOMALY_RATES["duplicate"]),
            ("missing_assessment_injection_rate", ANOMALY_RATES["missing_assessment"]),
            ("inflated_amount_injection_rate", ANOMALY_RATES["inflated_amount"]),
            ("inflation_multiplier", INFLATION_MULTIPLIER),
            ("inflation_multiplier_min", INFLATION_MULTIPLIER),
            ("inflation_multiplier_max", INFLATION_MULTIPLIER),
            ("anomaly_rules", "|".join(ANOMALY_TYPES)),
            ("batch_zscore_threshold", 3.0),
            ("modified_zscore_threshold", MODIFIED_ZSCORE_THRESHOLD),
            ("duplicate_window_days", config.PAYMENT_DUPLICATE_WINDOW_DAYS),
        ],
        columns=["parameter", "value"],
    )
    metadata.to_csv(RESULTS_DIR / "integrity_metadata.csv", index=False)
    pd.DataFrame(outlier_comparison_rows).to_csv(
        RESULTS_DIR / "integrity_outlier_comparison_by_seed.csv", index=False
    )
    return integrity_df, _run_latency_experiments()


def _latency_record(n_existing: int, mode: str, values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "existing_records": n_existing,
        "mode": mode,
        "inserts_timed": len(values),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def _run_latency_experiments() -> pd.DataFrame:
    """Benchmark incremental inserts against explicitly labelled full rescans."""
    rng = _seeded_rng(config.RANDOM_SEED)
    categories = tuple(config.RESOURCE_SPECS)
    n_max = 20_000
    existing = pd.DataFrame(
        {
            "payment_id": [f"L-P{i:05d}" for i in range(n_max)],
            "beneficiary_id": [f"L-B{i:05d}" for i in range(n_max)],
            "camp_id": [f"L-C{i % 50:03d}" for i in range(n_max)],
            "amount": np.maximum(20.0, rng.lognormal(np.log(180.0), 0.75, n_max)),
            "damage_category": rng.choice(categories, size=n_max),
            "assessment_id": [f"L-A{i:05d}" for i in range(n_max)],
            "date": pd.Timestamp("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 60, n_max), unit="D"),
        }
    )
    assessment_ids = set(existing["assessment_id"].astype(str))
    rows: list[dict[str, Any]] = []
    incremental_samples = 100
    rescan_samples = 5

    for n_existing in (1000, 5000, 20_000):
        checker = IncrementalIntegrityChecker(assessment_ids)
        for payment in existing.iloc[:n_existing].to_dict("records"):
            checker.add_existing(payment)
        insert_times: list[float] = []
        test_rows = existing.iloc[:incremental_samples]
        for offset, payment in enumerate(test_rows.to_dict("records")):
            payment["payment_id"] = f"NEW-{n_existing}-{offset}"
            payment["beneficiary_id"] = f"NEW-B-{n_existing}-{offset}"
            payment["assessment_id"] = f"L-A{offset:05d}"
            start = time.perf_counter()
            checker.check_and_add(payment)
            insert_times.append((time.perf_counter() - start) * 1000.0)
        rows.append(_latency_record(n_existing, "incremental_on_insert", insert_times))

        batch_times: list[float] = []
        dataset = existing.iloc[:n_existing].copy()
        assessments = pd.DataFrame({"assessment_id": sorted(assessment_ids)})
        for _ in range(rescan_samples):
            start = time.perf_counter()
            detect_duplicate_payments(dataset)
            detect_missing_assessment(dataset, assessments)
            detect_zscore_outliers(dataset)
            batch_times.append((time.perf_counter() - start) * 1000.0)
        rows.append(_latency_record(n_existing, "full_rescan", batch_times))

    latency_df = pd.DataFrame(rows)
    latency_df.to_csv(RESULTS_DIR / "integrity_latency.csv", index=False)
    return latency_df


def run_experiments() -> dict[str, pd.DataFrame]:
    """Run allocation and sensitivity sweeps, aggregating resources within seeds."""
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    resource_rows: list[dict[str, Any]] = []
    sensitivity_resource_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        camps = generate_camps(CAMPS_PER_SEED, _seeded_rng(seed))
        for ratio in SUPPLY_LEVELS:
            for resource in config.RESOURCE_SPECS:
                resource_demand = float(demand(camps, resource).sum())
                metrics = _record_metrics(camps, resource, ratio * resource_demand)
                for method, values in metrics.items():
                    resource_rows.append(
                        {
                            "seed": seed,
                            "supply_ratio": ratio,
                            "resource": resource,
                            "method": method,
                            "vwc": float(values["vwc"]),
                            "min_coverage": float(values["min_coverage"]),
                            "base_coverage": float(values["base_coverage"]),
                            "zero_coverage_camps": int(values["zero_coverage_camps"]),
                        }
                    )
        for resource in config.RESOURCE_SPECS:
            for sensitivity in _sensitivity_by_resource(camps, resource):
                sensitivity_resource_rows.append(
                    {
                        "seed": seed,
                        "resource": resource,
                        **sensitivity,
                    }
                )

    resource_frame = pd.DataFrame(resource_rows)
    seed_frame = (
        resource_frame.groupby(["seed", "supply_ratio", "method"], as_index=False)[
            ["vwc", "min_coverage", "base_coverage", "zero_coverage_camps"]
        ]
        .mean()
        .sort_values(["supply_ratio", "method", "seed"])
        .reset_index(drop=True)
    )
    summary = _seed_summary(
        seed_frame,
        ["supply_ratio", "method"],
        ["vwc", "min_coverage", "base_coverage", "zero_coverage_camps"],
    ).sort_values(["supply_ratio", "method"])

    sensitivity_resource_frame = pd.DataFrame(sensitivity_resource_rows)
    sensitivity_seed_frame = (
        sensitivity_resource_frame.groupby(["seed", "weight", "change"], as_index=False)[
            "spearman"
        ]
        .mean()
        .sort_values(["weight", "change", "seed"])
        .reset_index(drop=True)
    )
    sensitivity_summary = _seed_summary(
        sensitivity_seed_frame, ["weight", "change"], ["spearman"]
    )
    sensitivity_summary["spearman_min"] = sensitivity_seed_frame.groupby(
        ["weight", "change"], sort=True
    )["spearman"].min().to_numpy()

    per_camp_seed = generate_camps(CAMPS_PER_SEED, _seeded_rng(config.RANDOM_SEED))
    resource = "food"
    supply = SUPPLY_LEVELS[0] * float(demand(per_camp_seed, resource).sum())
    proposed = allocate_equity(per_camp_seed, resource, supply)
    baseline = allocate_baseline(per_camp_seed, resource, supply)
    per_camp = proposed[["camp_id", "P", "D", "rank"]].rename(
        columns={"P": "priority", "D": "demand"}
    )
    per_camp["alloc_equity"] = proposed["x"].to_numpy()
    per_camp["alloc_baseline"] = baseline.set_index("camp_id").loc[
        per_camp["camp_id"], "x"
    ].to_numpy()
    per_camp = per_camp.sort_values("rank").reset_index(drop=True)

    worst_rows: list[dict[str, Any]] = []
    for ratio, group in seed_frame.groupby("supply_ratio", sort=True):
        for method, method_group in group.groupby("method", sort=True):
            worst = method_group.loc[method_group["min_coverage"].idxmin()]
            worst_rows.append(
                {
                    "supply_ratio": ratio,
                    "method": method,
                    "worst_seed": int(worst["seed"]),
                    "worst_min_coverage": float(worst["min_coverage"]),
                    "vwc_at_worst_min_coverage": float(worst["vwc"]),
                    "base_coverage_at_worst_min_coverage": float(worst["base_coverage"]),
                }
            )
    worst_frame = pd.DataFrame(worst_rows)
    experiment_metadata_rows: list[tuple[str, Any]] = [
        ("seed_count", len(SEEDS)),
        ("first_seed", SEEDS[0]),
        ("last_seed", SEEDS[-1]),
        ("camps_per_seed", CAMPS_PER_SEED),
        ("resource_count", len(config.RESOURCE_SPECS)),
        ("resources", "|".join(config.RESOURCE_SPECS)),
        ("supply_levels", "|".join(str(value) for value in SUPPLY_LEVELS)),
        ("allocation_floor", config.PHI),
        ("planning_horizon_days", config.HORIZON_DAYS),
        ("random_seed_for_per_camp_output", config.RANDOM_SEED),
    ]
    experiment_metadata_rows.extend(
        (f"group_weight_{name}", weight)
        for name, weight in config.GROUP_WEIGHTS.items()
    )
    experiment_metadata_rows.extend(
        (f"resource_{resource}_nu0", spec.nu0)
        for resource, spec in config.RESOURCE_SPECS.items()
    )
    for resource, spec in config.RESOURCE_SPECS.items():
        experiment_metadata_rows.extend(
            (f"resource_{resource}_nu_{group}", value)
            for group, value in spec.nu.items()
        )
        experiment_metadata_rows.extend(
            (f"resource_{resource}_rho_{group}", value)
            for group, value in spec.rho.items()
        )
    experiment_metadata = pd.DataFrame(
        experiment_metadata_rows, columns=["parameter", "value"]
    )

    summary.to_csv(RESULTS_DIR / "summary_by_supply.csv", index=False)
    experiment_metadata.to_csv(RESULTS_DIR / "experiment_metadata.csv", index=False)
    seed_frame.to_csv(RESULTS_DIR / "allocation_by_seed.csv", index=False)
    resource_frame.to_csv(RESULTS_DIR / "allocation_by_seed_resource.csv", index=False)
    sensitivity_summary.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    sensitivity_seed_frame.to_csv(RESULTS_DIR / "sensitivity_by_seed.csv", index=False)
    sensitivity_resource_frame.to_csv(
        RESULTS_DIR / "sensitivity_by_seed_resource.csv", index=False
    )
    per_camp.to_csv(RESULTS_DIR / "per_camp_40.csv", index=False)
    worst_frame.to_csv(RESULTS_DIR / "allocation_worst_run.csv", index=False)

    print("\nAllocation summary (resource means within each seed; mean, SD, t-based 95% CI):")
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print("\nAllocation experiment parameters:")
    print(experiment_metadata.to_markdown(index=False))
    print("\nSensitivity summary (resource means within each seed; mean, SD, t-based 95% CI):")
    print(sensitivity_summary.to_markdown(index=False, floatfmt=".4f"))
    print("\nWorst single allocation run (lowest resource-averaged minimum coverage):")
    print(worst_frame.to_markdown(index=False, floatfmt=".4f"))
    print("\nproposed_no_floor minimum coverage by seed at 40% supply (first five seeds):")
    no_floor_first_five = seed_frame[
        (seed_frame["supply_ratio"] == SUPPLY_LEVELS[0])
        & (seed_frame["method"] == "proposed_no_floor")
        & (seed_frame["seed"] <= 5)
    ][["seed", "min_coverage", "zero_coverage_camps"]]
    print(no_floor_first_five.to_markdown(index=False, floatfmt=".4f"))
    print(
        "\nNo-floor check: allocate_equity receives floor=0.0; its implementation uses "
        "config.PHI only when floor is None, so zero is not replaced."
    )
    print(
        "The min_coverage metric omits zero-coverage camps; "
        "zero_coverage_camps is reported separately."
    )
    return {
        "summary_by_supply": summary,
        "allocation_by_seed": seed_frame,
        "allocation_by_seed_resource": resource_frame,
        "sensitivity": sensitivity_summary,
        "sensitivity_by_seed": sensitivity_seed_frame,
        "sensitivity_by_seed_resource": sensitivity_resource_frame,
        "per_camp_40": per_camp,
        "allocation_worst_run": worst_frame,
    }


def main() -> None:
    """Run allocation, integrity, latency, and figure generation for sim.run_all."""
    run_experiments()
    integrity, latency = run_integrity_experiments()
    print("\nIntegrity summary (totals across seeds; duplicate record and pair scoring):")
    print(integrity.to_markdown(index=False, floatfmt=".4f"))
    overall = integrity.loc[integrity["rule"] == "overall_unique_records"].iloc[0]
    per_rule_injected = int(
        integrity.loc[integrity["rule"].isin(ANOMALY_TYPES), "injected"].sum()
    )
    print(
        f"\nUnique-record reconciliation: {per_rule_injected} per-rule label occurrences; "
        f"{int(overall['injected'])} unique injected records; "
        f"{int(overall['multi_label_records'])} records carry multiple labels; "
        f"{int(overall['overlap_excess_labels'])} excess label occurrences."
    )
    print("\nInflated-amount outlier comparison:")
    print(pd.read_csv(RESULTS_DIR / "integrity_outlier_comparison.csv").to_markdown(index=False, floatfmt=".4f"))
    print("\nNatural amount variation by damage category:")
    print(pd.read_csv(RESULTS_DIR / "integrity_amount_distribution.csv").to_markdown(index=False, floatfmt=".4f"))
    print("\nIntegrity experiment parameters:")
    print(pd.read_csv(RESULTS_DIR / "integrity_metadata.csv").to_markdown(index=False))
    print("\nIntegrity latency (single-insert checks and full-rescan batch checks):")
    print(latency.to_markdown(index=False, floatfmt=".4f"))
    print(
        "\nLatency scope: incremental timings include in-memory check and index update only, "
        "not database or API time; full_rescan reruns the three batch detectors."
    )
    print(
        f"\nInflation multiplier range from integrity.py: "
        f"{INFLATION_MULTIPLIER}x to {INFLATION_MULTIPLIER}x"
    )
    print(f"Rule list from integrity.py: {', '.join(ANOMALY_TYPES)}")
    from sim.make_figures import main as make_figures

    make_figures()
    generated = sorted(
        str(path.relative_to(ROOT_DIR)).replace("\\", "/")
        for path in RESULTS_DIR.rglob("*")
        if path.is_file()
    )
    print("\nGenerated files:")
    print("\n".join(generated))
