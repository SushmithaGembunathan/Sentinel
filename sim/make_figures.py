from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return pd.read_csv(path)


def make_vwc_figure() -> None:
    summary = _read_csv("summary_by_supply.csv")
    baseline = summary[summary["method"] == "baseline"].sort_values("supply_ratio")
    proposed = summary[summary["method"] == "proposed"].sort_values("supply_ratio")

    fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)
    x = [0.4, 0.6, 0.8]
    for frame, marker, label in (
        (baseline, "o-", "Baseline"),
        (proposed, "s-", "Proposed"),
    ):
        lower = frame["vwc_mean"].to_numpy() - frame["vwc_ci95_low"].to_numpy()
        upper = frame["vwc_ci95_high"].to_numpy() - frame["vwc_mean"].to_numpy()
        ax.errorbar(
            x,
            frame["vwc_mean"].to_numpy(),
            yerr=[lower, upper],
            fmt=marker,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0%}" for v in x])
    ax.set_xlabel("Supply level")
    ax.set_ylabel("VWC")
    ax.legend(frameon=False)
    ax.set_title("VWC by supply level")
    fig.tight_layout()
    fig.savefig(FIGURES / "vwc_by_supply.png", dpi=300)
    plt.close(fig)


def make_min_coverage_figure() -> None:
    summary = _read_csv("summary_by_supply.csv")
    baseline = summary[summary["method"] == "baseline"].sort_values("supply_ratio")
    proposed = summary[summary["method"] == "proposed"].sort_values("supply_ratio")

    fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)
    x = [0.4, 0.6, 0.8]
    for frame, marker, label in (
        (baseline, "o-", "Baseline"),
        (proposed, "s-", "Proposed"),
    ):
        lower = (
            frame["min_coverage_mean"].to_numpy()
            - frame["min_coverage_ci95_low"].to_numpy()
        )
        upper = (
            frame["min_coverage_ci95_high"].to_numpy()
            - frame["min_coverage_mean"].to_numpy()
        )
        ax.errorbar(
            x,
            frame["min_coverage_mean"].to_numpy(),
            yerr=[lower, upper],
            fmt=marker,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0%}" for v in x])
    ax.set_xlabel("Supply level")
    ax.set_ylabel("Min coverage")
    ax.legend(frameon=False)
    ax.set_title("Minimum coverage by supply level")
    fig.tight_layout()
    fig.savefig(FIGURES / "min_coverage_by_supply.png", dpi=300)
    plt.close(fig)


def make_per_camp_figure() -> None:
    per_camp = _read_csv("per_camp_40.csv")
    if per_camp.empty:
        raise ValueError("per_camp_40.csv is empty")

    per_camp = per_camp.sort_values("rank").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)
    camp_ids = per_camp["camp_id"].tolist()
    x = range(len(camp_ids))
    ax.bar(x, per_camp["alloc_equity"].to_numpy(), label="Proposed")
    ax.plot(x, per_camp["alloc_baseline"].to_numpy(), color="black", marker="o", linewidth=1.5, label="Baseline")
    ax.set_xticks(list(x))
    ax.set_xticklabels(camp_ids, rotation=45, ha="right")
    ax.set_xlabel("Camp")
    ax.set_ylabel("Allocation")
    ax.legend(frameon=False)
    ax.set_title("Per-camp coverage at 40% supply (seed 42)")
    fig.tight_layout()
    fig.savefig(FIGURES / "per_camp_40.png", dpi=300)
    plt.close(fig)


def make_sensitivity_figure() -> None:
    sensitivity = _read_csv("sensitivity.csv")
    if sensitivity.empty:
        raise ValueError("sensitivity.csv is empty")

    labels = [
        f"{row.weight} {row.change}"
        for row in sensitivity.sort_values(["weight", "change"]).itertuples()
    ]
    values = sensitivity.sort_values(["weight", "change"])["spearman_mean"].to_numpy()
    lower = values - sensitivity.sort_values(["weight", "change"])[
        "spearman_ci95_low"
    ].to_numpy()
    upper = sensitivity.sort_values(["weight", "change"])[
        "spearman_ci95_high"
    ].to_numpy() - values

    fig, ax = plt.subplots(figsize=(3.45, 2.6), dpi=300)
    ax.bar(labels, values, yerr=[lower, upper], capsize=2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Sensitivity to weight perturbation")
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "sensitivity.png", dpi=300)
    plt.close(fig)


def main() -> None:
    make_vwc_figure()
    make_min_coverage_figure()
    make_per_camp_figure()
    make_sensitivity_figure()


if __name__ == "__main__":
    main()
