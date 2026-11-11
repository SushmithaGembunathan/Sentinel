# Reproducible experiment outputs

Run the complete experiment and figure pipeline from the repository root:

```powershell
python -m sim.run_all
```

The allocation experiment uses seeds 1-30, 50 camps per seed, all configured
resources, and the configured 40%, 60%, and 80% supply levels. Resource metrics
are averaged within each seed before reporting the across-seed mean, sample
standard deviation, and two-sided 95% Student-t confidence interval. Worst
resource-averaged minimum coverage is also recorded separately.

The integrity experiment creates 1,000 clean assessed records per seed, then
uses the anomaly rates and inflation multiplier defined in
`relief_framework/experiments.py` and `relief_framework/integrity.py`. The
overall row is a union of unique records per seed, while duplicate-pair
statistics require both members of an injected pair to be flagged.

Latency entries distinguish `incremental_on_insert` from `full_rescan`.
Incremental latency times the in-memory detector check and state update for one
insert; it excludes database and API work. Full-rescan latency reruns all three
batch detectors over the indicated existing-record count.

## Generated files

- `allocation_by_seed.csv`
- `allocation_by_seed_resource.csv`
- `allocation_worst_run.csv`
- `experiment_metadata.csv`
- `integrity_amount_distribution.csv`
- `integrity_by_rule.csv`
- `integrity_by_seed.csv`
- `integrity_latency.csv`
- `integrity_metadata.csv`
- `integrity_outlier_comparison.csv`
- `integrity_outlier_comparison_by_seed.csv`
- `per_camp_40.csv`
- `sensitivity.csv`
- `sensitivity_by_seed.csv`
- `sensitivity_by_seed_resource.csv`
- `summary_by_supply.csv`
- `figures/min_coverage_by_supply.png`
- `figures/per_camp_40.png`
- `figures/sensitivity.png`
- `figures/vwc_by_supply.png`
