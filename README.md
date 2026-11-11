# Relief Framework Prototype

This prototype implements the simulated disaster-relief allocation and integrity-verification workflow described in the paper. All data are synthetic and generated from a single seeded NumPy generator.

## Setup

```bash
cd C:\Users\SUSHMITHA\Sentinel
pip install -r requirements.txt
pytest
python experiments.py
```

## Reproducibility

The pipeline is seeded from `relief_framework.config.RANDOM_SEED`, so the same seed reproduces identical tables and figures.

## Ranking determinism

Allocation ranks are deterministic. When two camps have equal priority score, the rank is broken by higher vulnerable headcount first and then by camp ID in ascending order. Camps with zero shortfall are excluded from the positive-demand ranking denominator and kept at rank 0.

## Outputs

Figures and CSV summaries are written to the `relief_framework/outputs/` directory.
