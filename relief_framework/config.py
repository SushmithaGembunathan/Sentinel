from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

import numpy as np

GROUP_NAMES = ("base", "elderly", "disabled", "maternal", "infant")

# DESIGN PARAMETER (not official): group weights used in the equity score.
GROUP_WEIGHTS: Dict[str, float] = {
    "base": 1.0,
    "elderly": 1.5,
    "disabled": 2.0,
    "maternal": 2.0,
    "infant": 1.5,
}

PHI = 0.20
HORIZON_DAYS = 3
RANDOM_SEED = 42
RNG = np.random.default_rng(RANDOM_SEED)


@dataclass(frozen=True)
class ResourceSpec:
    """Configuration for one resource type."""

    name: str
    nu0: float
    nu: Dict[str, float]
    rho: Dict[str, float]


# DESIGN PARAMETER (not official): daily normative quantities and relevance multipliers.
# Each resource is configured once and reused in pricing and allocation logic.
RESOURCE_SPECS: Dict[str, ResourceSpec] = {
    "food": ResourceSpec(
        name="food",
        nu0=1.0,
        nu={"base": 0.0, "elderly": 0.05, "disabled": 0.08, "maternal": 0.10, "infant": 0.12},
        rho={"base": 1.0, "elderly": 1.0, "disabled": 1.2, "maternal": 1.1, "infant": 1.0},
    ),
    "water": ResourceSpec(
        name="water",
        nu0=2.0,
        nu={"base": 0.0, "elderly": 0.10, "disabled": 0.12, "maternal": 0.12, "infant": 0.15},
        rho={"base": 1.0, "elderly": 1.0, "disabled": 1.2, "maternal": 1.1, "infant": 1.0},
    ),
    "medical": ResourceSpec(
        name="medical",
        nu0=0.5,
        nu={"base": 0.0, "elderly": 0.04, "disabled": 0.08, "maternal": 0.06, "infant": 0.07},
        rho={"base": 1.0, "elderly": 1.5, "disabled": 2.0, "maternal": 1.0, "infant": 1.0},
    ),
    "infant_supplies": ResourceSpec(
        name="infant_supplies",
        nu0=0.25,
        nu={"base": 0.0, "elderly": 0.02, "disabled": 0.03, "maternal": 0.04, "infant": 0.22},
        rho={"base": 0.2, "elderly": 0.2, "disabled": 0.2, "maternal": 0.2, "infant": 3.0},
    ),
}

PAYMENT_DUPLICATE_WINDOW_DAYS = 7
SUPPLY_RATIOS = (0.40, 0.60, 0.80)

DEMO_USERS = {
    "coordinator1": {
        "password_hash": hashlib.pbkdf2_hmac("sha256", b"coordinator1", b"coordinator1salt", 200000).hex(),
        "salt": "coordinator1salt",
        "role": "coordinator",
        "camp_id": "C001",
    },
    "officer1": {
        "password_hash": hashlib.pbkdf2_hmac("sha256", b"officer1", b"officer1salt", 200000).hex(),
        "salt": "officer1salt",
        "role": "officer",
        "camp_id": None,
    },
    "auditor1": {
        "password_hash": hashlib.pbkdf2_hmac("sha256", b"auditor1", b"auditor1salt", 200000).hex(),
        "salt": "auditor1salt",
        "role": "auditor",
        "camp_id": None,
    },
}
