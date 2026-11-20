"""Shared telemetry generative process.

Used both to seed the demo business database and to synthesise labelled
training data for the pairwise model, so that the "given" pre-fitted
model and the demo archives live in the same world.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: plausible per-archetype parameter ranges
ARCHETYPES = {
    "pump": dict(vib=(2.5, 4.5), temp=(55.0, 75.0), cur=(10.0, 20.0), press=(3.0, 5.0)),
    "compressor": dict(vib=(4.0, 7.0), temp=(70.0, 95.0), cur=(25.0, 40.0), press=(6.0, 9.0)),
    "valve": dict(vib=(0.5, 1.5), temp=(20.0, 40.0), cur=(1.0, 3.0), press=(2.0, 4.0)),
    "motor": dict(vib=(1.5, 3.0), temp=(45.0, 65.0), cur=(8.0, 15.0), press=(0.8, 1.5)),
    "fan": dict(vib=(1.0, 2.2), temp=(30.0, 50.0), cur=(4.0, 8.0), press=(0.4, 0.9)),
    "blower": dict(vib=(2.0, 3.5), temp=(40.0, 60.0), cur=(9.0, 14.0), press=(1.0, 2.0)),
    "generator": dict(vib=(3.0, 5.5), temp=(60.0, 85.0), cur=(30.0, 50.0), press=(4.0, 6.0)),
    "hydro": dict(vib=(0.8, 1.8), temp=(25.0, 45.0), cur=(2.0, 5.0), press=(2.5, 4.5)),
}

VENDORS = {
    "pump": "FlowServe",
    "compressor": "AtlasCopco",
    "valve": "Emerson",
    "motor": "Siemens",
    "fan": "ebm-papst",
    "blower": "Howden",
    "generator": "Caterpillar",
    "hydro": "KSB",
}


def make_entity_params(rng: np.random.Generator, kind: str) -> dict:
    a = ARCHETYPES[kind]
    return {
        "kind": kind,
        "vib_mean": float(rng.uniform(*a["vib"])),
        "vib_std": float(rng.uniform(0.15, 0.5)),
        "temp_mean": float(rng.uniform(*a["temp"])),
        "temp_trend": float(rng.uniform(-0.02, 0.02)),
        "cur_mean": float(rng.uniform(*a["cur"])),
        "cur_amp": float(rng.uniform(0.10, 0.40)),
        "press_mean": float(rng.uniform(*a["press"])),
        "press_std": float(rng.uniform(0.05, 0.20)),
        "season_period": int(rng.integers(12, 36)),
        "ar_coef": float(rng.uniform(0.30, 0.80)),
    }


def jitter_params(rng: np.random.Generator, params: dict, scale: float = 0.05) -> dict:
    """Per-archive parameter jitter: same physical device, slightly
    different operating point / sensor calibration."""
    p = dict(params)
    for key in ("vib_mean", "temp_mean", "cur_mean", "press_mean"):
        p[key] = float(params[key] * (1.0 + rng.uniform(-scale, scale)))
    p["season_period"] = params["season_period"]
    return p


def sample_telemetry(
    rng: np.random.Generator,
    params: dict,
    archive_id: int,
    n_points: int,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    t = np.arange(n_points, dtype=float)
    phase = float(rng.uniform(0, 2 * np.pi))
    period = max(4, int(params["season_period"]))

    eps = rng.normal(0.0, params["vib_std"], n_points)
    ar = np.zeros(n_points)
    for k in range(1, n_points):
        ar[k] = params["ar_coef"] * ar[k - 1] + eps[k]
    vibration = params["vib_mean"] + ar + 0.3 * np.sin(2 * np.pi * t / period + phase)

    temperature = (
        params["temp_mean"]
        + params["temp_trend"] * t
        + 1.5 * np.sin(2 * np.pi * t / (period * 2) + phase)
        + rng.normal(0.0, 0.8, n_points)
    )

    current = params["cur_mean"] * (
        1.0 + params["cur_amp"] * np.sin(2 * np.pi * t / period + 2 * phase)
    ) + rng.normal(0.0, 0.05 * params["cur_mean"], n_points)

    pressure = (
        params["press_mean"]
        + 0.1 * np.cumsum(rng.normal(0.0, params["press_std"], n_points))
        + rng.normal(0.0, params["press_std"], n_points)
    )

    ts = pd.date_range(start=pd.Timestamp(start), periods=n_points, freq="D").strftime(
        "%Y-%m-%d"
    )
    return pd.DataFrame(
        {
            "archive_id": int(archive_id),
            "ts": ts,
            "vibration": vibration,
            "temperature": temperature,
            "current": current,
            "pressure": pressure,
        }
    )
