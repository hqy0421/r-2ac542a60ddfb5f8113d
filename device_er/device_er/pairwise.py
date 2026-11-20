"""Pairwise feature construction and match-probability scoring.

The pair vector consumed by the pre-fitted model is

    [ |f_i - f_j| / scale_f  for each selected tsfresh feature ]
    + [ attribute similarities ]

Attribute similarities use normalized text; a similarity is NaN when
either side lacks the attribute and is imputed with 0.5 ("unknown"),
while explicit both-present indicators keep the information that the
comparison was possible at all.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .errors import ModelError
from .master import is_null

UNKNOWN_SIMILARITY = 0.5

ATTR_SIM_NAMES = [
    "serial_eq",
    "asset_tag_eq",
    "vendor_eq",
    "model_token_jaccard",
    "location_eq",
    "install_date_eq",
    "firmware_eq",
    "warranty_eq",
    "serial_both_present",
    "asset_tag_both_present",
]


def _norm_text(value):
    if is_null(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return text or None


def _tokens(value):
    text = _norm_text(value)
    if text is None:
        return None
    return set(re.findall(r"[a-z0-9]+", text))


def _eq(a, b):
    na, nb = _norm_text(a), _norm_text(b)
    if na is None or nb is None:
        return np.nan
    return 1.0 if na == nb else 0.0


def _jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return np.nan
    return len(ta & tb) / len(ta | tb)


def _num_eq(a, b):
    if is_null(a) or is_null(b):
        return np.nan
    try:
        return 1.0 if float(a) == float(b) else 0.0
    except (TypeError, ValueError):
        return np.nan


def _both_present(a, b):
    return 1.0 if (not is_null(a) and not is_null(b)) else 0.0


def attribute_similarities(row_a, row_b) -> np.ndarray:
    """Fixed-order attribute similarity vector between two archive rows."""
    return np.array(
        [
            _eq(row_a["serial_no"], row_b["serial_no"]),
            _eq(row_a["asset_tag"], row_b["asset_tag"]),
            _eq(row_a["vendor"], row_b["vendor"]),
            _jaccard(row_a["model"], row_b["model"]),
            _eq(row_a["location"], row_b["location"]),
            _eq(row_a["install_date"], row_b["install_date"]),
            _eq(row_a["firmware"], row_b["firmware"]),
            _num_eq(row_a["warranty_months"], row_b["warranty_months"]),
            _both_present(row_a["serial_no"], row_b["serial_no"]),
            _both_present(row_a["asset_tag"], row_b["asset_tag"]),
        ],
        dtype=float,
    )


def build_pair_matrix(archives: pd.DataFrame, features: pd.DataFrame, model):
    """Assemble (pair list, design matrix) for the given pre-fitted model.

    Raises :class:`ModelError` when the extracted feature matrix does
    not provide the columns the model was fitted on.
    """
    selected = list(model.selected_features)
    missing = [c for c in selected if c not in features.columns]
    if missing:
        raise ModelError(
            "extracted features do not match the pre-fitted model",
            detail={"missing_features": missing[:10], "n_missing": len(missing)},
        )

    ids = [int(a) for a in features.index]
    feat = features[selected].to_numpy(dtype=float, copy=True)
    # model-side imputation of NaN feature cells (e.g. archives without
    # telemetry or degenerate series) with the training medians
    nan_mask = np.isnan(feat)
    if nan_mask.any():
        feat[nan_mask] = np.take(model.feature_impute, np.where(nan_mask)[1])

    by_id = archives.set_index("archive_id")
    attr_rows = {}
    for archive_id in ids:
        if archive_id not in by_id.index:
            raise ModelError(f"archive {archive_id} missing from archives table")
        attr_rows[archive_id] = by_id.loc[archive_id]

    pairs, vectors = [], []
    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            i, j = ids[a_idx], ids[b_idx]
            diffs = np.abs(feat[a_idx] - feat[b_idx]) / model.feature_scale
            sims = attribute_similarities(attr_rows[i], attr_rows[j])
            sims = np.where(np.isnan(sims), UNKNOWN_SIMILARITY, sims)
            pairs.append((i, j))
            vectors.append(np.concatenate([diffs, sims]))
    if not pairs:
        raise ModelError("need at least two archives to score pairs")
    return pairs, np.vstack(vectors)


def compute_pair_probabilities(archives, features, model) -> pd.DataFrame:
    """DataFrame(archive_id_a, archive_id_b, prob) for every pair."""
    pairs, design = build_pair_matrix(archives, features, model)
    probs = model.predict_pair_proba(design)
    frame = pd.DataFrame(
        {
            "archive_id_a": [p[0] for p in pairs],
            "archive_id_b": [p[1] for p in pairs],
            "prob": probs,
        }
    ).sort_values(["archive_id_a", "archive_id_b"], ignore_index=True)
    return frame
