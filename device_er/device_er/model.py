"""The pre-fitted pairwise model: training, persistence, loading.

The pipeline treats the model as *given*: ``run`` only loads and
applies it.  ``train`` (run once, or whenever the world changes)
synthesises labelled entity pairs with the shared generative process,
extracts the same native tsfresh features, selects the most
discriminative ones, fits a logistic-regression pairwise classifier and
persists it as

* ``pairwise_model.pkl``  - primary artefact (pickled dict)
* ``model_card.json``     - fully self-describing fallback from which
  the model can be reconstructed without unpickling

Loading validates shapes and versions; any problem raises
:class:`ModelError` so the caller keeps the old generation.
"""

from __future__ import annotations

import json
import pickle
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ModelError
from .features import CURATED_FC_PARAMETERS, extract_archive_features
from .pairwise import ATTR_SIM_NAMES, UNKNOWN_SIMILARITY, attribute_similarities
from .telemetry_gen import ARCHETYPES, VENDORS, make_entity_params, jitter_params, sample_telemetry

MODEL_FORMAT = "device-er/pairwise-model"
MODEL_VERSION = 1


@dataclass
class PairwiseModel:
    selected_features: list
    feature_scale: np.ndarray
    feature_impute: np.ndarray
    attr_names: list
    coef: np.ndarray
    intercept: float
    training: dict = field(default_factory=dict)
    fc_parameters: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    created_at: str = ""

    def predict_pair_proba(self, design: np.ndarray) -> np.ndarray:
        design = np.asarray(design, dtype=float)
        if design.shape[1] != self.coef.shape[0]:
            raise ModelError(
                f"pair vector has {design.shape[1]} dimensions, "
                f"model expects {self.coef.shape[0]}"
            )
        logits = design @ self.coef + self.intercept
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(probs, 1e-6, 1.0 - 1e-6)


def _versions():
    import sklearn
    import tsfresh

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "tsfresh": tsfresh.__version__,
    }


# ---------------------------------------------------------------- training


def _synthesize_training_world(rng, n_entities):
    """Archives + telemetry + entity labels from the generative process."""
    kinds = sorted(ARCHETYPES)
    sources = ["erp", "mes", "iot"]
    archive_rows, telemetry_frames, entity_of = [], [], {}
    for e in range(n_entities):
        kind = kinds[e % len(kinds)]
        params = make_entity_params(rng, kind)
        serial = f"SN-{90000 + e}"
        asset_tag = f"{kind[:3].upper()}-{1000 + e}"
        vendor = VENDORS[kind]
        model_name = f"{kind.title()}-{chr(65 + e % 6)}"
        location = f"Plant-{chr(65 + e % 8)}"
        for j in range(int(rng.integers(1, 4))):
            archive_id = 10000 + e * 10 + j
            entity_of[archive_id] = e
            source = sources[(e + j) % 3]

            def maybe(value, null_p):
                return None if rng.random() < null_p else value

            def noisy(text, p=0.15):
                if text is None:
                    return None
                if rng.random() < p / 2:
                    return text.upper() if rng.random() < 0.5 else text.lower()
                if rng.random() < p / 2:
                    return text + " "
                return text

            archive_rows.append(
                {
                    "archive_id": archive_id,
                    "source_system": source,
                    "serial_no": maybe(noisy(serial), 0.15),
                    "asset_tag": maybe(noisy(asset_tag), 0.20),
                    "vendor": maybe(noisy(vendor), 0.20),
                    "model": maybe(noisy(model_name), 0.15),
                    "location": maybe(noisy(location), 0.25),
                    "firmware": maybe(f"{e % 4}.{e % 7}.{j}", 0.30),
                    "install_date": maybe(f"20{16 + e % 8}-0{1 + e % 9}-1{e % 9}", 0.25),
                    "warranty_months": maybe(int(rng.choice([0, 12, 24, 36, 48])), 0.25),
                    "active": maybe(int(rng.random() < 0.85), 0.15),
                    "notes": maybe("" if rng.random() < 0.5 else f"note {e}", 0.3),
                }
            )
            n_points = int(rng.integers(40, 80))
            telemetry_frames.append(
                sample_telemetry(
                    rng,
                    jitter_params(rng, params, scale=0.05),
                    archive_id,
                    n_points,
                    start=f"2026-{1 + e % 3:02d}-01",
                )
            )
    archives = pd.DataFrame(archive_rows)
    telemetry = pd.concat(telemetry_frames, ignore_index=True)
    return archives, telemetry, entity_of


def train_pairwise_model(
    model_path,
    n_entities: int = 70,
    seed: int = 7,
    fc_parameters=None,
    top_k: int = 12,
    neg_ratio: int = 6,
    n_jobs: int = 1,
) -> dict:
    """Fit and persist the pairwise model; returns the model card."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    fc_parameters = dict(fc_parameters or CURATED_FC_PARAMETERS)
    rng = np.random.default_rng(seed)
    archives, telemetry, entity_of = _synthesize_training_world(rng, n_entities)
    ids = sorted(entity_of)
    features = extract_archive_features(telemetry, ids, fc_parameters, n_jobs=n_jobs)

    # labelled pairs: all same-entity pairs + sampled negatives
    pos_pairs, neg_pairs = [], []
    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1 :]:
            (pos_pairs if entity_of[a] == entity_of[b] else neg_pairs).append((a, b))
    rng.shuffle(neg_pairs)
    neg_pairs = neg_pairs[: max(len(pos_pairs) * neg_ratio, 50)]
    pairs = pos_pairs + neg_pairs
    labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

    index = {a: k for k, a in enumerate(ids)}
    feat_all = features.to_numpy(dtype=float)
    col_impute = np.nanmedian(
        np.where(np.isnan(feat_all), np.nan, feat_all), axis=0
    )
    col_impute = np.where(np.isnan(col_impute), 0.0, col_impute)
    feat_imp = np.where(np.isnan(feat_all), col_impute, feat_all)

    # univariate screening of candidate tsfresh features by |diff| AUC
    nan_rate = np.isnan(feat_all).mean(axis=0)
    std = feat_imp.std(axis=0)
    candidates = [
        c for c in range(features.shape[1]) if nan_rate[c] <= 0.2 and std[c] > 0
    ]
    if not candidates:
        raise ModelError("no usable tsfresh features for model training")
    pa = np.array([index[a] for a, _ in pairs])
    pb = np.array([index[b] for _, b in pairs])
    scores = []
    for c in candidates:
        diff = np.abs(feat_imp[pa, c] - feat_imp[pb, c])
        if diff.std() == 0:
            continue
        auc = roc_auc_score(labels, -diff)  # same entity -> smaller diff
        scores.append((abs(auc - 0.5), c, auc))
    if not scores:
        raise ModelError("no discriminative tsfresh features found")
    scores.sort(reverse=True)
    chosen = [c for _, c, _ in scores[:top_k]]
    selected = [str(features.columns[c]) for c in chosen]

    diffs = np.abs(feat_imp[pa][:, chosen] - feat_imp[pb][:, chosen])
    q25, q75 = np.percentile(diffs, [25, 75], axis=0)
    scale = np.where(q75 > q25, q75 - q25, diffs.std(axis=0))
    scale = np.where(scale > 0, scale, 1.0)

    by_id = archives.set_index("archive_id")
    sims = np.array(
        [
            attribute_similarities(by_id.loc[a], by_id.loc[b])
            for a, b in pairs
        ]
    )
    sims = np.where(np.isnan(sims), UNKNOWN_SIMILARITY, sims)
    design = np.hstack([diffs / scale, sims])

    clf = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(clf, design, labels, cv=cv, scoring="roc_auc")
    clf.fit(design, labels)
    train_prob = clf.predict_proba(design)[:, 1]
    metrics = {
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "train_auc": float(roc_auc_score(labels, train_prob)),
        "train_brier": float(brier_score_loss(labels, train_prob)),
    }

    model = PairwiseModel(
        selected_features=selected,
        feature_scale=scale,
        feature_impute=col_impute[chosen],
        attr_names=list(ATTR_SIM_NAMES),
        coef=clf.coef_.ravel().astype(float),
        intercept=float(clf.intercept_[0]),
        training={
            "n_entities": int(n_entities),
            "n_archives": int(len(ids)),
            "n_pairs": int(len(pairs)),
            "n_positive_pairs": int(len(pos_pairs)),
            "seed": int(seed),
            "top_k": int(top_k),
            **metrics,
        },
        fc_parameters=fc_parameters,
        versions=_versions(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    # stored-coefficient scoring must reproduce the fitted sklearn model
    check = model.predict_pair_proba(design[:8])
    assert np.allclose(check, train_prob[:8], atol=1e-8), "model serialisation mismatch"

    save_pairwise_model(model, model_path)
    return model_card(model)


def model_card(model: PairwiseModel) -> dict:
    return {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "created_at": model.created_at,
        "selected_features": list(model.selected_features),
        "feature_scale": [float(x) for x in model.feature_scale],
        "feature_impute": [float(x) for x in model.feature_impute],
        "attr_names": list(model.attr_names),
        "coef": [float(x) for x in model.coef],
        "intercept": float(model.intercept),
        "training": model.training,
        "fc_parameters": model.fc_parameters,
        "versions": model.versions,
    }


def save_pairwise_model(model: PairwiseModel, model_path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "selected_features": model.selected_features,
        "feature_scale": model.feature_scale,
        "feature_impute": model.feature_impute,
        "attr_names": model.attr_names,
        "coef": model.coef,
        "intercept": model.intercept,
        "training": model.training,
        "fc_parameters": model.fc_parameters,
        "versions": model.versions,
        "created_at": model.created_at,
    }
    tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    tmp.replace(model_path)
    card_path = model_path.with_suffix(".card.json")
    card_path.write_text(json.dumps(model_card(model), indent=2, ensure_ascii=False))


def _model_from_payload(payload: dict) -> PairwiseModel:
    if payload.get("format") != MODEL_FORMAT:
        raise ModelError(f"unrecognised model format: {payload.get('format')!r}")
    if int(payload.get("version", -1)) != MODEL_VERSION:
        raise ModelError(f"unsupported model version: {payload.get('version')!r}")
    model = PairwiseModel(
        selected_features=[str(c) for c in payload["selected_features"]],
        feature_scale=np.asarray(payload["feature_scale"], dtype=float),
        feature_impute=np.asarray(payload["feature_impute"], dtype=float),
        attr_names=[str(a) for a in payload["attr_names"]],
        coef=np.asarray(payload["coef"], dtype=float),
        intercept=float(payload["intercept"]),
        training=dict(payload.get("training", {})),
        fc_parameters=dict(payload.get("fc_parameters", {})),
        versions=dict(payload.get("versions", {})),
        created_at=str(payload.get("created_at", "")),
    )
    expected = len(model.selected_features) + len(model.attr_names)
    if model.coef.shape[0] != expected:
        raise ModelError(
            f"model is inconsistent: {model.coef.shape[0]} coefficients for "
            f"{expected} pair features"
        )
    if model.feature_scale.shape[0] != len(model.selected_features):
        raise ModelError("model feature_scale does not match selected features")
    if np.any(model.feature_scale <= 0):
        raise ModelError("model feature_scale must be strictly positive")
    return model


def load_pairwise_model(model_path) -> PairwiseModel:
    """Load the pre-fitted model; fall back to the JSON card."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelError(f"pairwise model not found: {model_path}")
    try:
        with open(model_path, "rb") as fh:
            payload = pickle.load(fh)
        return _model_from_payload(payload)
    except ModelError:
        raise
    except Exception as pickle_exc:
        card_path = model_path.with_suffix(".card.json")
        if not card_path.exists():
            raise ModelError(
                f"failed to load pairwise model {model_path}: {pickle_exc}"
            ) from pickle_exc
        try:
            payload = json.loads(card_path.read_text())
            return _model_from_payload(payload)
        except Exception as card_exc:
            raise ModelError(
                f"failed to load pairwise model {model_path}: {card_exc}"
            ) from card_exc
