"""Pipeline orchestration with keep-the-old-generation semantics.

Every failure domain (extraction / model / optimization / save) aborts
the run *before* anything is committed, so the previous generation and
the work-order references stay exactly as they were.  A successful run
returns the full report dict and has already: persisted a complete
re-openable generation, rebound the work orders in the database and
marked the generation current.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import db as db_mod
from .db import ATTRIBUTES
from .errors import InputError, PipelineError
from .features import extract_archive_features
from .master import build_master_records
from .model import load_pairwise_model, model_card
from .optimize import solve_partition
from .pairwise import compute_pair_probabilities
from .rebind import compute_rebinding
from .report import build_report
from .storage import GenerationStore, json_default


@dataclass
class PipelineConfig:
    db_path: Path
    model_path: Path
    generations_root: Path
    fc_parameters: dict | None = None
    n_jobs: int = 1

    @classmethod
    def from_root(cls, root, **overrides):
        root = Path(root)
        return cls(
            db_path=overrides.pop("db_path", root / "data" / "device_er.db"),
            model_path=overrides.pop(
                "model_path", root / "model" / "pairwise_model.pkl"
            ),
            generations_root=overrides.pop(
                "generations_root", root / "generations"
            ),
            **overrides,
        )


def _work_orders_after(work_orders: pd.DataFrame, updates) -> pd.DataFrame:
    after = work_orders.copy()
    new_ids = {u["wo_id"]: u["new_entity_id"] for u in updates}
    after["entity_id"] = [
        new_ids.get(int(wo)) for wo in after["wo_id"]
    ]
    return after


def run_pipeline(config: PipelineConfig) -> dict:
    """Run one full resolution pass; returns the report dict.

    Raises :class:`PipelineError` (with a failure ``domain``) if any
    stage fails; in that case the previous generation and all business
    data are guaranteed untouched.
    """
    started = time.monotonic()
    timing = {}
    config.db_path = Path(config.db_path)
    config.model_path = Path(config.model_path)
    config.generations_root = Path(config.generations_root)
    if not config.db_path.exists():
        raise InputError(f"business database not found: {config.db_path}")

    conn = db_mod.connect(config.db_path)
    try:
        inputs = db_mod.load_inputs(conn)
        archive_ids = [int(a) for a in inputs.archives["archive_id"]]

        # ---- 1. native tsfresh feature extraction ---------------------
        t0 = time.monotonic()
        features = extract_archive_features(
            inputs.telemetry, archive_ids,
            fc_parameters=config.fc_parameters, n_jobs=config.n_jobs,
        )
        timing["extraction_seconds"] = round(time.monotonic() - t0, 3)

        # ---- 2. pre-fitted pairwise model -> probabilities ------------
        t0 = time.monotonic()
        model = load_pairwise_model(config.model_path)
        pairs_df = compute_pair_probabilities(inputs.archives, features, model)
        timing["model_seconds"] = round(time.monotonic() - t0, 3)

        # ---- 3. global minimum-risk consistent partition --------------
        t0 = time.monotonic()
        probs = {
            (int(r.archive_id_a), int(r.archive_id_b)): float(r.prob)
            for r in pairs_df.itertuples()
        }
        result = solve_partition(
            archive_ids,
            probs,
            inputs.false_merge_cost,
            inputs.missed_merge_cost,
            must_link=inputs.must_link,
            cannot_link=inputs.cannot_link,
        )
        timing["optimization_seconds"] = round(time.monotonic() - t0, 3)

        # ---- 4. master records + rebinding plan (pure) ----------------
        masters = build_master_records(
            result.clusters, inputs.archives, inputs.source_priority, ATTRIBUTES
        )
        wo_records = inputs.work_orders.to_dict("records")
        updates = compute_rebinding(
            wo_records, result.archive_to_entity, set(archive_ids)
        )

        # ---- 5. report + atomic generation save (applies rebinding) ---
        t0 = time.monotonic()
        store = GenerationStore(config.generations_root)
        generation_id = store.next_id()
        timing["total_seconds"] = round(time.monotonic() - started, 3)
        report = build_report(
            generation_id=generation_id,
            inputs=inputs,
            features=features,
            pairs_df=pairs_df,
            result=result,
            masters=masters,
            rebind_updates=updates,
            model=model,
            model_path=config.model_path,
            timing=timing,
            attributes=ATTRIBUTES,
        )

        features_out = features.copy()
        features_out.insert(0, "archive_id", features_out.index)
        master_rows = []
        for m in masters:
            row = {"entity_id": m["entity_id"], "member_count": m["member_count"],
                   "members": " ".join(str(x) for x in m["members"])}
            for attr in ATTRIBUTES:
                row[attr] = m[attr]
            master_rows.append(row)
        masters_df = pd.DataFrame(master_rows)
        wo_after = _work_orders_after(inputs.work_orders, updates)
        changes_df = pd.DataFrame(updates)

        try:
            model_bytes = Path(config.model_path).read_bytes()
        except OSError:
            model_bytes = b""

        created_at = datetime.now(timezone.utc).isoformat()
        artifacts = {
            "features.csv": ("csv", features_out),
            "pair_probabilities.csv": ("csv", pairs_df),
            "partition.json": ("json", {
                "entities": [
                    {"entity_id": min(m), "members": list(m)}
                    for m in result.clusters
                ]
            }),
            "master_records.csv": ("csv", masters_df),
            "master_records.json": ("json", masters),
            "work_orders_after.csv": ("csv", wo_after),
            "work_order_changes.csv": ("csv", changes_df),
            "risk_pairs.csv": ("csv", result.pair_table),
            "report.json": ("json", report),
            "config.json": ("json", {
                "db_path": str(config.db_path),
                "model_path": str(config.model_path),
                "costs": {
                    "false_merge_cost": inputs.false_merge_cost,
                    "missed_merge_cost": inputs.missed_merge_cost,
                },
                "source_priority": inputs.source_priority,
                "must_link": inputs.must_link,
                "cannot_link": inputs.cannot_link,
                "fc_parameters": config.fc_parameters or model.fc_parameters,
            }),
            "model_card.json": ("json", model_card(model)),
            "model_copy.pkl": ("bytes", model_bytes),
        }
        manifest_extra = {
            "n_archives": len(archive_ids),
            "n_entities": len(result.clusters),
            "total_risk": result.total_risk,
            "solver": result.solver,
            "model_versions": model.versions,
        }
        registry_entry = {
            "generation_id": generation_id,
            "created_at": created_at,
            "status": "current",
            "path": str(config.generations_root / generation_id),
            "n_archives": len(archive_ids),
            "n_entities": len(result.clusters),
            "total_risk": result.total_risk,
            "report_json": json.dumps(report, ensure_ascii=False,
                                      default=json_default),
        }
        store.save(
            generation_id=generation_id,
            artifacts=artifacts,
            manifest_extra=manifest_extra,
            conn=conn,
            rebind_updates=updates,
            registry_entry=registry_entry,
        )
        timing["save_seconds"] = round(time.monotonic() - t0, 3)
        report["timing"] = timing
        return report
    except PipelineError:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise PipelineError(f"unexpected pipeline failure: {exc}") from exc
    finally:
        conn.close()
