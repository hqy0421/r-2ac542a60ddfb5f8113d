"""Report assembly: real features, pairwise probabilities, members,
master records, risk explanation and summary - everything the caller
asked to get back, in one JSON-safe dict."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rebind import changed as changed_updates


def _py(value):
    """numpy/pandas scalar -> JSON-safe python scalar (any NA -> None)."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _record_row(record, attributes):
    row = {"entity_id": record["entity_id"], "members": record["members"]}
    for attr in attributes:
        row[attr] = _py(record[attr])
    return row


def build_report(
    *,
    generation_id,
    inputs,
    features,
    pairs_df,
    result,
    masters,
    rebind_updates,
    model,
    model_path,
    timing,
    attributes,
):
    ids = result.archive_ids
    feature_sample = {}
    for archive_id, row in list(features.iterrows())[:5]:
        feature_sample[str(int(archive_id))] = {
            c: _py(row[c]) for c in features.columns[:8]
        }

    entities = [
        {"entity_id": min(members), "members": list(members)}
        for members in result.clusters
    ]
    master_rows = [_record_row(m, attributes) for m in masters]
    provenance = {str(m["entity_id"]): m["provenance"] for m in masters}

    pair_rows = [
        {
            "archive_id_a": int(r.archive_id_a),
            "archive_id_b": int(r.archive_id_b),
            "prob": round(float(r.prob), 6),
        }
        for r in pairs_df.itertuples()
    ]
    risk_pairs = [
        {
            "archive_id_a": int(r.archive_id_a),
            "archive_id_b": int(r.archive_id_b),
            "prob": round(float(r.prob), 6),
            "decision": r.decision,
            "risk": round(float(r.risk), 6),
            "constraint": r.constraint or None,
        }
        for r in result.pair_table.itertuples()
    ]

    baselines = result.baselines
    cc = baselines.get("threshold_connected_components", {})
    explanation_lines = [
        f"全局最低期望成对风险 R* = {result.total_risk:.4f}"
        f"（误合并风险 {result.merge_risk:.4f} + 漏合并风险 {result.split_risk:.4f}），"
        f"求解器：{result.solver}。",
        f"对比基线：全部拆分风险 {baselines['all_singletons']['total_risk']:.4f}；"
        + (
            f"全部合并风险 {baselines['all_merged']['total_risk']:.4f}；"
            if baselines.get("all_merged", {}).get("feasible")
            else "全部合并且违反必分约束（不可行）；"
        )
        + (
            f"阈值 τ={cc.get('threshold'):.3f} 连通分量风险 "
            f"{cc.get('total_risk'):.4f}，违反必合 {cc.get('must_link_violations')} 处、"
            f"必分 {cc.get('cannot_link_violations')} 处。"
            if cc
            else ""
        ),
        "划分满足全部必合/必分约束，且通过传递性（三角不等式）校验，"
        "为一致划分而非阈值连通图。",
    ]

    rebinding_changes = changed_updates(rebind_updates)
    n_nulled = sum(
        1 for u in rebind_updates
        if u["archive_id"] is not None and u["new_entity_id"] is None
    )

    report = {
        "generation_id": generation_id,
        "status": "ok",
        "features": {
            "path": "features.csv",
            "shape": [int(features.shape[0]), int(features.shape[1])],
            "archive_ids": ids,
            "columns": [str(c) for c in features.columns],
            "nan_cells": int(features.isna().sum().sum()),
            "archives_without_telemetry": [
                int(a) for a in features.index[features.isna().all(axis=1)]
            ],
            "sample_first5_archives_x_first8_features": feature_sample,
        },
        "pairwise_probabilities": {
            "path": "pair_probabilities.csv",
            "n_pairs": int(len(pairs_df)),
            "pairs": pair_rows,
        },
        "constraints": {
            "must_link": [list(map(int, t)) for t in result.must_link],
            "cannot_link": [list(map(int, t)) for t in result.cannot_link],
        },
        "costs": {
            "false_merge_cost": inputs.false_merge_cost,
            "missed_merge_cost": inputs.missed_merge_cost,
        },
        "partition": {"entities": entities, "n_entities": len(entities)},
        "master_records": {
            "path": "master_records.csv",
            "records": master_rows,
            "attribute_provenance": provenance,
        },
        "risk": {
            "total_expected_risk": round(result.total_risk, 6),
            "merge_risk": round(result.merge_risk, 6),
            "split_risk": round(result.split_risk, 6),
            "solver": result.solver,
            "per_pair": risk_pairs,
            "baselines": {
                k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                    for kk, vv in v.items()}
                for k, v in baselines.items()
            },
            "explanation": explanation_lines,
        },
        "rebinding": {
            "n_work_orders": int(len(rebind_updates)),
            "n_changed": len(rebinding_changes),
            "n_rebound_to_null": int(n_nulled),
            "updates": rebind_updates,
        },
        "model": {
            "path": str(model_path),
            "selected_features": list(model.selected_features),
            "training": model.training,
            "versions": model.versions,
        },
        "timing": timing,
        "summary": {
            "n_archives": len(ids),
            "n_entities": len(entities),
            "n_pairs": int(len(pairs_df)),
            "n_must_link": len(result.must_link),
            "n_cannot_link": len(result.cannot_link),
            "total_expected_risk": round(result.total_risk, 6),
            "risk_vs_all_singletons": round(
                baselines["all_singletons"]["total_risk"] - result.total_risk, 6
            ),
            "risk_vs_threshold_cc": (
                round(cc["total_risk"] - result.total_risk, 6)
                if cc else None
            ),
            "n_work_orders_rebound": len(rebinding_changes),
            "n_work_orders_nulled": int(n_nulled),
        },
    }
    return report
