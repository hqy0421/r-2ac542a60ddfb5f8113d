"""Command line interface.

    python -m device_er seed  --root DIR [--force]
    python -m device_er train --root DIR [--entities N] [--seed S]
    python -m device_er run   --root DIR
    python -m device_er show  --root DIR [--generation ID|current]
    python -m device_er list  --root DIR

Exit codes: 0 = ok, 2 = pipeline failure (previous generation kept),
1 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db as db_mod
from .errors import PipelineError
from .model import train_pairwise_model
from .pipeline import PipelineConfig, run_pipeline
from .seed import seed as seed_db
from .storage import GenerationStore, json_default


def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default))


def _config(args) -> PipelineConfig:
    return PipelineConfig.from_root(args.root)


def cmd_seed(args):
    path = seed_db(Path(args.root) / "data" / "device_er.db", force=args.force)
    _print({"status": "ok", "db": str(path)})
    return 0


def cmd_train(args):
    model_path = Path(args.root) / "model" / "pairwise_model.pkl"
    card = train_pairwise_model(
        model_path,
        n_entities=args.entities,
        seed=args.seed,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
    )
    _print({"status": "ok", "model": str(model_path), "card": card})
    return 0


def cmd_run(args):
    config = _config(args)
    try:
        report = run_pipeline(config)
    except PipelineError as exc:
        current = None
        try:
            conn = db_mod.connect(config.db_path)
            current = db_mod.current_generation(conn)
            conn.close()
        except Exception:
            pass
        _print({
            "status": "failed",
            "kept_generation": current,
            "error": exc.as_dict(),
        })
        return 2
    _print(report)
    return 0


def cmd_show(args):
    config = _config(args)
    store = GenerationStore(config.generations_root)
    conn = db_mod.connect(config.db_path)
    try:
        gid = args.generation
        if gid in (None, "current"):
            gid = db_mod.current_generation(conn)
            if gid is None:
                _print({"status": "empty", "message": "no current generation"})
                return 0
        generation = store.reopen(gid)
    finally:
        conn.close()
    report = generation.report
    _print({
        "status": "ok",
        "generation_id": generation.generation_id,
        "entities": report["partition"]["entities"],
        "master_records": report["master_records"]["records"],
        "risk": {
            "total_expected_risk": report["risk"]["total_expected_risk"],
            "merge_risk": report["risk"]["merge_risk"],
            "split_risk": report["risk"]["split_risk"],
            "baselines": report["risk"]["baselines"],
            "explanation": report["risk"]["explanation"],
        },
        "rebinding": {
            "n_work_orders": report["rebinding"]["n_work_orders"],
            "n_changed": report["rebinding"]["n_changed"],
            "n_rebound_to_null": report["rebinding"]["n_rebound_to_null"],
        },
        "summary": report["summary"],
    })
    return 0


def cmd_list(args):
    config = _config(args)
    conn = db_mod.connect(config.db_path)
    try:
        rows = conn.execute(
            "SELECT generation_id, created_at, status, n_archives, "
            "n_entities, total_risk FROM generations ORDER BY generation_id"
        ).fetchall()
        current = db_mod.current_generation(conn)
    finally:
        conn.close()
    _print({
        "current": current,
        "generations": [dict(r) for r in rows],
    })
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="device_er")
    parser.add_argument(
        "--root", default=".",
        help="project root holding data/ model/ generations/ (default: cwd)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed", help="create the demo business database")
    p.add_argument("--force", action="store_true", help="rebuild if it exists")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("train", help="fit and persist the pairwise model")
    p.add_argument("--entities", type=int, default=70)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=1)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("run", help="run one full resolution pass")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("show", help="re-open and display a generation")
    p.add_argument("--generation", default="current")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("list", help="list registered generations")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PipelineError as exc:
        _print({"status": "failed", "error": exc.as_dict()})
        return 2


if __name__ == "__main__":
    sys.exit(main())
