"""End-to-end: seed -> train -> run -> re-open, plus every
keep-the-old-generation failure domain."""

import json
import sqlite3

import pandas as pd
import pytest

from device_er import db as db_mod
from device_er.errors import (
    ExtractionError,
    ModelError,
    SaveError,
)
from device_er.model import train_pairwise_model
from device_er.pipeline import PipelineConfig, run_pipeline
from device_er.storage import GenerationStore

TINY_FC = {
    "mean": None,
    "standard_deviation": None,
    "maximum": None,
    "minimum": None,
    "quantile": [{"q": 0.25}, {"q": 0.75}],
    "linear_trend": [{"attr": "slope"}, {"attr": "intercept"}],
    "autocorrelation": [{"lag": 1}],
    "root_mean_square": None,
}


@pytest.fixture(scope="module")
def trained_runtime(tmp_path_factory):
    # module-scoped: train once, reuse across tests
    from device_er.seed import seed

    root = tmp_path_factory.mktemp("e2e")
    (root / "data").mkdir()
    db_path = seed(root / "data" / "device_er.db")
    model_path = root / "model" / "pairwise_model.pkl"
    train_pairwise_model(
        model_path, n_entities=24, seed=3, fc_parameters=dict(TINY_FC),
        top_k=6, n_jobs=1,
    )
    return {"root": root, "db": db_path, "model": model_path, "fc": dict(TINY_FC)}


def _config(rt, gens="generations"):
    return PipelineConfig(
        db_path=rt["db"],
        model_path=rt["model"],
        generations_root=rt["root"] / gens,
        fc_parameters=rt["fc"],
        n_jobs=1,
    )


def _run_ok(rt):
    return run_pipeline(_config(rt))


def test_full_run_persists_reopenable_generation(trained_runtime):
    rt = trained_runtime
    report = _run_ok(rt)
    assert report["status"] == "ok"
    assert report["generation_id"] == "gen_0001"
    # real features: one row per archive, native tsfresh columns
    n_archives = report["summary"]["n_archives"]
    assert report["features"]["shape"][0] == n_archives
    assert report["features"]["shape"][1] > 0
    # pairwise probabilities: complete and valid
    n_pairs = n_archives * (n_archives - 1) // 2
    assert len(report["pairwise_probabilities"]["pairs"]) == n_pairs
    for pair in report["pairwise_probabilities"]["pairs"]:
        assert 0.0 <= pair["prob"] <= 1.0
    # entity id = min member id; constraints satisfied
    entities = report["partition"]["entities"]
    for entity in entities:
        assert entity["entity_id"] == min(entity["members"])
    member_of = {}
    for entity in entities:
        for m in entity["members"]:
            member_of[m] = entity["entity_id"]
    for a, b in report["constraints"]["must_link"]:
        assert member_of[a] == member_of[b]
    for a, b in report["constraints"]["cannot_link"]:
        assert member_of[a] != member_of[b]
    # risk report: globally optimal <= baselines
    risk = report["risk"]
    assert risk["total_expected_risk"] <= risk["baselines"]["all_singletons"]["total_risk"] + 1e-9
    # work orders actually rebound in the database
    conn = sqlite3.connect(rt["db"])
    rows = dict(conn.execute("SELECT wo_id, entity_id FROM work_orders").fetchall())
    archived = dict(conn.execute("SELECT wo_id, archive_id FROM work_orders").fetchall())
    conn.close()
    for wo_id, archive_id in archived.items():
        if archive_id in (999, 555):
            assert rows[wo_id] is None            # disappeared source -> NULL
        else:
            assert rows[wo_id] == member_of[archive_id]
    # other business values untouched
    conn = sqlite3.connect(rt["db"])
    titles = conn.execute("SELECT title FROM work_orders WHERE wo_id = 16").fetchone()
    conn.close()
    assert titles[0] == "Legacy archive deleted upstream"
    # the generation re-opens and matches the report
    store = GenerationStore(rt["root"] / "generations")
    gen = store.reopen("gen_0001")
    assert gen.report["summary"] == report["summary"]
    assert gen.features.shape[0] == n_archives
    assert len(gen.masters) == len(entities)


def test_second_run_creates_next_generation(trained_runtime):
    report = _run_ok(trained_runtime)
    assert report["generation_id"] == "gen_0002"
    conn = db_mod.connect(trained_runtime["db"])
    current = db_mod.current_generation(conn)
    conn.close()
    assert current == "gen_0002"


def test_model_failure_keeps_old_generation(trained_runtime, tmp_path):
    rt = dict(trained_runtime)
    rt["model"] = tmp_path / "missing_model.pkl"
    before = _snapshot(rt)
    with pytest.raises(ModelError):
        _run_ok(rt)
    assert _snapshot(rt) == before


def test_extraction_failure_keeps_old_generation(trained_runtime):
    rt = trained_runtime
    conn = sqlite3.connect(rt["db"])
    conn.execute("DELETE FROM telemetry")
    conn.commit()
    conn.close()
    before = _snapshot(rt)
    with pytest.raises(ExtractionError):
        _run_ok(rt)
    assert _snapshot(rt) == before
    # restore telemetry for the following tests
    _reseed_telemetry(rt)


def test_save_failure_keeps_old_generation(trained_runtime):
    rt = trained_runtime
    before = _snapshot(rt)
    config = _config(rt)
    config.generations_root = rt["root"] / "blocked" / ".." / "blocked"
    # make the generations root a *file* so directory creation fails
    blocker = rt["root"] / "blocked"
    blocker.write_text("occupied")
    config.generations_root = blocker / "sub"
    with pytest.raises(SaveError):
        run_pipeline(config)
    assert _snapshot(rt) == before
    blocker.unlink()


def test_archive_disappearance_nulls_work_orders(trained_runtime):
    rt = trained_runtime
    conn = sqlite3.connect(rt["db"])
    conn.execute("DELETE FROM telemetry WHERE archive_id = 311")
    conn.execute("DELETE FROM archives WHERE archive_id = 311")
    conn.commit()
    wo18_before = conn.execute(
        "SELECT entity_id FROM work_orders WHERE wo_id = 18"
    ).fetchone()[0]
    conn.close()
    assert wo18_before is not None
    report = _run_ok(rt)
    assert report["status"] == "ok"
    conn = sqlite3.connect(rt["db"])
    wo18_after = conn.execute(
        "SELECT entity_id FROM work_orders WHERE wo_id = 18"
    ).fetchone()[0]
    conn.close()
    assert wo18_after is None                    # source archive gone -> NULL
    assert 311 not in report["features"]["archive_ids"]


def _snapshot(rt):
    conn = sqlite3.connect(rt["db"])
    wos = conn.execute(
        "SELECT wo_id, archive_id, entity_id, title, status, cost, opened_at "
        "FROM work_orders ORDER BY wo_id"
    ).fetchall()
    try:
        current = conn.execute(
            "SELECT value FROM meta WHERE key='current_generation'"
        ).fetchone()
    except sqlite3.Error:
        current = None
    gens = []
    try:
        gens = conn.execute(
            "SELECT generation_id, status FROM generations ORDER BY generation_id"
        ).fetchall()
    except sqlite3.Error:
        pass
    conn.close()
    return {"wos": wos, "current": current, "generations": gens}


def _reseed_telemetry(rt):
    # re-insert telemetry rows deleted by the extraction-failure test
    from device_er.seed import seed
    import shutil

    fresh = rt["root"] / "data" / "fresh.db"
    if fresh.exists():
        fresh.unlink()
    seed(fresh)
    src = sqlite3.connect(fresh)
    rows = src.execute(
        "SELECT archive_id, ts, vibration, temperature, current, pressure "
        "FROM telemetry"
    ).fetchall()
    src.close()
    conn = sqlite3.connect(rt["db"])
    conn.executemany(
        "INSERT INTO telemetry(archive_id, ts, vibration, temperature, current, "
        "pressure) VALUES (?,?,?,?,?,?)", rows,
    )
    conn.commit()
    conn.close()
    fresh.unlink()
