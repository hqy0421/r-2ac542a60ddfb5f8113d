"""Generation persistence: atomic save, re-open, corruption detection,
and keep-the-old-generation on failure."""

import json

import pandas as pd
import pytest

from device_er import db as db_mod
from device_er.errors import SaveError
from device_er.storage import GenerationStore


def _artifacts():
    return {
        "features.csv": ("csv", pd.DataFrame({"archive_id": [1], "f": [0.5]})),
        "pair_probabilities.csv": ("csv", pd.DataFrame(
            {"archive_id_a": [1], "archive_id_b": [2], "prob": [0.7]})),
        "partition.json": ("json", {"entities": [{"entity_id": 1, "members": [1, 2]}]}),
        "master_records.json": ("json", [{"entity_id": 1}]),
        "work_orders_after.csv": ("csv", pd.DataFrame({"wo_id": [1]})),
        "work_order_changes.csv": ("csv", pd.DataFrame({"wo_id": [1]})),
        "report.json": ("json", {"generation_id": "gen_0001", "status": "ok"}),
    }


def _registry(gid):
    return {"generation_id": gid, "created_at": "2026-11-20T00:00:00+00:00",
            "status": "current", "path": gid, "n_archives": 2, "n_entities": 1,
            "total_risk": 1.0, "report_json": "{}"}


@pytest.fixture()
def store_and_conn(tmp_path):
    db_path = tmp_path / "t.db"
    db_mod.initialize(db_path)
    conn = db_mod.connect(db_path)
    store = GenerationStore(tmp_path / "generations")
    yield store, conn
    conn.close()


def test_save_and_reopen_roundtrip(store_and_conn):
    store, conn = store_and_conn
    gid = store.save(generation_id="gen_0001", artifacts=_artifacts(),
                     manifest_extra={"k": "v"}, conn=conn,
                     rebind_updates=[], registry_entry=_registry("gen_0001"))
    assert gid == "gen_0001"
    assert db_mod.current_generation(conn) == "gen_0001"
    gen = store.reopen("gen_0001")
    assert gen.generation_id == "gen_0001"
    assert gen.report["status"] == "ok"
    assert gen.partition["entities"][0]["members"] == [1, 2]
    assert list(gen.features["f"]) == [0.5]
    assert gen.manifest["k"] == "v"


def test_reopen_detects_corruption(store_and_conn):
    store, conn = store_and_conn
    store.save(generation_id="gen_0001", artifacts=_artifacts(),
               manifest_extra={}, conn=conn,
               rebind_updates=[], registry_entry=_registry("gen_0001"))
    target = store.root / "gen_0001" / "features.csv"
    target.write_text(target.read_text() + "tampered")
    with pytest.raises(SaveError):
        store.reopen("gen_0001")


def test_failed_db_commit_keeps_previous_generation(store_and_conn, tmp_path):
    store, conn = store_and_conn
    store.save(generation_id="gen_0001", artifacts=_artifacts(),
               manifest_extra={}, conn=conn,
               rebind_updates=[], registry_entry=_registry("gen_0001"))
    # break the database file so the commit inside save() fails
    conn.close()
    db_path = tmp_path / "t.db"
    db_path.write_bytes(b"not a sqlite database")
    broken = db_mod.connect(db_path)
    with pytest.raises(SaveError):
        store.save(generation_id="gen_0002", artifacts=_artifacts(),
                   manifest_extra={}, conn=broken,
                   rebind_updates=[], registry_entry=_registry("gen_0002"))
    broken.close()
    assert not (store.root / "gen_0002").exists()      # promoted dir rolled back
    assert not list(store.root.glob(".tmp-*"))          # tmp dir cleaned
    gen = store.reopen("gen_0001")                      # old one intact
    assert gen.report["status"] == "ok"


def test_next_id_monotonic(store_and_conn):
    store, conn = store_and_conn
    assert store.next_id() == "gen_0001"
    store.save(generation_id="gen_0001", artifacts=_artifacts(),
               manifest_extra={}, conn=conn,
               rebind_updates=[], registry_entry=_registry("gen_0001"))
    assert store.next_id() == "gen_0002"
