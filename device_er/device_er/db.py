"""SQLite business database: schema + input loading.

The database is the system of record.  It holds the *current* device
archives (one row per source-system record), their telemetry time
series, the repair work orders whose ``entity_id`` reference the
pipeline rebinds, the must-link / cannot-link relations, the decision
costs and the source priorities.  Two small bookkeeping tables
(``generations`` and ``meta``) track which persisted generation is
currently in effect so a failed run always leaves the old one live.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from .errors import InputError

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_priority (
    source_system TEXT PRIMARY KEY,
    priority      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS archives (
    archive_id      INTEGER PRIMARY KEY,
    source_system   TEXT NOT NULL REFERENCES source_priority(source_system),
    serial_no       TEXT,
    asset_tag       TEXT,
    vendor          TEXT,
    model           TEXT,
    location        TEXT,
    firmware        TEXT,
    install_date    TEXT,
    warranty_months INTEGER,
    active          INTEGER,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS telemetry (
    archive_id  INTEGER NOT NULL REFERENCES archives(archive_id),
    ts          TEXT NOT NULL,
    vibration   REAL,
    temperature REAL,
    current     REAL,
    pressure    REAL
);
CREATE INDEX IF NOT EXISTS idx_telemetry_archive ON telemetry(archive_id, ts);

CREATE TABLE IF NOT EXISTS work_orders (
    wo_id       INTEGER PRIMARY KEY,
    archive_id  INTEGER,          -- original source reference (business value, never modified)
    entity_id   INTEGER,          -- resolved entity reference (rebound by the pipeline)
    title       TEXT,
    status      TEXT,
    cost        REAL,
    opened_at   TEXT
);

CREATE TABLE IF NOT EXISTS link_constraints (
    constraint_id INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('must_link', 'cannot_link')),
    archive_id_a  INTEGER NOT NULL,
    archive_id_b  INTEGER NOT NULL,
    reason        TEXT
);

CREATE TABLE IF NOT EXISTS costs (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    false_merge_cost  REAL NOT NULL,
    missed_merge_cost REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    status        TEXT NOT NULL,
    path          TEXT NOT NULL,
    n_archives    INTEGER,
    n_entities    INTEGER,
    total_risk    REAL,
    report_json   TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

#: archive attributes that participate in master-record merging,
#: in a stable, well-defined order.
ATTRIBUTES = [
    "serial_no",
    "asset_tag",
    "vendor",
    "model",
    "location",
    "firmware",
    "install_date",
    "warranty_months",
    "active",
    "notes",
]

#: telemetry kinds (tsfresh extracts features per kind)
TELEMETRY_KINDS = ["vibration", "temperature", "current", "pressure"]


@dataclass
class Inputs:
    archives: pd.DataFrame
    telemetry: pd.DataFrame
    work_orders: pd.DataFrame
    constraints: pd.DataFrame
    false_merge_cost: float
    missed_merge_cost: float
    source_priority: dict
    must_link: list = field(default_factory=list)
    cannot_link: list = field(default_factory=list)


def connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(db_path):
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def load_inputs(conn) -> Inputs:
    """Read every business input; raise :class:`InputError` on invalid data."""
    try:
        archives = pd.read_sql_query("SELECT * FROM archives ORDER BY archive_id", conn)
        telemetry = pd.read_sql_query(
            "SELECT * FROM telemetry ORDER BY archive_id, ts", conn
        )
        work_orders = pd.read_sql_query("SELECT * FROM work_orders ORDER BY wo_id", conn)
        constraints = pd.read_sql_query(
            "SELECT * FROM link_constraints ORDER BY constraint_id", conn
        )
        # keep nullable integers as real integers (pd.NA for NULL) so that
        # 0 / FALSE stay exactly what they are in master records
        for col in ("warranty_months", "active"):
            archives[col] = archives[col].astype("Int64")
        for col in ("archive_id", "entity_id"):
            work_orders[col] = work_orders[col].astype("Int64")
        priorities = {
            row["source_system"]: int(row["priority"])
            for row in conn.execute(
                "SELECT source_system, priority FROM source_priority"
            )
        }
        cost_row = conn.execute(
            "SELECT false_merge_cost, missed_merge_cost FROM costs WHERE id = 1"
        ).fetchone()
    except Exception as exc:  # pragma: no cover - defensive
        raise InputError(f"failed to read business inputs: {exc}") from exc

    if archives.empty:
        raise InputError("archives table is empty - nothing to resolve")
    if cost_row is None:
        raise InputError("costs table is empty - false/missed merge costs are required")
    if not priorities:
        raise InputError("source_priority table is empty")

    unknown_sources = set(archives["source_system"]) - set(priorities)
    if unknown_sources:
        raise InputError(
            f"archives use source systems without priority: {sorted(unknown_sources)}"
        )

    archive_ids = set(int(a) for a in archives["archive_id"])
    must_link, cannot_link = [], []
    for row in constraints.to_dict("records"):
        a, b = int(row["archive_id_a"]), int(row["archive_id_b"])
        if a == b:
            raise InputError(f"self-referential constraint on archive {a}")
        missing = [x for x in (a, b) if x not in archive_ids]
        if missing:
            raise InputError(
                f"constraint {row['constraint_id']} references unknown archives {missing}"
            )
        pair = (min(a, b), max(a, b))
        if row["kind"] == "must_link":
            must_link.append(pair)
        else:
            cannot_link.append(pair)

    return Inputs(
        archives=archives,
        telemetry=telemetry,
        work_orders=work_orders,
        constraints=constraints,
        false_merge_cost=float(cost_row["false_merge_cost"]),
        missed_merge_cost=float(cost_row["missed_merge_cost"]),
        source_priority=priorities,
        must_link=must_link,
        cannot_link=cannot_link,
    )


def current_generation(conn):
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'current_generation'"
    ).fetchone()
    return None if row is None else row["value"]


def set_current_generation(conn, generation_id):
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('current_generation', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (generation_id,),
    )


def register_generation(conn, entry):
    conn.execute(
        "INSERT INTO generations(generation_id, created_at, status, path, "
        "n_archives, n_entities, total_risk, report_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            entry["generation_id"],
            entry["created_at"],
            entry["status"],
            entry["path"],
            entry["n_archives"],
            entry["n_entities"],
            entry["total_risk"],
            entry["report_json"],
        ),
    )
    conn.execute(
        "UPDATE generations SET status = 'superseded' "
        "WHERE generation_id != ? AND status = 'current'",
        (entry["generation_id"],),
    )
