"""Seed a realistic demo business database.

Three source systems with a priority order (erp > mes > iot), fifteen
device archives forming nine true physical entities, telemetry per
archive, repair work orders (two of them referencing archives that no
longer exist), must-link / cannot-link relations and asymmetric merge
costs.  Deterministic (fixed RNG seed).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from . import db as db_mod
from .telemetry_gen import make_entity_params, jitter_params, sample_telemetry

SEED = 20261120

# (archive_id, source, serial, asset_tag, vendor, model, location, firmware,
#  install_date, warranty_months, active, notes)
ARCHIVES = [
    # E1 - pump, three systems
    (101, "erp", "SN-77001", "PMP-1001", "FlowServe", "PumpX-100", "Plant-A/B1",
     "2.1.0", "2019-03-11", 36, 1, "primary record"),
    (201, "mes", None, "PMP-1001", "flowserve", "PumpX-100", "Plant-A / B1",
     None, None, 0, 1, ""),                      # warranty 0 and notes '' must survive
    (301, "iot", "SN-77001", None, None, "PumpX-100", None,
     "2.1.3", None, None, 0, None),               # active 0 (FALSE) must survive
    # E2 - compressor, erp + mes
    (102, "erp", "SN-88002", "CMP-2002", "AtlasCopco", "Comp-50", "Plant-A/B2",
     "1.4.0", "2020-06-01", 24, 1, None),
    (202, "mes", "SN-88002", None, "Atlas Copco", "Comp-50", None,
     None, "2020-06-01", None, 1, "duplicate from MES"),
    # E3 - valve, erp + iot; iot telemetry drifted (must_link)
    (103, "erp", "SN-99003", "VLV-3003", "Emerson", "ValvePro-9", "Plant-B",
     None, "2018-01-20", 12, 1, None),
    (303, "iot", None, "VLV-3003", "Emerson", "ValvePro-9 ", "Plant-B",
     "3.0.1", None, None, None, ""),
    # E4 - motor, singleton
    (104, "erp", "SN-11004", "MTR-4004", "Siemens", "MotorS-4", "Plant-C",
     "5.2.0", "2021-11-05", 48, 1, None),
    # E5 - fan, mes + iot (no erp record)
    (205, "mes", "SN-55005", "FAN-5005", "ebm-papst", "FanZ-77", "Plant-A/Roof",
     None, None, 6, 1, None),
    (305, "iot", "SN-55005", None, None, "FanZ-77", "Roof",
     "0.9.2", None, None, 1, None),
    # E6 - same pump model line as E1 but a *different* physical unit (cannot_link);
    # its telemetry sits close to E1's operating point (same model line, similar
    # duty), which is exactly what makes the cannot_link relation necessary
    (106, "erp", "SN-77009", "PMP-1009", "FlowServe", "PumpX-100", "Plant-D",
     "2.1.0", "2022-02-14", 36, 1, None),
    # E10 - iot-only pump archive without a registered serial; its telemetry
    # falls between E1 and E6, forming a probability chain that breaks
    # naive threshold connected-components clustering
    (311, "iot", None, None, "FlowServe", "PumpX-100", None,
     "2.1.0", None, None, 1, None),
    # E7 - same-system duplicate inside MES (must_link)
    (207, "mes", "SN-66007", "BLW-7007", "Howden", "Blower-2", "Plant-B",
     None, "2017-08-30", None, 1, None),
    (208, "mes", None, "BLW-7007", "Howden", "Blower-2", None,
     "1.1.0", None, 0, 0, ""),                    # warranty 0 / active 0 / notes ''
    # singletons
    (109, "erp", "SN-12009", "GEN-9009", "Caterpillar", "Gen-500", "Plant-E",
     "7.0.0", "2016-05-19", None, 1, None),
    (310, "iot", "SN-13010", None, "KSB", "Hydro-3", None,
     None, None, None, None, None),
]

# entity structure used only to generate coherent telemetry
_ENTITY_OF = {
    101: "E1", 201: "E1", 301: "E1",
    102: "E2", 202: "E2",
    103: "E3", 303: "E3",
    104: "E4",
    205: "E5", 305: "E5",
    106: "E6",
    207: "E7", 208: "E7",
    109: "E8",
    310: "E9",
    311: "E10",
}
_ENTITY_KIND = {
    "E1": "pump", "E2": "compressor", "E3": "valve", "E4": "motor",
    "E5": "fan", "E6": "pump", "E7": "blower", "E8": "generator",
    "E9": "hydro", "E10": "pump",
}
# per-archive telemetry jitter scale (default 0.05); E3's iot copy has drifted,
# E6/E10 are pinned tightly to their operating points
_DRIFT = {"303": 0.35, "106": 0.02, "311": 0.02}
# E6 sits close to E1 (same model line, similar duty); E10 is its midpoint,
# which produces the p(101,311) > tau, p(106,311) > tau, p(101,106) << tau
# chain that naive threshold-CC clustering cannot resolve consistently
_E6_OFFSET = {"press_mean": 0.8, "cur_mean": 2.0, "vib_mean": 0.2, "temp_mean": 2.0}

WORK_ORDERS = [
    (1, 101, "Seal replacement", "closed", 1200.0, "2026-02-01"),
    (2, 201, "Vibration alarm", "open", 0.0, "2026-03-14"),
    (3, 301, "Firmware update", "open", 150.0, "2026-04-02"),
    (4, 102, "Oil change", "closed", 300.0, "2026-01-20"),
    (5, 202, "Inspection", "open", 80.0, "2026-05-11"),
    (6, 103, "Valve stuck", "open", 450.0, "2026-06-07"),
    (7, 303, "Sensor drift", "closed", 220.0, "2026-02-19"),
    (8, 104, "Bearing replace", "open", 900.0, "2026-07-01"),
    (9, 205, "Blade cleaning", "closed", 100.0, "2026-03-03"),
    (10, 305, "Remote reset", "open", 0.0, "2026-08-21"),
    (11, 207, "Belt replace", "open", 350.0, "2026-09-09"),
    (12, 208, "Noise check", "closed", 60.0, "2026-04-27"),
    (13, 106, "Commissioning", "closed", 500.0, "2026-02-14"),
    (14, 109, "Fuel service", "open", 700.0, "2026-10-05"),
    (15, 310, "Calibration", "open", 130.0, "2026-10-19"),
    (18, 311, "Retrofit kit", "open", 275.0, "2026-11-01"),
    (16, 999, "Legacy archive deleted upstream", "open", 75.0, "2026-05-30"),
    (17, 555, "Retired system reference", "closed", 410.0, "2025-12-15"),
]

CONSTRAINTS = [
    ("must_link", 207, 208, "MES duplicate confirmed by maintenance lead"),
    ("must_link", 103, 303, "Same valve confirmed via asset tag VLV-3003"),
    ("cannot_link", 101, 106, "Same model line but distinct physical units (ERP audit)"),
]

COSTS = (6.0, 1.5)  # false-merge cost, missed-merge cost
PRIORITIES = [("erp", 1), ("mes", 2), ("iot", 3)]


def seed(db_path, force=False):
    db_path = Path(db_path)
    if db_path.exists():
        if not force:
            raise FileExistsError(f"{db_path} exists - pass force=True to rebuild")
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_mod.initialize(db_path)

    rng = np.random.default_rng(SEED)
    entity_params = {e: make_entity_params(rng, k) for e, k in _ENTITY_KIND.items()}
    # E6 shares E1's model line and sits close to its operating point;
    # E10 is the midpoint between them (probability chain)
    entity_params["E6"] = {
        **entity_params["E1"],
        **{k: entity_params["E1"][k] + dv for k, dv in _E6_OFFSET.items()},
    }
    entity_params["E10"] = {
        k: (entity_params["E1"][k] + entity_params["E6"][k]) / 2.0
        if isinstance(entity_params["E1"][k], float)
        else entity_params["E1"][k]
        for k in entity_params["E1"]
    }

    conn = db_mod.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO source_priority(source_system, priority) VALUES (?, ?)",
            PRIORITIES,
        )
        conn.executemany(
            "INSERT INTO archives(archive_id, source_system, serial_no, asset_tag, "
            "vendor, model, location, firmware, install_date, warranty_months, "
            "active, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ARCHIVES,
        )
        telemetry_rows = []
        for (archive_id, *_rest) in ARCHIVES:
            ent = _ENTITY_OF[archive_id]
            params = jitter_params(
                rng, entity_params[ent], scale=_DRIFT.get(str(archive_id), 0.05)
            )
            n_points = int(rng.integers(48, 73))
            start = f"2026-{int(rng.integers(1, 3)):02d}-01"
            frame = sample_telemetry(rng, params, archive_id, n_points, start=start)
            telemetry_rows.extend(frame.itertuples(index=False, name=None))
        conn.executemany(
            "INSERT INTO telemetry(archive_id, ts, vibration, temperature, current, "
            "pressure) VALUES (?,?,?,?,?,?)",
            telemetry_rows,
        )
        conn.executemany(
            "INSERT INTO work_orders(wo_id, archive_id, entity_id, title, status, "
            "cost, opened_at) VALUES (?,?,NULL,?,?,?,?)",
            WORK_ORDERS,
        )
        conn.executemany(
            "INSERT INTO link_constraints(kind, archive_id_a, archive_id_b, reason) "
            "VALUES (?,?,?,?)",
            CONSTRAINTS,
        )
        conn.execute(
            "INSERT INTO costs(id, false_merge_cost, missed_merge_cost) VALUES (1,?,?)",
            COSTS,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return db_path
