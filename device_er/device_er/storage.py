"""Atomic, re-openable generation persistence (failure domain: ``save``).

A *generation* is a complete, self-contained snapshot of one resolution
run: native features, pairwise probabilities, partition, master
records, rebound work orders, the model that was used, the run
configuration, the human-readable report and a manifest with SHA-256
checksums of every file.

Promotion protocol (any failure keeps the previous generation live):

1. write every artefact into ``generations/.tmp-<gid>-<pid>/``
2. write ``manifest.json`` and verify it by re-reading the tmp dir
3. atomically rename the tmp dir to ``generations/<gid>/``
4. in ONE SQLite transaction: apply the work-order rebinding, register
   the generation, mark it current
5. best-effort ``CURRENT`` pointer file for humans

If anything throws before step 4 commits, the database is untouched
and the temporary/promoted directory is removed best-effort.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import db as db_mod
from .errors import SaveError
from .rebind import apply_rebinding

GENERATION_PREFIX = "gen_"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    import numpy as np

    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN -> None
        return None
    return str(value)


@dataclass
class Generation:
    """A re-opened, checksum-verified generation."""

    generation_id: str
    path: Path
    manifest: dict
    report: dict
    features: pd.DataFrame
    pairs: pd.DataFrame
    partition: dict
    masters: list
    work_orders_after: pd.DataFrame
    work_order_changes: pd.DataFrame


class GenerationStore:
    def __init__(self, root):
        self.root = Path(root)

    # ------------------------------------------------------------ ids
    def existing_ids(self):
        if not self.root.exists():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith(GENERATION_PREFIX)
        )

    def next_id(self):
        ids = self.existing_ids()
        if not ids:
            return f"{GENERATION_PREFIX}0001"
        last = max(int(g[len(GENERATION_PREFIX):]) for g in ids)
        return f"{GENERATION_PREFIX}{last + 1:04d}"

    # ------------------------------------------------------------ save
    def save(
        self,
        *,
        generation_id,
        artifacts: dict,
        manifest_extra: dict,
        conn,
        rebind_updates,
        registry_entry,
    ):
        """Write + verify + promote a generation, then commit DB changes.

        ``artifacts`` maps filename -> (kind, payload) with kind in
        ``json`` / ``csv`` / ``text`` / ``bytes``.
        """
        gid = generation_id
        final = self.root / gid
        tmp = self.root / f".tmp-{gid}-{os.getpid()}"
        promoted = False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise SaveError(f"generation directory already exists: {final}")
            tmp.mkdir(parents=True, exist_ok=False)
            files = {}
            for name, (kind, payload) in artifacts.items():
                target = tmp / name
                if kind == "json":
                    target.write_text(
                        json.dumps(payload, indent=2, ensure_ascii=False,
                                   default=json_default)
                    )
                elif kind == "csv":
                    payload.to_csv(target, index=False)
                elif kind == "text":
                    target.write_text(str(payload))
                elif kind == "bytes":
                    target.write_bytes(payload)
                else:  # pragma: no cover - defensive
                    raise SaveError(f"unknown artifact kind {kind!r}")
                files[name] = {
                    "sha256": _sha256(target),
                    "bytes": target.stat().st_size,
                }
            manifest = {
                "generation_id": gid,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": files,
                **manifest_extra,
            }
            (tmp / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False,
                           default=json_default)
            )
            self._verify_dir(tmp, manifest)  # re-read & check before promoting

            os.rename(tmp, final)
            promoted = True

            try:
                conn.execute("BEGIN IMMEDIATE")
                apply_rebinding(conn, rebind_updates)
                db_mod.register_generation(conn, registry_entry)
                db_mod.set_current_generation(conn, gid)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                shutil.rmtree(final, ignore_errors=True)
                raise SaveError(
                    f"database commit for generation {gid} failed; "
                    f"previous generation kept: {exc}"
                ) from exc

            try:  # human-facing pointer; the DB meta table is authoritative
                (self.root / "CURRENT").write_text(gid + "\n")
            except OSError:
                pass
            return gid
        except SaveError:
            shutil.rmtree(tmp, ignore_errors=True)
            if promoted:
                shutil.rmtree(final, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            if promoted:
                shutil.rmtree(final, ignore_errors=True)
            raise SaveError(
                f"failed to persist generation {gid}; previous generation "
                f"kept: {exc}"
            ) from exc

    # ------------------------------------------------------------ reopen
    @staticmethod
    def _verify_dir(path: Path, manifest: dict):
        for name, info in manifest["files"].items():
            target = path / name
            if not target.exists():
                raise SaveError(f"manifest entry missing on disk: {name}")
            if _sha256(target) != info["sha256"]:
                raise SaveError(f"checksum mismatch for {name}")

    def reopen(self, generation_id) -> Generation:
        path = self.root / generation_id
        if not path.is_dir():
            raise SaveError(f"generation not found: {path}")
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise SaveError(f"generation {generation_id} has no manifest")
        try:
            manifest = json.loads(manifest_path.read_text())
            self._verify_dir(path, manifest)
            report = json.loads((path / "report.json").read_text())
            features = pd.read_csv(path / "features.csv")
            pairs = pd.read_csv(path / "pair_probabilities.csv")
            partition = json.loads((path / "partition.json").read_text())
            masters = json.loads((path / "master_records.json").read_text())
            wo_after = pd.read_csv(path / "work_orders_after.csv")
            wo_changes = pd.read_csv(path / "work_order_changes.csv")
        except SaveError:
            raise
        except Exception as exc:
            raise SaveError(
                f"generation {generation_id} is corrupt: {exc}"
            ) from exc
        return Generation(
            generation_id=generation_id,
            path=path,
            manifest=manifest,
            report=report,
            features=features,
            pairs=pairs,
            partition=partition,
            masters=masters,
            work_orders_after=wo_after,
            work_order_changes=wo_changes,
        )

    def reopen_current(self, conn) -> Generation:
        gid = db_mod.current_generation(conn)
        if gid is None:
            raise SaveError("no current generation recorded in the database")
        return self.reopen(gid)
