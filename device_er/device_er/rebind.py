"""Rebinding of repair work orders to resolved entities.

Only the ``entity_id`` column is ever rewritten; every other business
value (original archive reference, title, status, cost, dates) stays
byte-identical.  A work order whose source archive no longer exists in
the current archives is rebound to NULL.
"""

from __future__ import annotations

from .master import is_null


def compute_rebinding(work_orders, archive_to_entity, current_archive_ids):
    """Pure function: list of per-work-order rebind updates."""
    updates = []
    for wo in work_orders:
        archive_id = wo.get("archive_id")
        if is_null(archive_id) or int(archive_id) not in current_archive_ids:
            new_entity = None  # source archive disappeared -> NULL
        else:
            new_entity = int(archive_to_entity[int(archive_id)])
        old = wo.get("entity_id")
        updates.append(
            {
                "wo_id": int(wo["wo_id"]),
                "archive_id": None if is_null(archive_id) else int(archive_id),
                "old_entity_id": None if is_null(old) else int(old),
                "new_entity_id": new_entity,
            }
        )
    return updates


def changed(updates):
    return [u for u in updates if u["old_entity_id"] != u["new_entity_id"]]


def apply_rebinding(conn, updates):
    """Rewrite work_orders.entity_id inside the caller's transaction."""
    conn.executemany(
        "UPDATE work_orders SET entity_id = ? WHERE wo_id = ?",
        [(u["new_entity_id"], u["wo_id"]) for u in updates],
    )
