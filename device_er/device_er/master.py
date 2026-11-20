"""Master-record construction.

Entity id = smallest member archive id.  Every attribute takes the
first non-NULL value in source-priority order.  NULL means *only*
None / NaN / NaT: the values 0, 0.0, False and '' are real business
values and are preserved.
"""

from __future__ import annotations

import pandas as pd

from .db import ATTRIBUTES


def is_null(value) -> bool:
    """True only for genuine NULLs; 0 / False / '' are real values."""
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def build_master_records(clusters, archives, source_priority, attributes=None):
    """Build one master record per entity cluster.

    ``clusters`` is a list of member-id lists; ``archives`` the archives
    DataFrame; ``source_priority`` maps source system -> priority (lower
    wins).  Returns a list of dicts sorted by entity id, each with the
    merged attributes plus per-attribute provenance (which archive
    supplied the value).
    """
    attributes = list(attributes or ATTRIBUTES)
    by_id = archives.set_index("archive_id")
    records = []
    for members in clusters:
        members = sorted(int(m) for m in members)
        entity_id = members[0]
        ordered = sorted(
            members,
            key=lambda a: (int(source_priority[by_id.loc[a, "source_system"]]), a),
        )
        record = {
            "entity_id": entity_id,
            "members": members,
            "member_count": len(members),
        }
        provenance = {}
        for attr in attributes:
            value, supplier = None, None
            for archive_id in ordered:
                candidate = by_id.loc[archive_id, attr]
                if not is_null(candidate):
                    value, supplier = candidate, archive_id
                    break
            record[attr] = value
            provenance[attr] = supplier
        record["provenance"] = provenance
        records.append(record)
    records.sort(key=lambda r: r["entity_id"])
    return records
