"""Work-order rebinding semantics."""

from device_er.rebind import changed, compute_rebinding


def _wos():
    return [
        {"wo_id": 1, "archive_id": 201, "entity_id": None},
        {"wo_id": 2, "archive_id": 999, "entity_id": 5},   # disappeared source
        {"wo_id": 3, "archive_id": None, "entity_id": None},
        {"wo_id": 4, "archive_id": 301, "entity_id": 301},  # stale, must be rebound
    ]


def test_rebind_to_min_member_entity():
    archive_to_entity = {101: 101, 201: 101, 301: 101}
    updates = compute_rebinding(_wos(), archive_to_entity, {101, 201, 301})
    by_id = {u["wo_id"]: u for u in updates}
    assert by_id[1]["new_entity_id"] == 101
    assert by_id[4]["new_entity_id"] == 101      # stale 301 rebound to cluster id
    assert by_id[4]["old_entity_id"] == 301


def test_disappeared_source_archive_becomes_null():
    updates = compute_rebinding(_wos(), {101: 101, 201: 101, 301: 101},
                                {101, 201, 301})
    by_id = {u["wo_id"]: u for u in updates}
    assert by_id[2]["new_entity_id"] is None     # archive 999 gone -> NULL
    assert by_id[2]["old_entity_id"] == 5
    assert by_id[3]["new_entity_id"] is None     # no source reference at all


def test_changed_detection():
    updates = compute_rebinding(_wos(), {101: 101, 201: 101, 301: 101},
                                {101, 201, 301})
    assert {u["wo_id"] for u in changed(updates)} == {1, 2, 4}
