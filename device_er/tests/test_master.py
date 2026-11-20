"""Master-record NULL semantics: 0 / FALSE / '' are values, not NULLs."""

import pandas as pd
import pytest

from device_er.master import build_master_records, is_null


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, pd.NaT])
def test_true_nulls(value):
    assert is_null(value)


@pytest.mark.parametrize("value", [0, 0.0, False, "", "0", 0.0j])
def test_zero_false_empty_string_are_values(value):
    assert not is_null(value)


def _archives():
    return pd.DataFrame(
        [
            # erp (priority 1): mostly NULLs
            dict(archive_id=10, source_system="erp", serial_no=None, asset_tag=None,
                 vendor=None, model=None, location=None, firmware=None,
                 install_date=None, warranty_months=pd.NA, active=pd.NA, notes=None),
            # mes (priority 2): zero / false / empty-string values
            dict(archive_id=20, source_system="mes", serial_no=None, asset_tag="TAG-1",
                 vendor=None, model="M", location=None, firmware=None,
                 install_date=None, warranty_months=0, active=0, notes=""),
            # iot (priority 3): real values that must lose to mes
            dict(archive_id=30, source_system="iot", serial_no="SN-9", asset_tag="TAG-9",
                 vendor="V", model="M9", location="L9", firmware="f9",
                 install_date="2020-01-01", warranty_months=99, active=1, notes="iot"),
        ]
    )


def test_first_non_null_by_source_priority():
    archives = _archives()
    prio = {"erp": 1, "mes": 2, "iot": 3}
    (rec,) = build_master_records([[30, 10, 20]], archives, prio)
    assert rec["entity_id"] == 10  # min member id
    assert rec["asset_tag"] == "TAG-1"      # mes wins over iot
    assert rec["serial_no"] == "SN-9"       # only iot has it
    assert rec["warranty_months"] == 0      # 0 is a value, beats iot's 99
    assert rec["active"] == 0               # FALSE is a value
    assert rec["notes"] == ""               # empty string is a value
    assert rec["firmware"] == "f9"
    assert rec["provenance"]["warranty_months"] == 20
    assert rec["provenance"]["notes"] == 20
    assert rec["provenance"]["serial_no"] == 30


def test_all_null_attribute_stays_null():
    archives = _archives()
    prio = {"erp": 1, "mes": 2, "iot": 3}
    (rec,) = build_master_records([[10]], archives, prio)
    assert rec["serial_no"] is None
    assert rec["provenance"]["serial_no"] is None
