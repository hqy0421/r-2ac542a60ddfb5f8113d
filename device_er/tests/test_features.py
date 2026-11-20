"""Native tsfresh extraction is real and deterministic."""

import numpy as np
import pandas as pd
import pytest

from device_er.errors import ExtractionError
from device_er.features import extract_archive_features
from device_er.telemetry_gen import make_entity_params, sample_telemetry


@pytest.fixture()
def telemetry():
    rng = np.random.default_rng(42)
    params = make_entity_params(rng, "pump")
    frames = [
        sample_telemetry(rng, params, 1, 50),
        sample_telemetry(rng, params, 2, 60),
    ]
    return pd.concat(frames, ignore_index=True)


def test_real_extraction_shape_and_values(telemetry, tiny_fc):
    X = extract_archive_features(telemetry, [1, 2], fc_parameters=tiny_fc)
    assert list(X.index) == [1, 2]
    assert X.shape[1] > 0
    # a natively computed feature matches a hand computation
    vib = telemetry.loc[telemetry.archive_id == 1, "vibration"]
    assert X.loc[1, "vibration__mean"] == pytest.approx(vib.mean())
    assert X.loc[2, "vibration__maximum"] == pytest.approx(
        telemetry.loc[telemetry.archive_id == 2, "vibration"].max()
    )


def test_extraction_is_deterministic(telemetry, tiny_fc):
    X1 = extract_archive_features(telemetry, [1, 2], fc_parameters=tiny_fc)
    X2 = extract_archive_features(telemetry, [1, 2], fc_parameters=tiny_fc)
    pd.testing.assert_frame_equal(X1, X2)


def test_archive_without_telemetry_gets_nan_row(telemetry, tiny_fc):
    X = extract_archive_features(telemetry, [1, 2, 3], fc_parameters=tiny_fc)
    assert X.loc[3].isna().all()


def test_empty_telemetry_raises(telemetry, tiny_fc):
    with pytest.raises(ExtractionError):
        extract_archive_features(telemetry.iloc[0:0], [1], fc_parameters=tiny_fc)
