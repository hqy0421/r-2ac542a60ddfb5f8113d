"""Native tsfresh feature extraction (failure domain: ``extraction``)."""

from __future__ import annotations

import pandas as pd

from .errors import ExtractionError

#: curated, deterministic tsfresh calculator set (real extraction, no
#: fabricated features).  Compact enough for fast, reproducible runs.
CURATED_FC_PARAMETERS = {
    "mean": None,
    "median": None,
    "standard_deviation": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "minimum": None,
    "maximum": None,
    "quantile": [{"q": 0.25}, {"q": 0.75}],
    "linear_trend": [
        {"attr": "slope"},
        {"attr": "intercept"},
        {"attr": "rvalue"},
    ],
    "fft_coefficient": [
        {"coeff": 0, "attr": "abs"},
        {"coeff": 1, "attr": "abs"},
        {"coeff": 2, "attr": "abs"},
        {"coeff": 3, "attr": "abs"},
    ],
    "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    "c3": [{"lag": 1}, {"lag": 2}],
    "cid_ce": [{"normalize": True}],
    "sample_entropy": None,
    "longest_strike_above_mean": None,
    "number_peaks": [{"n": 1}, {"n": 3}],
    "mean_abs_change": None,
    "mean_change": None,
    "absolute_sum_of_changes": None,
    "root_mean_square": None,
    "agg_autocorrelation": [{"f_agg": "mean", "maxlag": 40}],
    "number_crossing_m": [{"m": 0}],
    "spkt_welch_density": [{"coeff": 2}],
    "ratio_beyond_r_sigma": [{"r": 1}],
    "abs_energy": None,
    "mean_second_derivative_central": None,
}


def extract_archive_features(
    telemetry: pd.DataFrame,
    archive_ids,
    fc_parameters=None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the real tsfresh extraction for every requested archive.

    Returns a DataFrame indexed by ``archive_id`` with one column per
    (kind, calculator) feature.  Archives without any telemetry row get
    an all-NaN row (handled later by model-side imputation) - but an
    empty telemetry frame overall, or any tsfresh failure, raises
    :class:`ExtractionError` so the caller keeps the old generation.
    """
    ids = [int(a) for a in archive_ids]
    if not ids:
        raise ExtractionError("no archives supplied for feature extraction")
    frame = telemetry[telemetry["archive_id"].isin(ids)].copy()
    if frame.empty:
        raise ExtractionError(
            "telemetry table has no rows for the current archives - "
            "refusing to resolve entities without native features"
        )

    from tsfresh import extract_features  # local import: heavy module

    params = dict(fc_parameters or CURATED_FC_PARAMETERS)
    try:
        features = extract_features(
            frame,
            column_id="archive_id",
            column_sort="ts",
            default_fc_parameters=params,
            n_jobs=n_jobs,
            disable_progressbar=True,
            show_warnings=False,
        )
    except Exception as exc:
        raise ExtractionError(f"tsfresh extraction failed: {exc}") from exc

    if features is None or features.shape[1] == 0:
        raise ExtractionError("tsfresh returned an empty feature matrix")

    features = features.reindex(ids)  # archives without telemetry -> NaN row
    features.index.name = "archive_id"
    features = features.sort_index()
    return features
