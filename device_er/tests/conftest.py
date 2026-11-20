import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
for path in (str(PROJECT_ROOT), str(WORKSPACE)):
    if path not in sys.path:
        sys.path.insert(0, path)

#: tiny but real tsfresh calculator set for fast end-to-end tests
TINY_FC = {
    "mean": None,
    "standard_deviation": None,
    "maximum": None,
    "minimum": None,
    "quantile": [{"q": 0.25}, {"q": 0.75}],
    "linear_trend": [{"attr": "slope"}, {"attr": "intercept"}],
    "autocorrelation": [{"lag": 1}],
    "root_mean_square": None,
}


@pytest.fixture()
def tiny_fc():
    return dict(TINY_FC)


@pytest.fixture()
def runtime(tmp_path):
    """A fresh runtime root with a seeded database."""
    from device_er.seed import seed

    root = tmp_path / "rt"
    (root / "data").mkdir(parents=True)
    (root / "model").mkdir(parents=True)
    db_path = seed(root / "data" / "device_er.db")
    return {"root": root, "db": db_path, "model": root / "model" / "pairwise_model.pkl"}
