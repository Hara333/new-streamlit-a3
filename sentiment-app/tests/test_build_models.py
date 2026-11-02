"""build_models.py against the pipeline's result files: the store it writes must equal the store
shipped with the app (ST-B-xx). Needs the results folder of the run; without COS_RESULTS the
module is skipped."""
import json
import os
import sys

import numpy as np
import pytest

from conftest import APP_DIR, model_keys

RESULTS = os.environ.get("COS_RESULTS")
LOGS = os.environ.get("COS_LOGS")

if not RESULTS:
    pytest.skip("set COS_RESULTS to the results folder of a pipeline run", allow_module_level=True)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    """build_models.main() run in this process (so coverage sees it) with the arguments it would get
    on the command line."""
    import importlib.util
    out = tmp_path_factory.mktemp("models")
    spec = importlib.util.spec_from_file_location("build_models", APP_DIR / "build_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["build_models.py", "--results", RESULTS, "--out", str(out)] + (["--logs", LOGS] if LOGS else [])
    saved = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = saved
    return out


def test_st_b_01_the_rebuilt_catalog_equals_the_shipped_one(rebuilt):
    a = json.loads((rebuilt / "catalog.json").read_text(encoding="utf-8"))
    b = json.loads((APP_DIR / "models" / "catalog.json").read_text(encoding="utf-8"))
    assert a == b


@pytest.mark.parametrize("key", model_keys())
def test_st_b_02_every_rebuilt_weight_file_equals_the_shipped_one(rebuilt, key):
    with np.load(rebuilt / f"{key}.npz", allow_pickle=False) as a, np.load(APP_DIR / "models" / f"{key}.npz", allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            assert a[name].dtype == b[name].dtype and a[name].shape == b[name].shape, (key, name)
            assert np.array_equal(a[name], b[name]), (key, name)


def test_st_b_03_the_store_holds_nothing_but_the_fourteen_models_and_the_catalog(rebuilt):
    assert sorted(p.name for p in rebuilt.iterdir()) == sorted([f"{k}.npz" for k in model_keys()] + ["catalog.json"])
