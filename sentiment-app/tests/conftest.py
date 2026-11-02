"""Fixtures for the app's tests.

The functions of app.py are tested without a running page: the part of the file above the page
code is executed once into a namespace (Streamlit's cache decorators work in bare mode). The user
interface tests run the whole file under Streamlit's AppTest with a small stand-in encoder that has
bge-large's output width, so no 1.3 GB download is needed. The golden files under tests/golden hold
the expected readings, computed by the training pipeline's own code (tests/tools/make_golden.py).
"""
import base64
import json
import os
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
APP_FILE = APP_DIR / "app.py"
GOLDEN = Path(__file__).resolve().parent / "golden"
PAGE_MARKER = "# ---------- page ----------"


def pytest_addoption(parser):
    parser.addoption("--real-encoder", action="store_true", default=False,
                     help="also run the tests that need the real bge-large encoder (downloads or uses the cache)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: user interface and load tests, a few minutes")
    config.addinivalue_line("markers", "real_encoder: needs the real bge-large-en-v1.5 weights")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--real-encoder"):
        return
    skip = pytest.mark.skip(reason="run with --real-encoder")
    for item in items:
        if "real_encoder" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def app():
    """The app's functions and constants, without running the page."""
    src = APP_FILE.read_text(encoding="utf-8")
    ns = {"__file__": str(APP_FILE), "__name__": "app_under_test"}
    exec(compile(src[: src.index(PAGE_MARKER)], "app.py", "exec"), ns)
    return ns


@pytest.fixture(scope="session")
def catalog(app):
    return app["catalog"]()


@pytest.fixture(scope="session")
def golden_readings():
    g = json.loads((GOLDEN / "readings.json").read_text(encoding="utf-8"))
    g["X"] = [np.frombuffer(base64.b64decode(v), "<f4").astype(np.float64) for v in g["vectors"]]
    return g


@pytest.fixture(scope="session")
def golden_words():
    return json.loads((GOLDEN / "words.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden_cleaning():
    return json.loads((GOLDEN / "cleaning.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fake_encoder_dir(tmp_path_factory):
    """A one-layer BERT with hidden size 1,024 (so the pooled vector has the app's 2,048 values) and
    a vocabulary of the tokens w0..w599 plus single letters. Random weights: the readings it
    produces mean nothing, which is fine for tests of the interface and of the windowing."""
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertModel, BertTokenizerFast
    d = tmp_path_factory.mktemp("fake_bge")
    vocab = d / "vocab.txt"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(600)] + list("abcdefghijklmnopqrstuvwxyz")
    vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    tok.save_pretrained(d)
    import torch
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(tokens), hidden_size=1024, num_hidden_layers=1, num_attention_heads=16, intermediate_size=256, max_position_embeddings=512)
    BertModel(cfg).save_pretrained(d)
    return d


@pytest.fixture(scope="session")
def app_with_fake_encoder(fake_encoder_dir):
    """The app file itself, with the encoder pointed at the stand-in through the environment
    variable the app reads (SENTIMENT_ENCODER), so coverage is measured on app.py."""
    os.environ["SENTIMENT_ENCODER"] = str(fake_encoder_dir)
    return APP_FILE


@pytest.fixture(scope="session")
def no_encoder_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("nowhere") / "no_such_model"


@pytest.fixture
def apptest(app_with_fake_encoder):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(app_with_fake_encoder), default_timeout=180).run()


def model_keys():
    return [m["key"] for m in json.loads((APP_DIR / "models" / "catalog.json").read_text(encoding="utf-8"))]
