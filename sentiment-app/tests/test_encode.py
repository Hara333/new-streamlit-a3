"""The encoder step with a stub in place of bge-large: the windowing of long texts at its boundaries
and the token-weighted average, plus the caching of a repeated sentence (ST-E-xx). The stub keeps
every window it is given, so the test can see exactly how a text was cut."""
import numpy as np
import pytest
import torch


class StubTokenizer:
    """One token per whitespace-separated word, so a text of n words has n tokens."""
    cls_token_id, sep_token_id = 101, 102

    def __call__(self, text, add_special_tokens=False, verbose=False):
        return {"input_ids": [1000 + i for i, _ in enumerate(text.split())]}


class StubEncoder:
    """Returns a last hidden state whose every token vector equals the window length, so the pooled
    [CLS; mean] of a window is a constant known in closed form."""
    hidden = 4

    def __init__(self):
        self.windows = []
        self.config = type("C", (), {"hidden_size": self.hidden})()

    def __call__(self, input_ids):
        ids = input_ids[0].tolist()
        self.windows.append(ids)
        value = float(len(ids))
        h = torch.full((1, len(ids), self.hidden), value)
        return type("R", (), {"last_hidden_state": h})()


@pytest.fixture
def stub(app, monkeypatch):
    enc = StubEncoder()
    monkeypatch.setitem(app, "encoder", lambda: (StubTokenizer(), enc))
    app["encode"].clear()
    return enc


def text_of(n):
    return " ".join(f"t{i}" for i in range(n))


@pytest.mark.parametrize("n_tokens, n_windows", [(1, 1), (509, 1), (510, 1), (511, 2), (988, 2), (989, 3), (1500, 4)])
def test_st_e_01_window_count_at_the_boundaries(app, stub, n_tokens, n_windows):
    x, n, windows = app["encode"](text_of(n_tokens))
    assert (n, windows) == (n_tokens, n_windows)
    assert len(stub.windows) == n_windows


def test_st_e_02_windows_are_510_tokens_long_overlap_by_32_and_carry_cls_and_sep(app, stub):
    app["encode"](text_of(989))
    bodies = [w[1:-1] for w in stub.windows]
    assert all(w[0] == 101 and w[-1] == 102 for w in stub.windows)
    assert [len(b) for b in bodies] == [510, 510, 33]
    assert bodies[0][-32:] == bodies[1][:32] and bodies[1][-32:] == bodies[2][:32]
    assert bodies[2][-1] == 1000 + 988


def test_st_e_03_pooled_vector_is_the_token_weighted_average_of_the_windows(app, stub):
    x, _, windows = app["encode"](text_of(600))
    lengths = [len(w) for w in stub.windows]
    assert lengths == [512, 124]
    want = sum(L * L for L in lengths) / sum(lengths)  # each window pools to its own length
    assert x.shape == (8,) and np.allclose(x, want)


def test_st_e_04_single_window_text_pools_cls_then_mean(app, stub):
    x, n, windows = app["encode"]("one two three")
    assert (n, windows) == (3, 1) and np.allclose(x, 5.0)


def test_st_e_05_the_same_sentence_is_encoded_once(app, stub):
    app["encode"](text_of(20))
    app["encode"](text_of(20))
    assert len(stub.windows) == 1


def test_st_e_06_a_different_sentence_is_a_cache_miss(app, stub):
    app["encode"](text_of(20))
    app["encode"](text_of(21))
    assert len(stub.windows) == 2


@pytest.mark.real_encoder
def test_st_e_07_real_encoder_reads_a_sentence_within_the_time_budget(app):
    import time
    t0 = time.time()
    x, n, windows = app["encode"]("Shares jumped after the company beat earnings forecasts.")
    first = time.time() - t0
    t0 = time.time()
    app["encode"]("The outlook stayed cautious despite the beat.")
    second = time.time() - t0
    assert x.shape == (2048,) and windows == 1 and 8 <= n <= 16
    assert second < 5.0, f"a cached encoder took {second:.1f} s for one sentence"
    print(f"first sentence (model load included) {first:.1f} s, second {second:.2f} s")
