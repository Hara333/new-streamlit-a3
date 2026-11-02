"""Does the app give the pipeline's answer? The golden files hold what the training pipeline's own
torch heads and word logit computed for 47 real encoder vectors and twelve sentences; the app must
reproduce them (ST-P-xx). These double as the regression suite: a change to the arithmetic, the
export or the model store shows up here first."""
import numpy as np
import pytest

from conftest import model_keys

NETWORK_KEYS = [k for k in model_keys() if not k.startswith("words_")]
WORD_KEYS = [k for k in model_keys() if k.startswith("words_")]


@pytest.mark.parametrize("key", NETWORK_KEYS)
def test_st_p_01_network_readings_match_the_pipeline(app, catalog, golden_readings, key):
    model, meta = app["load_model"](key), catalog[key]
    want = golden_readings["models"][key]
    worst, n = 0.0, 0
    for i, x in enumerate(golden_readings["X"]):
        out = app["run_head"](model, meta, x)["out"]
        for reading, g in want.items():
            p = out[reading]["p"]
            assert out[reading]["classes"] == g["classes"]
            assert int(np.argmax(p)) == int(np.argmax(g["p"][i])), (key, reading, golden_readings["texts"][i][:50])
            worst = max(worst, float(np.abs(p - np.array(g["p"][i])).max()))
            n += 1
    assert worst < 1e-5, f"{key}: largest probability difference {worst:.2e} over {n} readings"


def test_st_p_02_a_known_headline_reads_negative_on_every_fingpt_model(app, catalog, golden_readings):
    i = golden_readings["texts"].index("Chip stocks fell hard on the worsening outlook for the global consumer.")
    for key in ("fingpt_network", "fingpt_network_sentfin_all", "fingpt_network_sentfin_mistakes", "fingpt_network_sentfin_grpo", "fingpt_linear"):
        out = app["run_head"](app["load_model"](key), catalog[key], golden_readings["X"][i])["out"]["sentiment"]
        assert out["classes"][int(np.argmax(out["p"]))] == "negative" and out["p"].max() > 0.99, key


def test_st_p_03_the_dialogue_models_read_sentiment_and_emotion_of_an_angry_turn(app, catalog, golden_readings):
    text = next(t for t in golden_readings["texts"] if t.startswith("it made me mad when i saw a dent"))
    x = golden_readings["X"][golden_readings["texts"].index(text)]
    for key in ("dialogue_network", "dialogue_network_fingpt", "dialogue_network_fingpt_sentfin_all", "dialogue_linear"):
        out = app["run_head"](app["load_model"](key), catalog[key], x)["out"]
        assert out["sentiment"]["classes"][int(np.argmax(out["sentiment"]["p"]))] == "negative", key
        top = out["emotion"]["classes"][int(np.argmax(out["emotion"]["p"]))]
        assert top in ("furious", "angry", "annoyed"), (key, top)


@pytest.mark.parametrize("key", WORD_KEYS)
def test_st_p_04_word_readings_match_the_pipeline_exactly(app, catalog, golden_words, key):
    model, meta = app["load_model"](key), catalog[key]
    g = golden_words["models"][key]
    assert meta["readings"][0]["classes"] == g["classes"]
    for text, row in zip(golden_words["cleaned"], g["rows"]):
        res = app["run_words"](model, meta, text)
        assert [w["word"] for w in res["words"]] == row["words"], text
        assert [w["word"] for w in res["words"] if w["j"] is None] == row["unknown"], text
        assert np.abs(res["p"] - np.array(row["p"])).max() < 1e-9, text
        for w in res["words"]:
            if w["j"] is not None:
                assert np.abs(w["p"] - np.array(row["per_word"][w["word"]])).max() < 1e-9


def test_st_p_05_the_app_cleans_the_word_texts_as_the_pipeline_did(app, golden_words):
    assert [app["clean"](t) for t in golden_words["texts"]] == golden_words["cleaned"]


def test_st_p_06_unknown_word_and_no_word_cases_are_flagged(app, catalog, golden_words):
    model, meta = app["load_model"]("words_sentfin"), catalog["words_sentfin"]
    res = app["run_words"](model, meta, "Profits collapsed and the zzzunknownword board resigned")
    assert "zzzunknownword" in [w["word"] for w in res["words"] if w["j"] is None]
    none = app["run_words"](model, meta, "qwertyuiop zxcvbnm")
    assert none["known"] == 0 and len(none["words"]) == 2
    empty = app["run_words"](model, meta, "12345 !!! $$$")
    assert empty["words"] == [] and empty["known"] == 0
