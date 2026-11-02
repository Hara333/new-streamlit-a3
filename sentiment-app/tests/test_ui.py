"""The page under Streamlit's AppTest with the stand-in encoder: the use cases of the test plan
walked end to end, every model, every switch (ST-U-xx). Readings are meaningless with the stand-in
encoder; what is checked is the flow, the elements shown and the absence of exceptions."""
import time

import pytest

from conftest import model_keys

pytestmark = pytest.mark.slow

SENTENCE = "Shares jumped after the company beat earnings forecasts, but the outlook stayed cautious."
UNKNOWN = "Profits collapsed and the zzzunknownword board resigned"
KEYS = model_keys()


def predict(at, key, text):
    at.selectbox[0].set_value(key).run()
    at.text_area[0].input(text)
    at.button[0].click().run()
    return at


def labels(at):
    return [m.label for m in at.metric]


class TestPageLoad:
    def test_st_u_01_dropdown_lists_the_fourteen_models_with_their_method_in_brackets(self, apptest, catalog):
        at = apptest
        picker = at.selectbox[0]
        assert len(picker.options) == 14 and picker.value == "fingpt_network"
        shown = list(picker.options)  # AppTest exposes the formatted labels as the options
        assert shown[0] == "FinGPT Network [deep learning]"
        assert all(s.endswith("]") and " [" in s for s in shown)
        assert shown == [f"{catalog[k]['name']} [{catalog[k]['tag']}]" for k in KEYS]
        assert at.title[0].value == "Sentiment prediction"
        assert not at.exception

    def test_st_u_02_encoder_line_changes_with_the_kind_of_model(self, apptest):
        at = apptest
        assert "bge-large" in at.caption[0].value
        at.selectbox[0].set_value("words_sentfin").run()
        assert at.caption[0].value.startswith("Encoder: none")

    def test_st_u_03_predict_with_nothing_typed_warns_and_shows_no_result(self, apptest):
        at = apptest
        at.text_area[0].input("   ")
        at.button[0].click().run()
        assert [w.value for w in at.warning] == ["Type a sentence first."]
        assert not any(h.value == "Result" for h in at.subheader)

    def test_st_u_04_details_can_be_read_before_any_sentence(self, apptest):
        at = apptest
        at.toggle[0].set_value(True).run()
        text = "\n".join(m.value for m in at.markdown)
        assert "**What it is**" in text and "Stochastic gradient descent" in text
        assert len(at.get("plotly_chart")) == 1 and not at.exception


class TestReadASentence:
    @pytest.mark.parametrize("key", KEYS)
    def test_st_u_05_every_model_reads_a_sentence_and_shows_its_own_result(self, apptest, catalog, key):
        at = predict(apptest, key, SENTENCE)
        m = catalog[key]
        assert any(h.value == "Result" for h in at.subheader) and not at.exception
        got = labels(at)
        if key == "words_fingpt":
            assert got == ["Shade", "Polarity, the shades summed"]
        elif m["kind"] == "words":
            assert got == ["Sentiment"] if len(m["readings"][0]["classes"]) == 3 else ["Emotion"]
        else:
            assert got == ["Sentiment", "Emotion"][: len(m["readings"])]
        charts = len(at.get("plotly_chart"))
        assert charts == (1 if key == "words_fingpt" else len(got))  # the shades model draws its two metrics over one bar chart
        names = [t.label for t in at.toggle]
        assert names[:2] == ["Show the calculation", "Model details"]
        assert ("Draw the network for this sentence" in names) == bool(m["acts"])
        assert ("Show the [CLS] values" in names) == bool(m["encoder"])
        assert names[-1] == ("Show the [CLS] values" if m["encoder"] else "Model details")  # last on the page
        if m["kind"] == "words":
            assert len(at.dataframe) == 1  # the word table

    @pytest.mark.parametrize("key", KEYS)
    def test_st_u_06_every_switch_on_for_every_model_without_an_exception(self, apptest, catalog, key):
        at = predict(apptest, key, SENTENCE)
        for t in at.toggle:
            t.set_value(True)
        at.run()
        for t in at.toggle:  # the nested "Show all 2,048 values" switch appears once the calculation is open
            t.set_value(True)
        t0 = time.time()
        at.run()
        elapsed = time.time() - t0
        assert not at.exception, [e.value for e in at.exception]
        m = catalog[key]
        if m["kind"] == "words":
            assert len(at.dataframe) == 2 and len(at.toggle) == 2
        else:
            assert len(at.toggle) == 4 + (1 if m["acts"] else 0)
            # the 2,048 values, the standardisation, one table per hidden layer, two per reading at the output, the [CLS] values
            assert len(at.dataframe) == 3 + len(m["acts"]) + 2 * len(m["readings"])
            # one chart per reading, the training curve when the model has one, the drawn network for deep learning
            assert len(at.get("plotly_chart")) == len(m["readings"]) + (1 if m.get("curve") else 0) + (1 if m["acts"] else 0)
        assert elapsed < 5.0, f"a rerun with everything open took {elapsed:.1f} s"

    def test_st_u_06b_cls_table_shows_the_first_half_of_the_encoder_values(self, apptest):
        at = predict(apptest, "fingpt_network", SENTENCE)
        next(t for t in at.toggle if t.label == "Show the [CLS] values").set_value(True).run()
        df = at.dataframe[-1].value
        assert list(df.columns) == ["position", "value"] and list(df["position"]) == list(range(1024))
        assert any(c.value.startswith("The first 1,024 of the encoder's 2,048 values") for c in at.caption) and not at.exception

    def test_st_u_07_switching_model_keeps_the_sentence_and_rereads_it(self, apptest):
        at = predict(apptest, "fingpt_network", SENTENCE)
        assert labels(at) == ["Sentiment"]
        at.selectbox[0].set_value("dialogue_network").run()
        assert labels(at) == ["Sentiment", "Emotion"] and any(h.value == "Result" for h in at.subheader)
        at.selectbox[0].set_value("words_fingpt").run()
        assert labels(at) == ["Shade", "Polarity, the shades summed"]

    def test_st_u_08_editing_the_text_after_a_prediction_says_which_text_the_result_is_for(self, apptest):
        at = predict(apptest, "words_sentfin", SENTENCE)
        at.text_area[0].input("something else entirely").run()
        assert any(c.value == f'Result for: "{SENTENCE}"' for c in at.caption)

    def test_st_u_09_a_text_longer_than_one_window_is_read_in_several_windows(self, apptest):
        long_text = " ".join([SENTENCE] * 60)
        at = predict(apptest, "fingpt_network", long_text)
        at.toggle[0].set_value(True).run()
        step = next(m.value for m in at.markdown if m.value.startswith("**1. Encoder.**"))
        assert " windows and gave 2,048 values" in step and not at.exception

    @pytest.mark.parametrize("text", ["<script>alert(1)</script>", "{{ 7 * 7 }} ${x}", chr(0x1F600) * 50 + " profit", chr(0x5E02) + chr(0x5834) + " shares", "x" * 5000])
    def test_st_u_10_hostile_or_odd_input_is_shown_as_text_and_raises_nothing(self, apptest, text):
        at = predict(apptest, "words_sentfin", text)
        assert not at.exception
        at.text_area[0].input("edited").run()
        assert any(text[:40] in c.value for c in at.caption)


class TestWordModels:
    def test_st_u_11_an_unknown_word_is_reported_and_left_out(self, apptest):
        at = predict(apptest, "words_sentfin", UNKNOWN)
        # SEntFiN's vocabulary lacks "collapsed" and "resigned" as well, so three of the seven words are N/A
        assert [i.value for i in at.info] == ["3 of 7 words are N/A (never seen in training) and add nothing; the reading rests on the other 4."]
        table = at.dataframe[0].value
        rows = dict(zip(table["word"], table["trigger"]))
        assert rows["zzzunknownword"] == "N/A" and rows["collapsed"] == "N/A" and rows["profits"] != "N/A"

    def test_st_u_12_no_known_word_gives_no_reading(self, apptest):
        at = predict(apptest, "words_sentfin", "qwertyuiop zxcvbnm")
        assert labels(at) == ["Reading"] and at.metric[0].value == "N/A"
        assert at.warning[0].value.startswith("None of the 2 words was seen in training")
        at.toggle[0].set_value(True).run()
        assert any("the reading above is N/A" in c.value for c in at.caption)

    def test_st_u_12b_all_words_known_shows_the_sum_without_an_na_line(self, apptest):
        at = predict(apptest, "words_sentfin", "Shares fall as profit drops")
        assert not at.info and not at.warning
        at.toggle[0].set_value(True).run()
        assert not any(c.value.startswith("N/A, left out") for c in at.caption)
        assert list(at.dataframe[1].value["term"]) == ["intercept", "shares", "fall", "as", "profit", "drops", "score (sum)", "probability"]

    def test_st_u_13_no_letters_at_all(self, apptest):
        at = predict(apptest, "words_fingpt", "12345 !!! $$$")
        assert labels(at) == ["Reading"] and at.warning[0].value.startswith("No words to read")

    def test_st_u_14_the_shades_model_shows_the_nine_shades_and_their_polarity(self, apptest):
        at = predict(apptest, "words_fingpt", "Shares jumped after the beat")
        assert at.metric[0].label == "Shade" and at.metric[1].label == "Polarity, the shades summed"
        assert set(at.dataframe[0].value.columns) >= {"word", "texts in training", "trigger", "strong negative", "strong positive"}


FRESH_PROCESS_SCRIPT = """
import json, sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=120).run()
out = {"words": [], "errors": [], "exceptions": [], "result_shown": None}
for key in ("words_fingpt", "words_sentfin", "words_dialogue"):
    at.selectbox[0].set_value(key).run()
    at.text_area[0].input("Profits collapsed after the board resigned")
    at.button[0].click().run()
    out["words"].append([m.value for m in at.metric])
    out["exceptions"] += [e.value for e in at.exception]
at.selectbox[0].set_value("fingpt_network").run()
out["errors"] = [e.value[:60] for e in at.error]
out["exceptions"] += [e.value for e in at.exception]
out["result_shown"] = any(h.value == "Result" for h in at.subheader)
print("RESULT " + json.dumps(out))
"""


class TestWithoutAnEncoder:
    def test_st_u_15_word_models_never_need_the_encoder_and_a_network_fails_with_a_message(self, no_encoder_dir):
        # a fresh interpreter: Streamlit's resource cache is process wide, so the stand-in encoder
        # loaded by the other tests would otherwise be reused here and hide the failure
        import json
        import os
        import subprocess
        import sys
        from conftest import APP_FILE
        env = dict(os.environ, SENTIMENT_ENCODER=str(no_encoder_dir))
        run = subprocess.run([sys.executable, "-c", FRESH_PROCESS_SCRIPT, str(APP_FILE)], capture_output=True, text=True, timeout=600, env=env)
        line = next((l for l in run.stdout.splitlines() if l.startswith("RESULT ")), None)
        assert line, run.stdout[-2000:] + run.stderr[-2000:]
        out = json.loads(line[len("RESULT "):])
        assert all(out["words"]) and out["exceptions"] == []
        assert len(out["errors"]) == 1 and out["errors"][0].startswith("The encoder could not be loaded")
        assert out["result_shown"] is False
