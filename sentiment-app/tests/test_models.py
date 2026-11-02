"""The model store models/: the catalog and the weight files it points at, loaded the way the app
loads them (ST-F-xx)."""
import re

import numpy as np
import pytest

from conftest import APP_DIR, model_keys

EXPECTED = {
    "fingpt_network": ("network", ["leaky_relu", "tanh"], [256, 128], ["sentiment"], 557827),
    "fingpt_network_sentfin_all": ("network", ["leaky_relu", "tanh"], [256, 128], ["sentiment"], 557827),
    "fingpt_network_sentfin_mistakes": ("network", ["leaky_relu", "tanh"], [256, 128], ["sentiment"], 557827),
    "fingpt_network_sentfin_grpo": ("network", ["leaky_relu", "tanh"], [256, 128], ["sentiment"], 557827),
    "dialogue_network": ("network", ["relu", "silu", "elu"], [256, 128, 32], ["sentiment", "emotion"], 562723),
    "dialogue_network_fingpt": ("network", ["relu", "silu", "elu"], [256, 128, 32], ["sentiment", "emotion"], 562723),
    "dialogue_network_fingpt_sentfin_mistakes": ("network", ["relu", "silu", "elu"], [256, 128, 32], ["sentiment", "emotion"], 562723),
    "dialogue_network_fingpt_sentfin_all": ("network", ["relu", "silu", "elu"], [256, 128, 32], ["sentiment", "emotion"], 562723),
    "dialogue_network_fingpt_sentfin_grpo": ("network", ["relu", "silu", "elu"], [256, 128, 32], ["sentiment", "emotion"], 562723),
    "fingpt_linear": ("linear", [], [], ["sentiment"], 6147),
    "dialogue_linear": ("linear", [], [], ["sentiment", "emotion"], 71715),
    "words_fingpt": ("words", [], [], ["label"], 285309),
    "words_sentfin": ("words", [], [], ["label"], 24360),
    "words_dialogue": ("words", [], [], ["label"], 565312),
}
TAGS = {"network": ("deep learning",), "linear": ("linear",), "words": ("MNLM",)}


class TestCatalog:
    def test_st_f_01_fourteen_models_in_the_documented_order(self, catalog):
        assert list(catalog) == list(EXPECTED)

    def test_st_f_02_names_are_unique_and_every_tag_names_its_method(self, catalog):
        names = [m["name"] for m in catalog.values()]
        assert len(set(names)) == 14
        for key, m in catalog.items():
            kind = EXPECTED[key][0]
            assert m["tag"].startswith(TAGS[kind][0]), (key, m["tag"])
            reinforced = kind == "network" and key not in ("fingpt_network", "dialogue_network")
            assert ("reinforced" in m["tag"]) == reinforced, (key, m["tag"])
            if reinforced:
                assert ("twice" in m["tag"]) == key.startswith("dialogue_network_fingpt_sentfin")

    @pytest.mark.parametrize("key", list(EXPECTED))
    def test_st_f_03_kind_shape_readings_and_weight_count(self, catalog, key):
        kind, acts, widths, readings, params = EXPECTED[key]
        m = catalog[key]
        assert m["kind"] == kind and m["acts"] == acts and m["widths"] == widths
        assert [r["name"] for r in m["readings"]] == readings
        assert m["params"] == params
        assert m["encoder"] == (kind != "words")

    def test_st_f_04_class_lists_match_the_reading(self, catalog):
        for m in catalog.values():
            for r in m["readings"]:
                if r["name"] == "sentiment":
                    assert r["classes"] == ["negative", "neutral", "positive"]
                if r["name"] == "emotion":
                    assert len(r["classes"]) == 32 and len(set(r["classes"])) == 32
                assert r["temperature"] > 0

    def test_st_f_05_shades_model_maps_every_shade_to_a_polarity(self, catalog):
        m = catalog["words_fingpt"]
        classes = m["readings"][0]["classes"]
        assert len(classes) == 9 and set(m["polarity"]) == set(classes)
        assert set(m["polarity"].values()) == {"negative", "neutral", "positive"}
        assert all("polarity" not in catalog[k] or not catalog[k]["polarity"] for k in catalog if k != "words_fingpt")

    def test_st_f_06_details_sections_and_curves_are_well_formed(self, catalog):
        for key, m in catalog.items():
            titles = [s["title"] for s in m["details"]]
            assert titles and all(s["rows"] and all(len(r) == 2 for r in s["rows"]) for s in m["details"]), key
            assert titles[0] == "What it is"
            c = m.get("curve")
            if c:
                assert len(c["x"]) == len(c["y"]) > 1 and c["title"] and c["xlabel"] and c["ylabel"]
            if m["kind"] == "network" and "reinforced" not in m["tag"]:
                assert "Stochastic gradient descent" in titles
            if m["kind"] in ("linear", "words"):
                text = " ".join(b for s in m["details"] for _, b in s["rows"])
                assert "L-BFGS" in text and "no stochastic gradient descent" in text

    def test_st_f_07_catalog_is_plain_ascii_without_dashes_or_project_paths(self):
        s = (APP_DIR / "models" / "catalog.json").read_text(encoding="utf-8")
        assert all(ord(ch) < 128 for ch in s)
        assert chr(0x2014) not in s and chr(0x2013) not in s
        assert not re.search(r"/home/\w+|(?<![\w.:])/(?:[a-z_][\w.-]*/){3,}|ssh|@", s)  # no absolute path, host or login


class TestWeightFiles:
    @pytest.mark.parametrize("key", model_keys())
    def test_st_f_08_npz_loads_without_pickle_and_matches_the_catalog(self, app, catalog, key):
        m = catalog[key]
        with np.load(APP_DIR / "models" / f"{key}.npz", allow_pickle=False) as z:
            files = set(z.files)
            arrays = {k: z[k] for k in z.files}
        if m["kind"] == "words":
            assert files == {"vocab", "texts", "coef", "intercept"}
            vocab = arrays["vocab"].tobytes().decode("utf-8").split("\n")
            K, V = arrays["coef"].shape
            assert V == len(vocab) == m["vocabulary"] == len(arrays["texts"])
            assert K == len(m["readings"][0]["classes"]) == len(arrays["intercept"])
            assert vocab == sorted(vocab) and len(set(vocab)) == V
            assert all(re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w) for w in vocab)
            assert arrays["coef"].dtype == np.float32 and arrays["texts"].min() >= 1
            return
        assert arrays["mean"].shape == arrays["sd"].shape == (2048,) and arrays["sd"].min() > 0
        prev = 2048
        for i, w in enumerate(m["widths"]):
            assert arrays[f"W{i}"].shape == (w, prev) and arrays[f"b{i}"].shape == (w,)
            prev = w
        for r in m["readings"]:
            assert arrays[f"out_{r['name']}_W"].shape == (len(r["classes"]), prev)
            assert arrays[f"out_{r['name']}_b"].shape == (len(r["classes"]),)
        n = sum(a.size for k, a in arrays.items() if k not in ("mean", "sd"))
        assert n == m["params"]

    @pytest.mark.parametrize("key", model_keys())
    def test_st_f_09_load_model_returns_float64_arrays_in_the_app_layout(self, app, catalog, key):
        model = app["load_model"](key)
        m = catalog[key]
        if m["kind"] == "words":
            assert model["coef"].dtype == np.float64 and len(model["index"]) == len(model["vocab"])
            assert model["index"][model["vocab"][0]] == 0
        else:
            assert model["mean"].dtype == np.float64 and len(model["layers"]) == len(m["acts"])
            assert set(model["outs"]) == {r["name"] for r in m["readings"]}

    def test_st_f_10_adapted_fingpt_networks_differ_from_their_starting_network(self, app):
        base = app["load_model"]("fingpt_network")["layers"][0][0]
        for key in ("fingpt_network_sentfin_all", "fingpt_network_sentfin_mistakes", "fingpt_network_sentfin_grpo"):
            W = app["load_model"](key)["layers"][0][0]
            rel = np.linalg.norm(W - base) / np.linalg.norm(base)
            assert rel > 0.1, (key, rel)

    def test_st_f_11_store_size_is_small_enough_to_ship(self):
        total = sum(f.stat().st_size for f in (APP_DIR / "models").iterdir())
        assert total < 30 * 1024 * 1024
