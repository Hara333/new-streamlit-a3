"""The arithmetic of a reading and the small helpers behind the page, on synthetic weights where the
right answer is known in closed form (ST-M-xx)."""
import numpy as np
import pytest


class TestSoftmaxAndActivations:
    def test_st_m_01_softmax_sums_to_one_and_is_shift_invariant(self, app):
        p = app["softmax"](np.array([1.0, 2.0, 3.0]))
        assert p.sum() == pytest.approx(1.0) and np.allclose(p, app["softmax"](np.array([101.0, 102.0, 103.0])))
        assert np.allclose(app["softmax"](np.array([1000.0, -1000.0])), [1.0, 0.0])

    @pytest.mark.parametrize("name, x, want", [
        ("relu", [-2.0, 0.0, 3.0], [0.0, 0.0, 3.0]),
        ("leaky_relu", [-2.0, 0.0, 3.0], [-0.02, 0.0, 3.0]),
        ("tanh", [0.0, 1.0], [0.0, np.tanh(1.0)]),
        ("silu", [0.0, 1.0], [0.0, 1 / (1 + np.exp(-1.0))]),
        ("elu", [-1.0, 0.0, 2.0], [np.expm1(-1.0), 0.0, 2.0]),
        ("sigmoid", [0.0], [0.5]),
    ])
    def test_st_m_02_each_activation_at_its_boundary_and_on_both_sides(self, app, name, x, want):
        assert np.allclose(app["ACT"][name](np.array(x)), want)

    def test_st_m_03_elu_does_not_overflow_on_large_positive_input(self, app):
        with np.errstate(over="raise"):
            out = app["ACT"]["elu"](np.array([1000.0, -1000.0]))
        assert out[0] == 1000.0 and out[1] == pytest.approx(-1.0)


def tiny_network():
    """2 inputs -> 2 relu units -> one reading of 2 classes, with weights chosen by hand."""
    model = {"mean": np.array([1.0, -1.0]), "sd": np.array([2.0, 0.5]),
             "layers": [(np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([0.0, 0.5]), "relu")],
             "outs": {"sentiment": (np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([0.0, 0.0]))}}
    meta = {"readings": [{"name": "sentiment", "classes": ["a", "b"], "temperature": 2.0}], "acts": ["relu"]}
    return model, meta


class TestRunHead:
    def test_st_m_04_standardisation_then_layers_then_temperature(self, app):
        model, meta = tiny_network()
        res = app["run_head"](model, meta, np.array([3.0, -1.5]))
        assert np.allclose(res["z"], [1.0, -1.0])           # (3-1)/2, (-1.5+1)/0.5
        pre, h = res["layers"][0]
        assert np.allclose(pre, [1.0, 1.5]) and np.allclose(h, [1.0, 1.5])
        out = res["out"]["sentiment"]
        assert np.allclose(out["logits"], [2.0, 3.0])
        assert np.allclose(out["p"], app["softmax"](np.array([2.0, 3.0]) / 2.0))
        assert out["classes"] == ["a", "b"] and out["T"] == 2.0

    def test_st_m_05_relu_cuts_a_negative_unit(self, app):
        model, meta = tiny_network()
        res = app["run_head"](model, meta, np.array([-3.0, -1.0]))
        pre, h = res["layers"][0]
        assert pre[0] == -2.0 and h[0] == 0.0

    def test_st_m_06_a_head_without_hidden_layers_reads_the_standardised_input_directly(self, app):
        model = {"mean": np.zeros(3), "sd": np.ones(3), "layers": [], "outs": {"sentiment": (np.eye(3), np.zeros(3))}}
        meta = {"readings": [{"name": "sentiment", "classes": ["n", "u", "p"], "temperature": 1.0}], "acts": []}
        res = app["run_head"](model, meta, np.array([0.0, 0.0, 5.0]))
        assert res["h"] is res["z"] and res["out"]["sentiment"]["p"].argmax() == 2

    def test_st_m_07_two_readings_share_the_trunk(self, app):
        model, meta = tiny_network()
        model["outs"]["emotion"] = (np.array([[1.0, 1.0]]), np.array([0.0]))
        meta["readings"].append({"name": "emotion", "classes": ["only"], "temperature": 1.0})
        res = app["run_head"](model, meta, np.array([3.0, -1.5]))
        assert res["out"]["emotion"]["logits"][0] == pytest.approx(2.5) and res["out"]["emotion"]["p"][0] == 1.0


def word_model():
    vocab = ["loss", "profit", "shares"]
    return {"vocab": vocab, "index": {w: j for j, w in enumerate(vocab)}, "texts": np.array([10, 20, 30]),
            "coef": np.array([[2.0, -2.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 2.0, 0.0]]), "intercept": np.array([0.0, 0.5, 0.0])}


class TestRunWords:
    # decision table over the words of a text: none / all unknown / some unknown / all known

    def test_st_m_08_no_words_at_all(self, app):
        res = app["run_words"](word_model(), {"readings": [{"classes": ["n", "u", "p"]}]}, "123 !!!")
        assert res["words"] == [] and res["known"] == 0 and np.allclose(res["p"], app["softmax"](np.array([0.0, 0.5, 0.0])))

    def test_st_m_09_every_word_unknown_gives_the_intercept_prior_and_marks_each_na(self, app):
        res = app["run_words"](word_model(), {"readings": [{"classes": ["n", "u", "p"]}]}, "qwerty zxcv")
        assert res["known"] == 0 and all(r["j"] is None for r in res["words"])
        assert np.allclose(res["p"], app["softmax"](np.array([0.0, 0.5, 0.0])))

    def test_st_m_10_known_words_add_their_coefficients_once_each(self, app):
        res = app["run_words"](word_model(), {"readings": [{"classes": ["n", "u", "p"]}]}, "profit profit shares zzz")
        assert [r["word"] for r in res["words"]] == ["profit", "shares", "zzz"] and res["known"] == 2
        assert np.allclose(res["logits"], [0.0 - 2.0, 0.5, 0.0 + 2.0])
        assert res["words"][0]["texts"] == 20 and np.allclose(res["words"][0]["p"], app["softmax"](np.array([-2.0, 0.5, 2.0])))

    def test_st_m_11_a_word_on_its_own_is_the_one_word_text(self, app):
        res = app["run_words"](word_model(), {"readings": [{"classes": ["n", "u", "p"]}]}, "loss")
        assert np.allclose(res["words"][0]["p"], res["p"])


class TestPercent:
    # boundary value analysis around the rounding of 0% and 100%

    @pytest.mark.parametrize("p, d, want", [
        (0.0, 1, "0.0%"), (0.00049, 1, "<0.1%"), (0.0005, 1, "0.1%"), (0.5, 1, "50.0%"),
        (0.9995, 1, ">99.9%"), (0.99949, 1, "99.9%"), (1.0, 1, "100.0%"),
        (0.0049, 0, "<1%"), (0.0051, 0, "1%"), (0.9951, 0, ">99%"), (0.9949, 0, "99%"), (1.0, 0, "100%"),
        (0.000049, 2, "<0.01%"), (0.99996, 2, ">99.99%"),
    ])
    def test_st_m_12_pct(self, app, p, d, want):
        assert app["pct"](p, d) == want

    def test_st_m_12b_exactly_half_a_unit_rounds_to_even_and_shows_as_below(self, app):
        # found by the boundary values: 0.5 percent formats as 0% in Python (round half to even), so it shows as <1%;
        # 0.05 percent is stored slightly above its decimal value and rounds up to 0.1%
        assert app["pct"](0.005, 0) == "<1%" and app["pct"](0.0005, 1) == "0.1%"


class TestDrawingHelpers:
    def test_st_m_13_spread_is_evenly_spaced_top_down_and_centred(self, app):
        y = app["spread"](5)
        assert np.allclose(np.diff(y), -0.2) and y.max() == pytest.approx(1.0) and y.min() == pytest.approx(0.2)
        assert app["spread"](1).tolist() == [0.6]
        assert np.allclose(app["spread"](3, step=0.1), [0.7, 0.6, 0.5])

    def test_st_m_14_strongest_keeps_the_largest_contributions_by_size(self, app):
        W = np.array([[1.0, -5.0, 0.1], [0.0, 0.0, 2.0]])
        values = np.array([1.0, 1.0, 1.0])
        got = app["strongest"](W, values, np.array([0, 1, 2]), [0, 1], 2)
        assert (1, 0, -5.0) in got and (0, 0, 1.0) in got and (2, 1, 2.0) in got and len(got) == 4

    def test_st_m_15_pathway_sizes_and_membership(self, app):
        model, meta = tiny_network()
        model["layers"] = [(np.random.RandomState(0).randn(6, 20), np.zeros(6), "relu")]
        model["mean"], model["sd"] = np.zeros(20), np.ones(20)
        model["outs"] = {"sentiment": (np.random.RandomState(1).randn(3, 6), np.zeros(3))}
        meta = {"readings": [{"name": "sentiment", "classes": ["a", "b", "c"], "temperature": 1.0}], "acts": ["relu"]}
        res = app["run_head"](model, meta, np.random.RandomState(2).randn(20))
        keep_in, keep = app["pathway"](model, res, [("sentiment", 0), ("sentiment", 1), ("sentiment", 2)], n_in=5, n_unit=4)
        assert len(keep_in) == 5 and len(set(keep_in)) == 5 and all(0 <= j < 20 for j in keep_in)
        assert len(keep) == 1 and len(keep[0]) == 4 and all(0 <= u < 6 for u in keep[0])

    @pytest.mark.parametrize("j, want", [(0, "[CLS] vector"), (1023, "[CLS] vector"), (1024, "token mean"), (2047, "token mean")])
    def test_st_m_16_feature_part_boundaries(self, app, j, want):
        assert app["feature_part"](j, 1024) == want

    def test_st_m_17_top_label(self, app):
        assert app["top_label"](["n", "u", "p"], np.array([0.2, 0.5, 0.3])) == ("u", 0.5)
