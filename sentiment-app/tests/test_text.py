"""The text path: what a typed sentence goes through before any model sees it, checked against the
training pipeline's cleaning (golden/cleaning.json) and on its own branches (ST-T-xx)."""
import pytest

CURLY = chr(0x2019)


class TestCleanMatchesTraining:
    def test_st_t_01_every_golden_case_cleans_as_the_pipeline_did(self, app, golden_cleaning):
        for case in golden_cleaning:
            assert app["clean"](case["raw"]) == case["clean"], repr(case["raw"])

    def test_st_t_02_every_golden_case_tokenises_as_the_pipeline_did(self, app, golden_cleaning):
        for case in golden_cleaning:
            got = app["words"](app["clean"](case["raw"]))
            want = list(dict.fromkeys(case["tokens"]))  # the app keeps each distinct word once
            assert got == want, repr(case["raw"])


class TestClean:
    # equivalence partitions of the raw text: plain, whitespace runs, invisible characters, placeholders, mojibake

    def test_st_t_03_plain_text_is_unchanged(self, app):
        assert app["clean"]("Shares jumped 5% after the beat.") == "Shares jumped 5% after the beat."

    def test_st_t_04_whitespace_runs_collapse_and_ends_are_trimmed(self, app):
        assert app["clean"]("  Shares\t\tjumped \n after  ") == "Shares jumped after"

    def test_st_t_05_invisible_characters_are_removed(self, app):
        assert app["clean"](chr(0xFEFF) + "a" + chr(0x200B) + "b" + chr(0xA0) + "c") == "ab c"

    def test_st_t_06_dialogue_comma_placeholder(self, app):
        assert app["clean"]("yes_comma_ it was") == "yes, it was"

    def test_st_t_07_euro_mojibake_is_repaired(self, app):
        assert app["clean"]("profit of " + chr(0xE2) + chr(0x201A) + chr(0xAC) + "2.1bn") == "profit of " + chr(0x20AC) + "2.1bn"

    def test_st_t_08_stray_a_hat_before_a_space_is_dropped(self, app):
        assert app["clean"]("Monday" + chr(0xC2) + " ahead") == "Monday ahead"

    def test_st_t_09_repair_that_would_lose_most_of_the_text_is_refused(self, app):
        s = (chr(0xE2) + chr(0x2026)) * 6
        assert app["fix_mojibake"](s) == s

    def test_st_t_10_text_without_mojibake_pairs_is_returned_as_the_same_object(self, app):
        s = "nothing to repair here"
        assert app["fix_mojibake"](s) is s

    @pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
    def test_st_t_11_empty_or_blank_input_cleans_to_the_empty_string(self, app, raw):
        assert app["clean"](raw) == ""


class TestWords:
    def test_st_t_12_lower_case_letters_only_in_order_of_first_appearance(self, app):
        assert app["words"]("Up, UP and Away 2024!") == ["up", "and", "away"]

    def test_st_t_13_apostrophes_inside_a_word_stay_and_curly_ones_are_straightened(self, app):
        assert app["words"]("the company" + CURLY + "s don't") == ["the", "company's", "don't"]

    def test_st_t_14_digits_and_symbols_are_not_words(self, app):
        assert app["words"]("12345 $AAPL +5% #win") == ["aapl", "win"]

    def test_st_t_15_no_letters_means_no_words(self, app):
        assert app["words"]("") == [] and app["words"]("123 !!!") == []
