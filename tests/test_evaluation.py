"""
Tests for src/evaluation.py

Covers answer extraction, correctness checking, and evaluate_response.
No GPU / model loading tests are included.
"""

import pytest

from src.evaluation import (
    check_correctness,
    evaluate_response,
    extract_letter_answer,
    extract_number_answer,
)


# ---------------------------------------------------------------------------
# extract_number_answer
# ---------------------------------------------------------------------------


class TestExtractNumberAnswer:
    # ---- #### pattern ----

    def test_hash_pattern_basic(self):
        assert extract_number_answer("Working...\n#### 42") == "42"

    def test_hash_pattern_strips_commas(self):
        assert extract_number_answer("#### 1,234") == "1234"

    def test_hash_pattern_large_number(self):
        assert extract_number_answer("#### 1,000,000") == "1000000"

    def test_hash_pattern_decimal(self):
        assert extract_number_answer("#### 3.14") == "3.14"

    def test_hash_pattern_whitespace(self):
        assert extract_number_answer("####   99  ") == "99"

    # ---- fallback to last number ----

    def test_fallback_last_number(self):
        assert extract_number_answer("The answer is 7.") == "7"

    def test_fallback_multiple_numbers_returns_last(self):
        # "I have 3 apples and 5 oranges." → last number is 5
        assert extract_number_answer("I have 3 apples and 5 oranges.") == "5"

    def test_fallback_strips_commas(self):
        assert extract_number_answer("Total cost: 2,500 dollars.") == "2500"

    def test_fallback_decimal(self):
        assert extract_number_answer("Result: 0.75") == "0.75"

    # ---- no number ----

    def test_no_number_returns_none(self):
        assert extract_number_answer("No numbers here!") is None

    def test_empty_string_returns_none(self):
        assert extract_number_answer("") is None

    def test_none_input_returns_none(self):
        assert extract_number_answer(None) is None

    # ---- hash pattern takes priority over fallback ----

    def test_hash_pattern_preferred_over_last_number(self):
        # The text has numbers before #### and the answer after ####
        result = extract_number_answer("I paid 100 dollars.\n#### 42")
        assert result == "42"


# ---------------------------------------------------------------------------
# extract_letter_answer
# ---------------------------------------------------------------------------


class TestExtractLetterAnswer:
    # ---- word-boundary matches ----

    def test_standalone_A(self):
        assert extract_letter_answer("The answer is A.") == "A"

    def test_standalone_B(self):
        assert extract_letter_answer("Answer: B") == "B"

    def test_standalone_C(self):
        assert extract_letter_answer("I think C is correct.") == "C"

    def test_standalone_D(self):
        assert extract_letter_answer("D seems most plausible.") == "D"

    def test_first_letter_returned(self):
        # Should return first letter encountered (A before C)
        assert extract_letter_answer("A or C?") == "A"

    def test_case_insensitive_input_uppercased(self):
        # extract_letter_answer only looks for uppercase A-D per spec,
        # but result should be uppercase
        result = extract_letter_answer("The answer is A")
        assert result == "A"

    # ---- punctuation fallback ----

    def test_letter_followed_by_period(self):
        assert extract_letter_answer("A.") == "A"

    def test_letter_followed_by_paren(self):
        assert extract_letter_answer("B) Mercury") == "B"

    def test_letter_followed_by_comma(self):
        assert extract_letter_answer("C, which is correct") == "C"

    def test_letter_followed_by_colon(self):
        assert extract_letter_answer("D: the final answer") == "D"

    # ---- no letter ----

    def test_no_letter_returns_none(self):
        assert extract_letter_answer("No valid answer here.") is None

    def test_empty_string_returns_none(self):
        assert extract_letter_answer("") is None

    def test_none_input_returns_none(self):
        assert extract_letter_answer(None) is None

    def test_letter_outside_AD_returns_none(self):
        # E is not in A-D
        assert extract_letter_answer("The answer is E.") is None

    # ---- letter embedded in word should not match as word-boundary ----

    def test_letter_in_word_no_false_positive(self):
        # "Above" contains 'A' but not as a standalone word
        # The regex \b[A-D]\b should NOT match the 'A' in "Above"
        # (it would only match if 'A' is surrounded by non-word characters)
        result = extract_letter_answer("Above all, consider the context.")
        # 'A' in 'Above' is at start, preceded by nothing (word boundary exists)
        # so \bA\b won't match because 'A' is followed by 'b' (word char)
        # We expect None here since there's no standalone A-D
        assert result is None


# ---------------------------------------------------------------------------
# check_correctness
# ---------------------------------------------------------------------------


class TestCheckCorrectness:
    # ---- math ----

    def test_math_exact_match(self):
        assert check_correctness("42", "42", "math") is True

    def test_math_float_tolerance(self):
        assert check_correctness("3.14159", "3.14159", "math") is True

    def test_math_within_tolerance(self):
        # Difference < 1e-3
        assert check_correctness("1.0001", "1.0", "math") is True

    def test_math_outside_tolerance(self):
        assert check_correctness("1.01", "1.0", "math") is False

    def test_math_wrong_number(self):
        assert check_correctness("5", "42", "math") is False

    def test_math_none_extracted(self):
        assert check_correctness(None, "42", "math") is False

    def test_math_non_numeric_extracted(self):
        assert check_correctness("abc", "42", "math") is False

    # ---- multiple-choice (science, truthfulness, commonsense) ----

    def test_mc_correct_A(self):
        assert check_correctness("A", "A", "science") is True

    def test_mc_correct_B(self):
        assert check_correctness("B", "B", "commonsense") is True

    def test_mc_case_insensitive(self):
        assert check_correctness("a", "A", "truthfulness") is True

    def test_mc_wrong_letter(self):
        assert check_correctness("B", "A", "science") is False

    def test_mc_none_extracted(self):
        assert check_correctness(None, "A", "science") is False

    def test_mc_with_whitespace(self):
        assert check_correctness(" A ", "A", "science") is True


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    def test_math_correct(self):
        result = evaluate_response("The answer is #### 42", "42", "math")
        assert result["correct"] is True
        assert result["extracted_answer"] == "42"
        assert result["raw_response"] == "The answer is #### 42"

    def test_math_incorrect(self):
        result = evaluate_response("I think the answer is 10", "42", "math")
        assert result["correct"] is False
        assert result["extracted_answer"] == "10"

    def test_math_no_number(self):
        result = evaluate_response("I have no idea.", "42", "math")
        assert result["correct"] is False
        assert result["extracted_answer"] is None

    def test_mc_correct(self):
        result = evaluate_response("The answer is A.", "A", "science")
        assert result["correct"] is True
        assert result["extracted_answer"] == "A"

    def test_mc_incorrect(self):
        result = evaluate_response("I choose B.", "A", "science")
        assert result["correct"] is False
        assert result["extracted_answer"] == "B"

    def test_mc_no_letter(self):
        result = evaluate_response("I don't know.", "A", "commonsense")
        assert result["correct"] is False
        assert result["extracted_answer"] is None

    def test_raw_response_preserved(self):
        raw = "Some lengthy model output with B) as the answer."
        result = evaluate_response(raw, "B", "truthfulness")
        assert result["raw_response"] == raw

    def test_returns_required_keys(self):
        result = evaluate_response("#### 7", "7", "math")
        assert set(result.keys()) == {"extracted_answer", "correct", "raw_response"}

    def test_commonsense_correct(self):
        result = evaluate_response("Option C is the best continuation.", "C", "commonsense")
        assert result["correct"] is True

    def test_truthfulness_case_insensitive(self):
        result = evaluate_response("b)", "B", "truthfulness")
        assert result["correct"] is True

    def test_math_with_commas_in_answer(self):
        result = evaluate_response("The total is #### 1,000", "1000", "math")
        assert result["correct"] is True
        assert result["extracted_answer"] == "1000"


# ---------------------------------------------------------------------------
# Letter extraction: article / contraction fix
# ---------------------------------------------------------------------------


class TestExtractLetterAnswerArticleFix:
    """The standalone match must not fire on the article 'a' or contractions."""

    def test_article_a_not_matched(self):
        # Previously returned "A" (the article), mislabeling the answer.
        assert extract_letter_answer("a bit tricky, so D.") == "D"

    def test_contraction_d_not_matched(self):
        # The 'd' in "I'd" sits at a word boundary; it must not match.
        assert extract_letter_answer("I'd say B") == "B"

    def test_article_a_only_returns_none(self):
        assert extract_letter_answer("a tricky question indeed") is None

    def test_answer_phrase_lowercase_letter(self):
        # Lowercase letters are still accepted inside an explicit phrase.
        assert extract_letter_answer("the answer is c") == "C"

    def test_answer_phrase_takes_precedence(self):
        # "B" appears first in the text, but the explicit phrase names C.
        assert extract_letter_answer("B is tempting but the answer is C") == "C"

    def test_lowercase_with_punctuation_still_works(self):
        # 'b)' is an answer format, not an article; stays accepted.
        assert extract_letter_answer("b) Mercury") == "B"

    def test_trailing_d_of_word_not_matched(self):
        # 'd' at the end of "word." is inside a word: no boundary match.
        assert extract_letter_answer("That is my final word.") is None


# ---------------------------------------------------------------------------
# Number extraction: explicit answer phrase
# ---------------------------------------------------------------------------


class TestExtractNumberAnswerPhrase:

    def test_phrase_beats_last_number(self):
        # The last number (3) is not the answer; the phrase names 18.
        text = "The answer is 18. Double-check: 3 apples each."
        assert extract_number_answer(text) == "18"

    def test_hash_pattern_still_wins(self):
        text = "The answer is 5.\n#### 7"
        assert extract_number_answer(text) == "7"

    def test_fallback_unchanged_without_phrase(self):
        assert extract_number_answer("Step 7*6=42") == "42"


# ---------------------------------------------------------------------------
# Correctness matrix: problem_id join
# ---------------------------------------------------------------------------


class TestComputeCorrectnessMatrixJoin:

    @staticmethod
    def _results(model, ids_correct):
        return [
            {"problem_id": pid, "model": model, "correct": c}
            for pid, c in ids_correct
        ]

    def test_reordered_results_align_correctly(self):
        from src.evaluation import compute_correctness_matrix

        m1 = self._results("m1", [("p0", True), ("p1", False), ("p2", True)])
        # Same problems, different order: must land in m1's row order.
        m2 = self._results("m2", [("p2", False), ("p0", True), ("p1", True)])
        matrix, ids, names = compute_correctness_matrix([m1, m2])
        assert ids == ["p0", "p1", "p2"]
        assert names == ["m1", "m2"]
        assert matrix[:, 0].tolist() == [True, False, True]
        assert matrix[:, 1].tolist() == [True, True, False]

    def test_missing_problem_raises(self):
        import pytest as _pytest

        from src.evaluation import compute_correctness_matrix

        m1 = self._results("m1", [("p0", True), ("p1", False)])
        m2 = self._results("m2", [("p0", True)])
        with _pytest.raises(ValueError, match="missing"):
            compute_correctness_matrix([m1, m2])


# ---------------------------------------------------------------------------
# Re-scoring saved generations (src/rescore.py)
# ---------------------------------------------------------------------------


class TestRescoreResults:

    @staticmethod
    def _problem(pid, task_type, gold):
        return {"problem_id": pid, "task_type": task_type, "gold_answer": gold}

    def test_rescore_flips_with_corrected_gold(self):
        from src.rescore import rescore_results

        # Old pipeline scored this wrong: numeric ARC key "3" never matched
        # the extracted letter. With the corrected gold "C" it is right.
        results = [{
            "problem_id": "arc_0001",
            "raw_response": "The answer is C.",
            "extracted_answer": "C",
            "gold_answer": "3",
            "correct": False,
        }]
        problems = {"arc_0001": self._problem("arc_0001", "science", "C")}
        summary = rescore_results(results, problems)
        assert results[0]["correct"] is True
        assert summary["n_flipped"] == 1
        assert summary["accuracy_after"] == 1.0

    def test_truthfulqa_left_untouched(self):
        from src.rescore import rescore_results

        results = [{
            "problem_id": "tq_0001",
            "raw_response": "A",
            "extracted_answer": "A",
            "gold_answer": "A",
            "correct": True,
        }]
        problems = {"tq_0001": self._problem("tq_0001", "truthfulness", "C")}
        summary = rescore_results(results, problems)
        # Prompt changed for TruthfulQA; old generations cannot be re-scored.
        assert results[0]["correct"] is True
        assert results[0]["gold_answer"] == "A"
        assert summary["n_skipped"] == 1
        assert summary["n_rescored"] == 0

    def test_unknown_problem_id_skipped(self):
        from src.rescore import rescore_results

        results = [{
            "problem_id": "ghost_0001",
            "raw_response": "#### 42",
            "gold_answer": "42",
            "correct": True,
        }]
        summary = rescore_results(results, {})
        assert summary["n_skipped"] == 1
        assert results[0]["correct"] is True


# ---------------------------------------------------------------------------
# Letter extraction: valid_letters range (TruthfulQA has up to 13 choices)
# ---------------------------------------------------------------------------


class TestExtractLetterAnswerRange:

    def test_gold_beyond_d_extractable(self):
        # With a fixed A-D range, a 13-choice problem whose gold is "G" was
        # unanswerable for every model.
        letters = "ABCDEFGHIJKLM"
        assert extract_letter_answer("The answer is G.", letters) == "G"
        assert extract_letter_answer("g) the correct one", letters) == "G"

    def test_letters_outside_range_ignored(self):
        # Default range stays A-D: "E" is not a valid answer there.
        assert extract_letter_answer("The answer is E.") is None

    def test_bare_capital_i_is_pronoun(self):
        letters = "ABCDEFGHIJKLM"
        assert extract_letter_answer("I would say B", letters) == "B"

    def test_i_with_punctuation_is_answer(self):
        letters = "ABCDEFGHIJKLM"
        assert extract_letter_answer("I) the ninth option", letters) == "I"

    def test_i_in_answer_phrase_is_answer(self):
        letters = "ABCDEFGHIJKLM"
        assert extract_letter_answer("the answer is I", letters) == "I"

    def test_five_choice_arc(self):
        assert extract_letter_answer("E is correct", "ABCDE") == "E"


class TestArticleAfterAnswerCue:
    """Regression: the article directly after the cue word must not parse as 'A'."""

    def test_article_after_answer_is(self):
        assert extract_letter_answer("The answer is a bit tricky, so D.") == "D"

    def test_article_after_answer_is_with_late_letter(self):
        assert extract_letter_answer("The answer is a hard one, I would pick B") == "B"

    def test_uppercase_a_after_cue_is_an_answer(self):
        assert extract_letter_answer("the answer is A") == "A"

    def test_lowercase_c_after_cue_still_accepted(self):
        assert extract_letter_answer("the answer is c") == "C"

    def test_cue_words_case_insensitive(self):
        assert extract_letter_answer("ANSWER: B") == "B"
        assert extract_letter_answer("The Answer is C") == "C"
