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
