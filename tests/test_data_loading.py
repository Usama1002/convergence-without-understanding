"""
Tests for src/data_loading.py
"""

import pytest

from src.data_loading import (
    _extract_gsm8k_answer,
    _format_choices,
    format_prompt,
    load_all_problems,
    load_dataset_for_eval,
)

# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestExtractGsm8kAnswer:
    def test_basic(self):
        assert _extract_gsm8k_answer("He had 3 apples.\n#### 42") == "42"

    def test_strips_commas(self):
        assert _extract_gsm8k_answer("Some working.\n#### 1,234") == "1234"

    def test_large_number(self):
        assert _extract_gsm8k_answer("#### 1,000,000") == "1000000"

    def test_whitespace_around_number(self):
        assert _extract_gsm8k_answer("#### 7 ") == "7"

    def test_no_hash_fallback(self):
        # When there is no #### marker the whole string is returned stripped
        result = _extract_gsm8k_answer("  42  ")
        assert result == "42"


class TestFormatChoices:
    def test_two_choices(self):
        result = _format_choices(["Yes", "No"])
        assert "A) Yes" in result
        assert "B) No" in result

    def test_four_choices(self):
        choices = ["alpha", "beta", "gamma", "delta"]
        result = _format_choices(choices)
        for letter, text in zip("ABCD", choices):
            assert f"{letter}) {text}" in result

    def test_empty_list(self):
        assert _format_choices([]) == ""


# ---------------------------------------------------------------------------
# Integration tests – dataset loading
# ---------------------------------------------------------------------------

REQUIRED_BASE_FIELDS = {"problem_id", "domain", "task_type", "question", "gold_answer"}


def _check_base_fields(problem: dict, dataset_name: str) -> None:
    missing = REQUIRED_BASE_FIELDS - problem.keys()
    assert not missing, f"[{dataset_name}] Missing fields {missing} in {problem}"


class TestLoadGsm8k:
    @pytest.fixture(scope="class")
    def problems(self):
        return load_dataset_for_eval("gsm8k")

    def test_count(self, problems):
        assert len(problems) == 200

    def test_fields(self, problems):
        for p in problems:
            _check_base_fields(p, "gsm8k")

    def test_no_choices_field(self, problems):
        for p in problems:
            assert "choices" not in p

    def test_problem_id_format(self, problems):
        for p in problems:
            assert p["problem_id"].startswith("gsm8k_")

    def test_domain(self, problems):
        for p in problems:
            assert p["domain"] == "math"

    def test_gold_answer_no_commas(self, problems):
        for p in problems:
            assert "," not in p["gold_answer"]


class TestLoadArcChallenge:
    @pytest.fixture(scope="class")
    def problems(self):
        return load_dataset_for_eval("arc_challenge")

    def test_count(self, problems):
        assert len(problems) == 200

    def test_fields(self, problems):
        for p in problems:
            _check_base_fields(p, "arc_challenge")

    def test_has_choices(self, problems):
        for p in problems:
            assert "choices" in p
            assert isinstance(p["choices"], list)
            assert len(p["choices"]) > 0

    def test_has_choice_labels(self, problems):
        for p in problems:
            assert "choice_labels" in p
            assert len(p["choice_labels"]) == len(p["choices"])

    def test_problem_id_format(self, problems):
        for p in problems:
            assert p["problem_id"].startswith("arc_challenge_")

    def test_domain(self, problems):
        for p in problems:
            assert p["domain"] == "science"


class TestLoadTruthfulQA:
    @pytest.fixture(scope="class")
    def problems(self):
        return load_dataset_for_eval("truthfulqa")

    def test_count(self, problems):
        assert len(problems) == 200

    def test_fields(self, problems):
        for p in problems:
            _check_base_fields(p, "truthfulqa")

    def test_has_choices(self, problems):
        for p in problems:
            assert "choices" in p
            assert isinstance(p["choices"], list)
            assert len(p["choices"]) > 0

    def test_gold_answer_is_letter(self, problems):
        valid_letters = set("ABCDEFGHIJ")
        for p in problems:
            assert p["gold_answer"] in valid_letters, (
                f"gold_answer '{p['gold_answer']}' is not a letter"
            )

    def test_problem_id_format(self, problems):
        for p in problems:
            assert p["problem_id"].startswith("truthfulqa_")

    def test_domain(self, problems):
        for p in problems:
            assert p["domain"] == "truthfulness"


class TestLoadHellaSwag:
    @pytest.fixture(scope="class")
    def problems(self):
        return load_dataset_for_eval("hellaswag")

    def test_count(self, problems):
        assert len(problems) == 200

    def test_fields(self, problems):
        for p in problems:
            _check_base_fields(p, "hellaswag")

    def test_has_choices(self, problems):
        for p in problems:
            assert "choices" in p
            assert len(p["choices"]) == 4

    def test_gold_answer_is_letter(self, problems):
        valid_letters = set("ABCD")
        for p in problems:
            assert p["gold_answer"] in valid_letters

    def test_problem_id_format(self, problems):
        for p in problems:
            assert p["problem_id"].startswith("hellaswag_")

    def test_domain(self, problems):
        for p in problems:
            assert p["domain"] == "commonsense"


class TestLoadAllProblems:
    @pytest.fixture(scope="class")
    def all_problems(self):
        return load_all_problems()

    def test_total_count(self, all_problems):
        assert len(all_problems) == 800

    def test_200_per_domain(self, all_problems):
        from collections import Counter

        counts = Counter(p["domain"] for p in all_problems)
        assert counts["math"] == 200
        assert counts["science"] == 200
        assert counts["truthfulness"] == 200
        assert counts["commonsense"] == 200


# ---------------------------------------------------------------------------
# Tests for format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    # -- Math (no choices) --

    def test_math_variant0_returns_tuple(self):
        problem = {
            "problem_id": "gsm8k_0000",
            "domain": "math",
            "task_type": "math",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
        user_prompt, system_prompt = format_prompt(problem, prompt_variant=0)
        assert isinstance(user_prompt, str)
        assert isinstance(system_prompt, str)

    def test_math_variant0_system_prompt(self):
        from src.config import SYSTEM_PROMPTS

        problem = {
            "problem_id": "gsm8k_0000",
            "domain": "math",
            "task_type": "math",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
        _, system_prompt = format_prompt(problem, prompt_variant=0)
        assert system_prompt == SYSTEM_PROMPTS["math"]

    def test_math_question_in_user_prompt(self):
        problem = {
            "problem_id": "gsm8k_0000",
            "domain": "math",
            "task_type": "math",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
        user_prompt, _ = format_prompt(problem)
        assert "What is 2+2?" in user_prompt

    def test_math_no_choices_in_user_prompt(self):
        problem = {
            "problem_id": "gsm8k_0000",
            "domain": "math",
            "task_type": "math",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
        user_prompt, _ = format_prompt(problem)
        assert "Choices:" not in user_prompt

    # -- MC (with choices) --

    def test_mc_choices_in_user_prompt(self):
        problem = {
            "problem_id": "arc_challenge_0000",
            "domain": "science",
            "task_type": "science",
            "question": "Which planet is closest to the Sun?",
            "choices": ["Venus", "Mercury", "Earth", "Mars"],
            "choice_labels": ["A", "B", "C", "D"],
            "gold_answer": "B",
        }
        user_prompt, _ = format_prompt(problem)
        assert "Choices:" in user_prompt
        assert "Mercury" in user_prompt
        assert "A) Venus" in user_prompt

    def test_mc_variant0_system_prompt(self):
        from src.config import SYSTEM_PROMPTS

        problem = {
            "problem_id": "arc_challenge_0000",
            "domain": "science",
            "task_type": "science",
            "question": "Which planet is closest to the Sun?",
            "choices": ["Venus", "Mercury", "Earth", "Mars"],
            "choice_labels": ["A", "B", "C", "D"],
            "gold_answer": "B",
        }
        _, system_prompt = format_prompt(problem, prompt_variant=0)
        assert system_prompt == SYSTEM_PROMPTS["science"]

    # -- Alternative prompt variants --

    def test_variant1_uses_alternative(self):
        from src.config import ALTERNATIVE_PROMPTS

        problem = {
            "problem_id": "gsm8k_0000",
            "domain": "math",
            "task_type": "math",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
        _, system_prompt = format_prompt(problem, prompt_variant=1)
        assert system_prompt == ALTERNATIVE_PROMPTS["math"][0]

    def test_variant2_uses_alternative(self):
        from src.config import ALTERNATIVE_PROMPTS

        problem = {
            "problem_id": "gsm8k_0000",
            "domain": "math",
            "task_type": "math",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
        _, system_prompt = format_prompt(problem, prompt_variant=2)
        assert system_prompt == ALTERNATIVE_PROMPTS["math"][1]

    def test_variant3_uses_alternative(self):
        from src.config import ALTERNATIVE_PROMPTS

        problem = {
            "problem_id": "gsm8k_0000",
            "domain": "math",
            "task_type": "math",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
        _, system_prompt = format_prompt(problem, prompt_variant=3)
        assert system_prompt == ALTERNATIVE_PROMPTS["math"][2]

    def test_hellaswag_format(self):
        problem = {
            "problem_id": "hellaswag_0000",
            "domain": "commonsense",
            "task_type": "commonsense",
            "question": "She walked to the store.",
            "choices": ["She bought milk.", "She flew away.", "She slept.", "She swam."],
            "gold_answer": "A",
        }
        user_prompt, system_prompt = format_prompt(problem)
        assert "She walked to the store." in user_prompt
        assert "A) She bought milk." in user_prompt

    def test_truthfulqa_format(self):
        problem = {
            "problem_id": "truthfulqa_0000",
            "domain": "truthfulness",
            "task_type": "truthfulness",
            "question": "Is the earth flat?",
            "choices": ["Yes", "No"],
            "gold_answer": "B",
        }
        user_prompt, system_prompt = format_prompt(problem)
        assert "Is the earth flat?" in user_prompt
        assert "A) Yes" in user_prompt
        assert "B) No" in user_prompt


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestLoadDatasetErrors:
    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_for_eval("nonexistent_dataset")


# ---------------------------------------------------------------------------
# Gold-answer fixes (pure parser tests, no dataset download)
# ---------------------------------------------------------------------------


class TestArcAnswerKeyMapping:
    @staticmethod
    def _row(labels, key):
        return {
            "question": "Which one?",
            "choices": {
                "text": [f"choice {l}" for l in labels],
                "label": list(labels),
            },
            "answerKey": key,
        }

    def test_numeric_answer_key_maps_to_letter(self):
        # Some ARC items use numeric labels; the raw key "3" can never match
        # the letter the model answers, so it must map via position -> "C".
        from src.data_loading import _parse_arc_challenge

        problem = _parse_arc_challenge(self._row(["1", "2", "3", "4"], "3"), 0)
        assert problem["gold_answer"] == "C"

    def test_letter_answer_key_unchanged(self):
        from src.data_loading import _parse_arc_challenge

        problem = _parse_arc_challenge(self._row(["A", "B", "C", "D"], "B"), 0)
        assert problem["gold_answer"] == "B"


class TestTruthfulQAGoldNotAlwaysA:
    @staticmethod
    def _row(question):
        # HF truthful_qa mc1_targets always lists the correct answer first.
        return {
            "question": question,
            "mc1_targets": {
                "choices": ["correct one", "wrong 1", "wrong 2", "wrong 3"],
                "labels": [1, 0, 0, 0],
            },
        }

    def test_gold_letter_varies_across_problems(self):
        from src.data_loading import _parse_truthfulqa

        golds = {
            _parse_truthfulqa(self._row(f"q{i}"), i)["gold_answer"]
            for i in range(30)
        }
        # Without shuffling every gold is "A"; with the fix the letters vary.
        assert len(golds) > 1

    def test_gold_letter_points_at_correct_choice(self):
        from src.data_loading import _parse_truthfulqa

        for i in range(10):
            problem = _parse_truthfulqa(self._row(f"q{i}"), i)
            gold_idx = ord(problem["gold_answer"]) - ord("A")
            assert problem["choices"][gold_idx] == "correct one"

    def test_shuffle_is_deterministic(self):
        from src.data_loading import _parse_truthfulqa

        p1 = _parse_truthfulqa(self._row("same question"), 5)
        p2 = _parse_truthfulqa(self._row("same question"), 5)
        assert p1["choices"] == p2["choices"]
        assert p1["gold_answer"] == p2["gold_answer"]
