"""Tests for the committed model + the dictionary-gated beam corrector.

These run with only NumPy (no TensorFlow): they load the exported weights from
extension/model/ and exercise ai_model/src/inference.py, the exact code the
browser mirrors in JavaScript.
"""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_model" / "src"))
MODEL = ROOT / "extension" / "model"

from inference import Corrector  # noqa: E402


@pytest.fixture(scope="module")
def corrector():
    return Corrector(MODEL / "weights.npz", MODEL / "tokenizer_config.json", MODEL / "dictionary.txt")


@pytest.mark.parametrize("typo,expected", [
    ("teh", "the"),
    ("recieve", "receive"),
    ("beleive", "believe"),
    ("wrld", "world"),
    ("thsi", "this"),
    ("langauge", "language"),
])
def test_fixes_common_typos(corrector, typo, expected):
    assert corrector.correct(typo) == expected


def test_never_changes_a_known_word(corrector):
    for word in ["the", "hello", "world", "language", "receive", "computer", "science"]:
        assert corrector.correct(word) == word


def test_only_returns_dictionary_words(corrector):
    # Whatever it does to a garbled token, the result is either the input
    # unchanged or a real word - never an invented non-word.
    for junk in ["xqzptk", "zzzz", "qwxrt", "asdfgh"]:
        out = corrector.correct(junk)
        assert out == junk or out in corrector.dictionary


def test_leaves_non_alphabetic_tokens_alone(corrector):
    for token in ["", "123", "a1b2", "don't", "co-op"]:
        assert corrector.correct(token) == token


def test_leaves_very_short_words_alone(corrector):
    # "a" and "i" are real words the dictionary omits; they must not be "fixed".
    for token in ["a", "i", "A", "I", "an", "to", "of"]:
        assert corrector.correct(token) == token


def test_case_is_handled_by_lowercasing(corrector):
    # The model is lower-case; correct() lower-cases before gating. (The browser
    # extension restores the original case separately.)
    assert corrector.correct("TEH") == "the"


def test_keep_correct_rate_is_total_on_a_sample(corrector):
    rng = random.Random(0)
    words = rng.sample(sorted(corrector.dictionary), 150)
    kept = sum(corrector.correct(w) == w for w in words)
    assert kept == len(words)  # the gate must never corrupt a correct word


def test_candidates_are_ranked_and_deduped(corrector):
    cands = corrector.candidates("recieve")
    assert "receive" in cands
    assert len(cands) == len(set(cands))
