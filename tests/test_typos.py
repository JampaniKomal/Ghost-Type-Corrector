"""Tests for the synthetic typo generator and the corpus word extraction.
Both are dependency-free (no TensorFlow), so these run fast in CI."""
import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "ai_model" / "src"
sys.path.insert(0, str(SRC))

from typos import ALPHABET, add_typo  # noqa: E402


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_dataset", "01_build_dataset.py")


def _osa_distance(a, b):
    """Optimal string alignment distance: like edit distance but an adjacent
    transposition (swap) counts as a single operation, which is what the 'swap'
    typo is."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def test_a_typo_is_at_most_one_edit_away():
    rng = random.Random(1)
    for _ in range(2000):
        word = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 10)))
        assert _osa_distance(word, add_typo(word, rng)) <= 1


def test_every_typo_kind_can_change_the_word():
    rng = random.Random(0)
    changed = {add_typo("keyboard", rng) for _ in range(50)}
    assert any(c != "keyboard" for c in changed)
    assert all(set(c) <= set(ALPHABET) for c in changed)


def test_iter_words_lowercases_and_strips_non_letters(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("The QUICK, brown fox! a to\n123 e-mail\n", encoding="utf-8")
    words = list(build.iter_words(corpus))
    assert "the" in words and "quick" in words and "brown" in words
    assert "email" in words          # 'e-mail' -> letters only
    assert "a" not in words          # too short (< 2)
    assert all(w.islower() and w.isalpha() for w in words)
