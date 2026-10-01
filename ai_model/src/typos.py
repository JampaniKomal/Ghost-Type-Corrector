"""Synthetic typo generation - the four classic single-character slips used to
build training pairs. Kept dependency-free so it can be imported without
TensorFlow (training imports it; so do the tests)."""
from __future__ import annotations

import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def add_typo(word: str, rng: random.Random | None = None) -> str:
    """Return ``word`` with one random typo (delete, insert, substitute or
    swap). May return the word unchanged when the chosen edit does not apply
    (e.g. a delete on a very short word)."""
    r = rng or random
    kind = r.choice(["delete", "insert", "substitute", "swap"])
    i = r.randrange(len(word))
    if kind == "delete" and len(word) > 3:
        return word[:i] + word[i + 1:]
    if kind == "insert":
        return word[:i] + r.choice(ALPHABET) + word[i:]
    if kind == "substitute":
        return word[:i] + r.choice(ALPHABET) + word[i + 1:]
    if kind == "swap" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word
