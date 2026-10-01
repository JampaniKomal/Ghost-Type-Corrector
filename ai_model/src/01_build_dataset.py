#!/usr/bin/env python3
"""Step 1 - build the word dictionary from a text corpus.

The corrector works a word at a time (that is how the extension uses it), so the
unit of everything is the word, not the sentence. This reads a corpus, keeps the
most frequent alphabetic words, and writes them to ``dictionary.txt``. That file
is used twice: the training pairs are generated from it, and it is the gate at
inference time (a known word is never "corrected", and a proposal is only used
if it too is a real word).

Usage:
    python 01_build_dataset.py [corpus.txt] [--top 15000]

Defaults to the bundled ``ai_model/data/sample_corpus.txt`` so the pipeline runs
out of the box. The committed model was trained on the full Leipzig Corpora
Collection (British English); see the README.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

ALPHABET = set("abcdefghijklmnopqrstuvwxyz")
DATA = Path(__file__).resolve().parent.parent / "data"


def iter_words(corpus: Path):
    with open(corpus, encoding="utf-8", errors="ignore") as f:
        for line in f:
            for token in line.lower().split():
                word = "".join(ch for ch in token if ch in ALPHABET)
                if 2 <= len(word) <= 18:
                    yield word


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", nargs="?", default=str(DATA / "sample_corpus.txt"))
    ap.add_argument("--top", type=int, default=15000, help="dictionary size")
    args = ap.parse_args()

    freq = collections.Counter(iter_words(Path(args.corpus)))
    words = [w for w, _ in freq.most_common(args.top)]

    out = DATA / "dictionary.txt"
    out.write_text("\n".join(words), encoding="utf-8")
    print(f"corpus: {args.corpus}")
    print(f"unique words seen: {len(freq):,}")
    print(f"dictionary written: {out} ({len(words):,} words)")
    print(f"most common: {', '.join(words[:12])}")


if __name__ == "__main__":
    main()
