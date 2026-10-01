#!/usr/bin/env python3
"""Step 2 - train the character-level seq2seq corrector.

For every dictionary word we make training pairs: the word paired with itself
(so the model learns to leave correct words alone) and several noisy variants
produced by the four classic single-character typos (delete, insert, substitute,
swap). The model is a small encoder-decoder LSTM over characters; it reads a
(possibly misspelled) word and emits the corrected characters.

Usage:
    python 02_train_model.py [--epochs 18]

Reads ``ai_model/data/dictionary.txt`` (from step 1) and writes
``ai_model/autocorrect_model.h5`` and ``ai_model/data/tokenizer_config.json``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typos import add_typo

DATA = Path(__file__).resolve().parent.parent / "data"
MODEL = Path(__file__).resolve().parent.parent / "autocorrect_model.h5"

# Architecture / training (kept small enough to run client-side in the browser).
EMBEDDING_DIM = 96
LATENT_DIM = 160
MAX_LEN = 20
IDENTITY_COPIES = 2   # pairs that map a word to itself (keep correct words)
NOISY_COPIES = 4      # misspelled variants per word
SEED = 0


def build_pairs(words):
    pairs = []
    for w in words:
        pairs.extend([(w, w)] * IDENTITY_COPIES)
        seen = {w}
        for _ in range(NOISY_COPIES):
            n = add_typo(w)
            if n not in seen:
                pairs.append((n, w))
                seen.add(n)
    random.shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=384)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    words = [w for w in (DATA / "dictionary.txt").read_text(encoding="utf-8").splitlines() if w]
    print(f"dictionary: {len(words):,} words")
    pairs = build_pairs(words)
    print(f"training pairs: {len(pairs):,}")

    chars = sorted(set("".join(w for p in pairs for w in p)))
    vocab = ["", "\t", "\n"] + chars
    c2i = {c: i for i, c in enumerate(vocab)}
    i2c = {i: c for i, c in enumerate(vocab)}
    V, START, END, PAD = len(vocab), 1, 2, 0

    def vec(ws):
        out = np.zeros((len(ws), MAX_LEN), dtype="int32")
        for r, w in enumerate(ws):
            idx = [START] + [c2i.get(c, PAD) for c in w] + [END]
            for j, v in enumerate(idx[:MAX_LEN]):
                out[r, j] = v
        return out

    enc = vec([a for a, _ in pairs])
    dec = vec([b for _, b in pairs])
    tgt = np.concatenate([dec[:, 1:], np.zeros((len(dec), 1), "int32")], axis=1)

    ei = keras.Input((MAX_LEN,))
    ee = keras.layers.Embedding(V, EMBEDDING_DIM, mask_zero=True, name="encoder_embedding")(ei)
    _, h, c = keras.layers.LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")(ee)
    di = keras.Input((MAX_LEN,))
    de = keras.layers.Embedding(V, EMBEDDING_DIM, mask_zero=True, name="decoder_embedding")(di)
    do, _, _ = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True,
                                 name="decoder_lstm")(de, initial_state=[h, c])
    out = keras.layers.Dense(V, activation="softmax", name="output_dense")(do)
    model = keras.Model([ei, di], out)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"parameters: {model.count_params():,}")

    model.fit([enc, dec], tgt, batch_size=args.batch_size, epochs=args.epochs,
              validation_split=0.05, verbose=2)

    model.save(MODEL)
    tok = {"char_to_index": c2i, "index_to_char": {str(k): v for k, v in i2c.items()},
           "vocab_size": V, "max_seq_length": MAX_LEN, "start_token_index": START,
           "end_token_index": END, "pad_token_index": PAD,
           "embedding_dim": EMBEDDING_DIM, "latent_dim": LATENT_DIM}
    (DATA / "tokenizer_config.json").write_text(json.dumps(tok, indent=2), encoding="utf-8")
    print(f"saved {MODEL} and tokenizer_config.json")
    print("next: python 03_export_for_browser.py")


if __name__ == "__main__":
    main()
