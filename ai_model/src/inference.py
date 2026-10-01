#!/usr/bin/env python3
"""Autocorrect inference: a character-level seq2seq greedy decoder plus a
dictionary gate.

The neural model proposes a correction; the dictionary decides whether to use
it. A word already in the dictionary is never touched, and a proposal is only
accepted if it is itself a real word. This is what keeps the corrector from
"fixing" words that were already right - the single most important property
for an autocorrect that types for you.

Inference needs only NumPy: it reads the exported weights (``weights.npz``),
the tokenizer and the dictionary. TensorFlow is only needed to *train* and
*export* the model, never to run it. The browser extension runs this exact
same math in plain JavaScript (``extension/js/inference.js``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Corrector:
    def __init__(self, weights_path, tokenizer_path, dictionary_path):
        w = np.load(weights_path)
        self.enc_emb, self.enc_k, self.enc_rk, self.enc_b = (
            w["enc_emb"], w["enc_k"], w["enc_rk"], w["enc_b"])
        self.dec_emb, self.dec_k, self.dec_rk, self.dec_b = (
            w["dec_emb"], w["dec_k"], w["dec_rk"], w["dec_b"])
        self.den_k, self.den_b = w["den_k"], w["den_b"]
        self.U = self.enc_rk.shape[0]

        cfg = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
        self.c2i = cfg["char_to_index"]
        self.i2c = {int(k): v for k, v in cfg["index_to_char"].items()}
        self.maxlen = cfg["max_seq_length"]
        self.start = cfg["start_token_index"]
        self.end = cfg["end_token_index"]
        self.pad = cfg["pad_token_index"]

        self.dictionary = {
            line.strip()
            for line in Path(dictionary_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _step(self, x, h, c, k, rk, b):
        """One LSTM cell step. Keras gate order is input, forget, cell, output."""
        u = self.U
        z = x @ k + h @ rk + b
        i = self._sigmoid(z[:u])
        f = self._sigmoid(z[u:2 * u])
        g = np.tanh(z[2 * u:3 * u])
        o = self._sigmoid(z[3 * u:])
        c2 = f * c + i * g
        return o * np.tanh(c2), c2

    def _encode(self, word):
        h = np.zeros(self.U, dtype=np.float32)
        c = np.zeros(self.U, dtype=np.float32)
        seq = [self.start] + [self.c2i.get(ch, self.pad) for ch in word] + [self.end]
        for t in seq:
            h, c = self._step(self.enc_emb[t], h, c, self.enc_k, self.enc_rk, self.enc_b)
        return h, c

    @staticmethod
    def _log_softmax(v):
        v = v - v.max()
        e = np.exp(v)
        return np.log(e / e.sum())

    def propose(self, word):
        """The greedy neural suggestion, no dictionary gate (may be a non-word)."""
        h, c = self._encode(word.lower())
        tok, out = self.start, ""
        for _ in range(self.maxlen):
            h, c = self._step(self.dec_emb[tok], h, c, self.dec_k, self.dec_rk, self.dec_b)
            tok = int(np.argmax(h @ self.den_k + self.den_b))
            if tok in (self.end, self.pad):
                break
            ch = self.i2c.get(tok, "")
            if ch not in ("\t", "\n", ""):
                out += ch
        return out

    def candidates(self, word, beam_width=5, max_candidates=8):
        """Beam search: the most likely corrections, best first. Giving the
        dictionary gate several candidates instead of one roughly doubles the
        real correction rate, because the top greedy guess is often a near-miss
        non-word while a slightly lower-scored beam is the real word."""
        h0, c0 = self._encode(word.lower())
        beams = [(0.0, [self.start], h0, c0)]
        finished = []
        for _ in range(self.maxlen):
            nxt = []
            for score, seq, h, c in beams:
                nh, nc = self._step(self.dec_emb[seq[-1]], h, c, self.dec_k, self.dec_rk, self.dec_b)
                logp = self._log_softmax(nh @ self.den_k + self.den_b)
                for t in np.argsort(logp)[-beam_width:]:
                    t = int(t)
                    if t in (self.end, self.pad):
                        finished.append((score + logp[t], seq))
                    else:
                        nxt.append((score + logp[t], seq + [t], nh, nc))
            if not nxt:
                break
            nxt.sort(key=lambda b: -b[0] / max(len(b[1]), 1))
            beams = nxt[:beam_width]
        pool = finished + [(s, sq) for s, sq, _, _ in beams]
        pool.sort(key=lambda b: -b[0] / max(len(b[1]), 1))
        words, seen = [], set()
        for _, seq in pool:
            txt = "".join(self.i2c.get(t, "") for t in seq if t not in (self.start, self.end, self.pad))
            txt = txt.replace("\t", "").replace("\n", "")
            if txt and txt not in seen:
                seen.add(txt)
                words.append(txt)
            if len(words) >= max_candidates:
                break
        return words

    def correct(self, word):
        """Dictionary-gated beam correction. A known word is never changed; an
        unknown word is replaced by the highest-ranked real-word candidate, or
        left as-is if none of the candidates are in the dictionary."""
        lower = word.lower()
        if not lower.isalpha() or lower in self.dictionary:
            return word
        for cand in self.candidates(lower):
            if cand != lower and cand in self.dictionary:
                return cand
        return word


def _default_paths():
    model = Path(__file__).resolve().parent.parent.parent / "extension" / "model"
    return model / "weights.npz", model / "tokenizer_config.json", model / "dictionary.txt"


def main():
    import sys

    corrector = Corrector(*_default_paths())
    words = sys.argv[1:] or ["teh", "recieve", "beleive", "wrld", "adress",
                             "definately", "the", "hello", "world"]
    for w in words:
        print(f"  {w:14} -> {corrector.correct(w)}")


if __name__ == "__main__":
    main()
