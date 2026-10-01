#!/usr/bin/env python3
"""Step 3 - export the trained model to framework-independent weights.

Pulls the raw weight matrices out of the Keras model and writes them twice:

* ``extension/model/weights.npz`` - NumPy, read by ai_model/src/inference.py
* ``extension/model/weights.json`` - base64 float32, read by the browser's
  extension/js/inference.js

It also copies the tokenizer and dictionary next to them. After this the model
runs with only NumPy (Python) or plain JavaScript (browser) - no TensorFlow and
no TensorFlow.js, which is the whole point: the extension never again needs a
toolchain to rebuild a servable model.

Usage: python 03_export_for_browser.py
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np
from tensorflow import keras

AI = Path(__file__).resolve().parent.parent
DATA = AI / "data"
MODEL_DIR = AI.parent / "extension" / "model"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(str(AI / "autocorrect_model.h5"), compile=False)

    enc_emb = model.get_layer("encoder_embedding").get_weights()[0]
    enc_k, enc_rk, enc_b = model.get_layer("encoder_lstm").get_weights()
    dec_emb = model.get_layer("decoder_embedding").get_weights()[0]
    dec_k, dec_rk, dec_b = model.get_layer("decoder_lstm").get_weights()
    den_k, den_b = model.get_layer("output_dense").get_weights()

    arrays = dict(enc_emb=enc_emb, enc_k=enc_k, enc_rk=enc_rk, enc_b=enc_b,
                  dec_emb=dec_emb, dec_k=dec_k, dec_rk=dec_rk, dec_b=dec_b,
                  den_k=den_k, den_b=den_b)
    arrays = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in arrays.items()}

    np.savez(MODEL_DIR / "weights.npz", **arrays)
    meta = {"U": int(enc_rk.shape[0]), "V": int(enc_emb.shape[0]), "EMB": int(enc_emb.shape[1])}
    tensors = {k: base64.b64encode(v.tobytes()).decode("ascii") for k, v in arrays.items()}
    (MODEL_DIR / "weights.json").write_text(json.dumps({"meta": meta, "tensors": tensors}))

    for name in ("tokenizer_config.json", "dictionary.txt"):
        shutil.copyfile(DATA / name, MODEL_DIR / name)

    print(f"exported to {MODEL_DIR}:")
    for p in sorted(MODEL_DIR.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
