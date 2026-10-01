# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-10-01

A rebuild that makes the project actually run. The idea is unchanged — invisible
client-side neural autocorrect in the browser — but the model now corrects, and
the in-browser inference that was never finished now works.

### Fixed
- **In-browser inference never ran.** `sandbox.html` loaded `js/lib/tf.min.js`,
  which was never committed, and the TensorFlow.js model it expected was never
  exported. Replaced the whole TensorFlow.js path with a **dependency-free
  JavaScript inference engine** (`extension/js/inference.js`) that runs the LSTM
  encoder/decoder by hand — lighter than shipping TF.js, and it actually loads.
- **The model collapsed to frequent words.** The old checkpoint emitted generic
  words regardless of input: it was trained on whole sentences but the extension
  feeds it one word at a time. Retrained **word-by-word** to match real usage.
- **The model corrupted correct words.** Added a **dictionary gate**: a known
  word is never changed, and a proposal is only used if it is itself a real word
  (0% corruption of correct words on a held-out sample).
- **The README referenced files that did not exist** (a `docs/` tree, a
  `convert_direct.py` script). Documentation now matches the repository.

### Added
- Clean, reproducible pipeline: `01_build_dataset.py` (corpus → dictionary),
  `02_train_model.py` (word-level seq2seq), `03_export_for_browser.py`
  (framework-independent weights).
- Shared inference with a **beam search + dictionary gate** in both Python
  (`ai_model/src/inference.py`, NumPy only) and JavaScript, which roughly
  doubles the correction rate versus greedy decoding.
- A bundled `sample_corpus.txt` so the whole pipeline runs out of the box
  without the multi-hundred-MB training corpus.
- `extension/demo.html` — try the corrector in a browser with no install.
- Tests (pytest for inference and typo generation; a Node test asserting the
  browser engine matches the Python reference) and CI (ruff, Python 3.10–3.12,
  the JS parity test).

### Changed
- Model is smaller (~0.2M parameters, ~3 MB) and trains in minutes on CPU.
- Replaced the Conda environment files with a `requirements.txt`; inference
  needs only NumPy, and the browser needs nothing.

### Removed
- The large training corpus from the working tree (kept out of HEAD; the public
  Leipzig corpus is linked instead) and the stale TensorFlow.js conversion path.

## [1.0.0] - 2025-10
- Initial prototype: a character-level seq2seq training pipeline and a Manifest
  V3 extension intended to run the model via TensorFlow.js in a sandbox.
