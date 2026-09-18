# Ghost Type Corrector

AI-powered browser extension providing invisible, contextual autocorrection with a seamless mobile keyboard experience on desktop browsers.

Advanced rebuild of the initial prototype (https://github.com/JAMPANIKOMAL/invisible-autocorrect-extension) using a character-level sequence-to-sequence neural network instead of frequency dictionaries.

## Project evolution

This is Phase 2 of a 4-part exploration into automatic typing correction.
It successfully trains a real seq2seq model (below), but integrating it
*inside the browser extension sandbox* proved too heavy — see Known
Limitations. That specific problem motivated the next phase,
[Type-Correcter-Ai](https://github.com/JampaniKomal/Type-Correcter-Ai),
which moved the same trained model into a Flask web app instead, and
eventually [AI_Corrector_Project](https://github.com/JampaniKomal/AI_Corrector_Project),
a full cross-application desktop corrector.

## Features

- **Invisible Correction:** No popups or underlines. Typos are corrected instantly on spacebar press.
- **Contextual AI:** Fixes spelling errors and grammatical mistakes using context understanding.
- **Mobile-Style Undo:** Press Backspace once to revert unwanted corrections.
- **Browser-Native Feel:** Disables default spellcheck for a clean typing experience.

## Quick Start

```powershell
# 1. Create environment
conda env create -f environment-gpu.yml
conda activate ghost-corrector-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install tensorflowjs==3.18.0

# 2. Train model
cd ai_model\src
python 01_data_preprocessing.py
python 02_model_training.py
python convert_direct.py
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed commands.

## Documentation

**Complete documentation available in the `docs/` folder:**

- **[docs/README.md](docs/README.md)** - Documentation index and navigation guide
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Complete setup instructions
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Fast reference commands  
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** - Training parameters and tuning
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[docs/API.md](docs/API.md)** - Technical reference and code documentation

## Project Structure

```
Ghost-Type-Corrector/
├── docs/                            # Documentation
│   ├── INSTALLATION.md              # Complete setup guide
│   ├── QUICKSTART.md                # Fast reference
│   ├── CONFIGURATION.md             # Parameter tuning
│   ├── TROUBLESHOOTING.md           # Common issues
│   └── API.md                       # Technical reference
│
├── ai_model/                        # Model development
│   ├── src/
│   │   ├── 01_data_preprocessing.py # Data cleaning and typo generation
│   │   ├── 02_model_training.py     # Seq2seq LSTM training
│   │   └── convert_direct.py        # TensorFlow.js conversion
│   ├── data/                        # Training data (generated)
│   ├── notebooks/                   # Research notebooks
│   └── autocorrect_model.h5         # Trained model (generated)
│
├── extension/                       # Browser extension
│   ├── model/                       # TensorFlow.js model (generated)
│   ├── assets/
│   ├── js/
│   └── manifest.json
│
├── environment-gpu.yml              # GPU training environment
├── environment-cpu.yml              # CPU training environment
├── README.md                        # This file
└── LICENSE                          # MIT License
```

## Technical Details

- **Model:** Character-level sequence-to-sequence LSTM (encoder-decoder)
- **Framework:** TensorFlow 2.10 with Keras API
- **Deployment:** TensorFlow.js for browser inference
- **Training:** GPU-accelerated (CUDA 11.2) or CPU
- **Dataset:** Leipzig Corpora Collection (British English, 1M sentences)

## Performance

| Hardware    | Training Time | Model Size | Accuracy |
|-------------|---------------|------------|----------|
| RTX 3080    | 5 min         | 3 MB       | 60-70%   |
| RTX 3050    | 7 min         | 3 MB       | 60-70%   |
| Intel i7    | 50 min        | 3 MB       | 60-70%   |

## Known limitations

- **The extension's browser-side inference is incomplete.**
  `extension/sandbox.html` loads `js/lib/tf.min.js` (TensorFlow.js), but
  that file was never committed and there's no fetch/build step that
  produces it — the extension as checked in cannot actually run inference
  in the browser. The model-training pipeline (`ai_model/`) itself is real
  and works; only the in-browser deployment half is unfinished. This is
  the specific limitation documented in this project's successor,
  [Type-Correcter-Ai](https://github.com/JampaniKomal/Type-Correcter-Ai),
  which moved the same trained model to a Flask web app instead of
  fighting the browser sandbox further.

## Acknowledgements

- Dataset: Leipzig Corpora Collection, Leipzig University

## License

MIT License - See LICENSE file for details.
