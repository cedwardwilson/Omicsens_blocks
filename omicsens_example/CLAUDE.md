# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Quaisr block** — a Python module deployed on the Quaisr platform. The block performs spectroscopic analysis for peptide weight estimation using machine learning inference on Raman spectra data.

## Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running
```bash
# Run the main inference pipeline
python main.py
```

### Quaisr CLI
```bash
# Deploy the block to the Quaisr platform
quaisr deploy
```

## Architecture

### Entry Point
`main.py` — contains `run_spectrum_inference()` which orchestrates the full inference pipeline:
1. Load spectral data for configured peptides (e.g. DGDVI, ESRVS) using `Data` class
2. Apply downsampling via `load_multiple_peptide_configs()`
3. Load a pretrained CNN2Layer model from `trained_models/`
4. Run `neural_network_regression_solver()` to predict peptide mixture weights
5. Perform Monte Carlo uncertainty quantification via `ParametricSpectrumAnalysis`
6. Generate plots (solution fit, sampling distributions, confidence intervals)

### `src/spectra_analytics/` — Core Library

| Module | Purpose |
|---|---|
| `dataset.py` | `Data` class for loading `.npy` spectra; `load_multiple_peptide_configs()` applies 18 downsampling strategies |
| `spectrum_expansion.py` | `SpectrumExpansion` — regression solvers (numpy, sklearn, PyTorch NN) and R² metrics |
| `statistical_analysis.py` | `ParametricSpectrumAnalysis` — Monte Carlo sampling, confidence intervals |
| `downsample_spectra.py` | `DownsampleSpectra` — 10 strategies: LTTB, VIP, SPA, CARS, resample, decimate, average, max, regular, poly |
| `denoise_spectra.py` | `DenoiseSpectra` — wavelet (db4), Fourier, and Savitzky-Golay filters |
| `simulated_dataset.py` | `SimulatedSpectraDataset` — synthetic spectrum generation for training |
| `ML/model.py` | Neural network architectures: `CNN2Layer`, `MLP`, `BilateralLSTM`, `RBFNeuralNetwork` |
| `ML/training.py` | Training loops: `DeepLearningTraining` (PyTorch, Visdom), `RegressorTraining` (sklearn) |
| `utils.py` | `load_checkpoint()`, `R2score()` |

### Trained Models (`trained_models/`)
Pretrained `.pth.tar` (PyTorch) and `.joblib` (sklearn) checkpoints. Checkpoint format: `{'config', 'state_dict', 'optimizer'}`. The default inference model is `CNN2Layer_16_32_3ACD.pth.tar`.

### Data Format
Spectral data is stored as `.npy` files (not included in this repo). The `Data` class loads data supporting H2O/D2O solvent conditions, simulated vs. experimental spectra, and water/no-water configurations.

## Quaisr Platform Notes

- `Quaisr.toml` defines block metadata (name, version, author, Python version, network access)
- `network-access.public = false` — this block runs without external network calls
- The `quaisr` package is a required dependency (provides platform integration)
