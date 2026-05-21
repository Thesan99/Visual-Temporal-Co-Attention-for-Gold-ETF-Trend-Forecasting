# Visual–Temporal Co-Attention for Gold ETF Trend Forecasting

This repository contains the author-generated materials supporting the paper:

> **Visual–Temporal Co-Attention for Gold ETF Trend Forecasting**
> Naseth Siv and Jaegi Jeon.

It includes the preprocessing pipeline, candlestick chart image construction, expanding-window split procedure, label construction, and summary CSV files used to generate the tables and figures in the paper.

---

## Data and Code Availability

The raw GLD ETF OHLCV data analyzed in this study are publicly available from Yahoo Finance and are **not redistributed in this repository**. The author-generated materials in this repository are made openly available for academic use under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## Data Source

- **Ticker:** GLD (SPDR Gold Shares ETF)
- **Source:** https://finance.yahoo.com/quote/GLD/history
- **Date range:** 3 January 2005 – 31 December 2025
- **Fields:** Open, High, Low, Close, Volume
- **Frequency:** Daily

The raw data are fetched directly from Yahoo Finance via the `yfinance` Python package; see the first cells of each notebook for the exact download call.

---

## Repository Structure
```
visual-temporal-coattention/
├── README.md
├── 1. Week 1/                                 # 5-day input-window
│   ├── 1. 1W Image Generating.ipynb           # (Phase 1) image construction
│   └── 2. 1W Data Processing.ipynb            # (Phase 2) data loader construction
├── 2. Month 1/                                # 20-day input-window
│   ├── 1. 1M Image Generating.ipynb
│   └── 2. 1M Data Processing.ipynb
├── 3. Month 3/                                # 60-day input-window
│   ├── 1. 3M Image Generating.ipynb
│   └── 2. 3M Data Processing.ipynb
├── appendix_results/                          # summary CSVs for appendix tables/figures
│   ├── 1. cv_schedule.csv
│   ├── 2. cv_sample_sizes.csv
│   ├── 3. class_distribution.csv
│   ├── 4. predictive_performance.csv
│   ├── 5. backtesting_performance.csv
│   └── 6. seed_stability_20day.csv
└── main_results/                              # summary CSVs for main paper tables/figures
├── combined_checkpoints_5day.csv
├── combined_checkpoints_20day.csv
├── combined_checkpoints_60day.csv
├── Fig. seed_stability_all_windows.csv
├── table3_predictive_20day.csv
├── table4_backtesting_20day.csv
└── table5_seed_stability_20day.csv
```
Each folder contains two notebooks: **Phase 1** constructs the candlestick chart images, and **Phase 2** builds the input tensors, labels, and PyTorch DataLoaders that feed the model.


---

## Pipeline Overview

The notebooks follow the same two-phase pipeline, differing only in input window length (5, 20, or 60 trading days). **Phase 1** constructs OHLC chart images from raw market data. **Phase 2** prepares the resulting arrays and labels into the tensor formats consumed by the model branches.

### Phase 1 — Image Construction (Input Creation)

This phase transforms raw daily OHLCV data into grayscale candlestick chart images, one image per sliding window.

1. **Download** raw daily OHLCV for GLD from Yahoo Finance (2005–2025).
2. **High/Low correction:** enforce `High = max(High, Open, Close)` and `Low = min(Low, Open, Close)` per row to ensure OHLC consistency. No corrections were required in the GLD dataset over the 2005–2025 period.
3. **Moving average:** compute the 20-day rolling mean of the close price (`MA20`) and drop the initial rows with missing values.
4. **Expanding-window splits:** construct 10 cross-validation passes. The training window starts in 2005 and grows by one year per pass, the validation year immediately follows the training window, and the test year immediately follows the validation year. Test years span 2016–2025.
5. **Window stitching:** the last `window_size + 4` rows of the validation set are stitched to the front of the test set to provide lookback context at the test-set boundary. This buffer covers the input window plus the 5-day label horizon and ensures no leakage of test labels into validation.
6. **Sliding-window tensors:** OHLCV+MA arrays are converted into sliding windows of shape `(num_samples, window_size, num_features)`. Within each window, the close-price column is replaced by a compounded path normalized to start at 1.0, and High, Low, Open, and MA20 are rescaled by the same factor so that the entire window lies on a comparable price scale.
7. **OHLC chart rendering:** each window is rendered as a compact grayscale OHLC + MA + Volume image. Each bar uses a center wick with a left tick for the open price and a right tick for the close price; the MA20 line is overlaid on the price panel, and a slim per-day volume bar appears in the bottom panel. Images are saved as PNG files under `charts_passes/pass{k}/{train,val,test}/`. These PNGs are the **image inputs** to the visual branch in Phase 2.

### Phase 2 — DataLoader Construction

This phase loads the Phase 1 outputs into the tensor structures consumed during model training.

#### 1. Visual branch (CNN and ViT inputs)

PNG charts are read from `charts_passes/` and stacked into a NumPy array of shape `(N, 1, H, W)` with values rescaled to `[0, 1]`. This array is consumed directly by the CNN branch and is cached to `X_img_cnn_cache.npz` to avoid re-reading PNGs on subsequent runs. For the ViT branch, the same array is reshaped into non-overlapping patches forming a token grid. The patch grid and image dimensions depend on the input window length; exact values are listed in the parameter table below.

#### 2. Temporal branch (sequence inputs)

The post-cleaning OHLCV+MA DataFrames are sliced into sliding windows of length `window_size` to form sequence tensors of shape `(N, window_size, num_features)`.

#### 3. Label construction

The binary classification target is the sign of the **5-day forward return of the close price**:

- **Class 1** if the 5-day forward return is positive.
- **Class 0** if the 5-day forward return is non-positive.

The label is computed per row and aligned with the last day of each input window. The first `window_size - 1` rows are dropped so that each label corresponds to a fully populated input window. The same 5-day label horizon is used across all three input-window settings.

#### 4. DataLoaders

The arrays from the three components above are wrapped into PyTorch `TensorDataset` and `DataLoader` objects for each split. Train loaders shuffle and drop the last partial batch; validation and test loaders preserve chronological order and keep all samples.

### Pipeline parameters per notebook

| Parameter | 5-day | 20-day | 60-day |
|---|---|---|---|
| Input window length | 5 | 20 | 60 |
| Val→test lookback buffer | 9 (4+5) | 24 (4+20) | 64 (4+60) |
| First-row label trim (`iloc[N:]`) | 4 | 19 | 59 |
| Chart image size (H × W) | 32 × 15 | 96 × 180 | 96 × 180 |
| ViT token grid | 8 × 5 (40 tokens) | 8 × 10 (80 tokens) | 8 × 20 (160 tokens) |
| ViT patch size | 4 × 3 | 8 × 6 | 12 × 9 |
| Temporal feature scaling | None | None | None |
| Label horizon | 5-day forward return | 5-day forward return | 5-day forward return |

---

## Summary CSV Files

### `main_results/`

Summary CSV files for the main paper tables and figures:

| File | Corresponding table / figure |
|---|---|
| `combined_checkpoints_5day.csv` | Per-pass and per-seed results — 5-day window |
| `combined_checkpoints_20day.csv` | Per-pass and per-seed results — 20-day window |
| `combined_checkpoints_60day.csv` | Per-pass and per-seed results — 60-day window |
| `table3_predictive_20day.csv` | Table 3 — Predictive performance at 20-day window |
| `table4_backtesting_20day.csv` | Table 4 — Backtesting performance at 20-day window |
| `table5_seed_stability_20day.csv` | Table 5 — Seed stability at 20-day window |
| `Fig. seed_stability_all_windows.csv` | Figure 7 (Avg ACC across model progression) and Figure 8 (Avg ACC vs CAGR scatter) |

### `appendix_results/`

Summary CSV files for the appendix tables and figures:

| File | Corresponding table / figure |
|---|---|
| `1. cv_schedule.csv` | Table A1 — Expanding-window evaluation schedule |
| `2. cv_sample_sizes.csv` | Table A2 — Windowed sample sizes per pass |
| `3. class_distribution.csv` | Table A3 — Class counts per pass |
| `4. predictive_performance.csv` | Table A4 — Predictive performance across all windows |
| `5. backtesting_performance.csv` | Table A5 & Figure E1 — Backtesting performance and risk-return profile (Vol% and CAGR% columns are used to generate Figure E1) |
| `6. seed_stability_20day.csv` | Table A6 — Seed-level performance ranges (20-day) |

---

## Installation

### Requirements

**Hardware:**
- GPU: NVIDIA GPU with 24 GB+ VRAM (e.g., RTX 3090) with CUDA 12.8 or later — required for training the deep learning models downstream.

**Software:**
- Python 3.9 or later
- Active internet connection (raw OHLCV data is fetched from Yahoo Finance at runtime)
- Approximately 500 MB of free disk space for generated chart images and intermediate caches

### Setup

```bash
git clone https://github.com/<username>/visual-temporal-coattention.git
cd visual-temporal-coattention

# (Recommended) create a clean virtual environment
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate

# Install dependencies
pip install --force-reinstall "numpy==1.26.4" "pandas==2.0.3"
pip install yfinance Pillow matplotlib imageio torch torchvision jupyter scikit-learn
```

### Standard Notebook Architecture

Each notebook follows a two-cell import design to isolate environment-level configurations from core frameworks:

**Cell 1 — Environment Setup:**
```python
import os
import sys
import time
```

**Cell 2 — Core Frameworks:**
```python
# Data analysis
import numpy as np
import pandas as pd
import yfinance as yf

# Deep learning
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

# Image processing
import imageio
from PIL import Image, ImageDraw, ImageFont

# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
```

---

## Reproducing the Pipeline

Each window-length folder contains two notebooks that should be run in order:

```bash
# 5-day input window
jupyter notebook "1. Week 1/1. 1W Image Generating.ipynb"
jupyter notebook "1. Week 1/2. 1W Data Processing.ipynb"

# 20-day input window
jupyter notebook "2. Month 1/1. 1M Image Generating.ipynb"
jupyter notebook "2. Month 1/2. 1M Data Processing.ipynb"

# 60-day input window
jupyter notebook "3. Month 3/1. 3M Image Generating.ipynb"
jupyter notebook "3. Month 3/2. 3M Data Processing.ipynb"
```

**Phase 1 (Image Generating)** downloads the data, constructs the expanding-window splits, and renders the candlestick chart images to `charts_passes/`. **Phase 2 (Data Processing)** loads the PNGs, builds the sequence tensors and binary labels, and assembles the PyTorch DataLoaders.

### Notes

- The `!nvidia-smi` cell at the top of each notebook is a hardware check carried over from the original Colab/Kaggle environment. It can be skipped; it does not affect the pipeline.
- The `pip install yfinance` cell inside the notebooks is redundant once dependencies are installed via the setup command above.
- Generated images and intermediate files (`charts_passes/`, `X_splits/`, `X_img_cnn_cache.npz`) are large and are regenerated on each run; they are not committed to the repository.

---

## How to Cite

If you use these materials, please cite both the paper and the repository.

**Paper:**

```

```

**Repository:**

```
https://github.com/Thesan99/Visual-Temporal-Co-Attention-for-Gold-ETF-Trend-Forecasting
```

---
