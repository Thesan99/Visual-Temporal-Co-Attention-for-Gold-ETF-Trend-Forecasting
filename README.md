# Visual-Temporal-Co-Attention-for-Gold-ETF-Trend-Forecasting
This study examines whether synchronized OHLC chart images and numerical price-volume sequences provide complementary information for gold ETF trend forecasting and whether this effect depends on the input window. We evaluate a multimodal framework.

## Data and Code Availability

The raw GLD ETF OHLCV data analyzed in this study are publicly available from Yahoo Finance and are **not redistributed in this repository**. The author-generated materials in this repository — preprocessing notebooks, split and label construction code, and summary CSV files — are released under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. Source code is additionally released under the MIT License.

A citable version of this repository is archived at Zenodo with DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX).

---

## Data Source

- **Ticker:** GLD (SPDR Gold Shares ETF)
- **Source:** https://finance.yahoo.com/quote/GLD/history
- **Date range:** 2 January 2005 – 31 December 2025
- **Fields:** Open, High, Low, Close, Volume
- **Frequency:** Daily

The raw data are fetched directly from Yahoo Finance via the `yfinance` Python package; see `notebooks/01_preprocessing_5day.ipynb` (first cells) for the exact download call.

---

## Repository Structure

```
visual-temporal-coattention/
├── README.md
├── LICENSE              # MIT (code)
├── LICENSE-DATA         # CC BY 4.0 (materials and CSVs)
├── CITATION.cff
├── requirements.txt
├── notebooks/
│   ├── 01_preprocessing_5day.ipynb     # 5-day window pipeline
│   ├── 02_preprocessing_20day.ipynb    # 20-day window pipeline
│   └── 03_preprocessing_60day.ipynb    # 60-day window pipeline
└── results/
    ├── cv_schedule.csv
    ├── cv_sample_sizes.csv
    ├── class_distribution.csv
    ├── predictive_performance.csv
    ├── backtesting_performance.csv
    ├── seed_stability_20day.csv
    └── risk_return.csv
```

---

## Preprocessing Pipeline

The three notebooks under `notebooks/` implement the same preprocessing pipeline, differing only in the input window length (5, 20, or 60 trading days). Each notebook performs the following steps:

1. **Download** raw daily OHLCV for GLD from Yahoo Finance (2005–2025).
2. **High/Low correction:** ensure `High = max(High, Open, Close)` and `Low = min(Low, Open, Close)` per row to repair any inconsistencies in the raw feed.
3. **Moving average:** compute the 20-day rolling mean of the close price (`MA20`) and drop the initial rows with missing values.
4. **Expanding-window splits:** construct 10 cross-validation passes. The training window starts in 2005 and grows by one year per pass, the validation year immediately follows the training window, and the test year immediately follows the validation year. Test years span 2016–2025.
5. **Window stitching:** the last `window_size - 1` rows of the validation set are stitched to the front of the test set to provide lookback context at the test-set boundary without leaking labels.
6. **Return feature:** a backward-looking 4-day simple return is appended for inspection (not used as a target).
7. **Sliding-window tensors:** OHLCV+MA arrays are converted into sliding windows of shape `(num_samples, window_size, num_features)`. Within each window, the close-price column is replaced by a compounded path normalized to start at 1.0; High, Low, Open, and MA20 are rescaled by the same factor so that the entire window is on a comparable price scale.
8. **OHLC chart rendering:** each window is rendered as a compact grayscale OHLC + MA + volume image (`H=32` pixels, `px_per_bar=3`). Images are saved as PNG files under `charts_passes/pass{k}/{train,val,test}/`.

## Label Construction

The binary classification target is the sign of the **5-day forward return of the close price**:

- **Class 1** if the 5-day forward return is positive,
- **Class 0** if the 5-day forward return is non-positive.

The label is computed per row and aligned to the date at which the corresponding input window ends. The same target definition is used across all three input-window settings.

---

## Summary CSV Files

The `results/` directory contains the values used to generate the tables and figures in the paper:

| File | Corresponding table / figure |
|---|---|
| `cv_schedule.csv` | Table A1 — expanding-window evaluation schedule |
| `cv_sample_sizes.csv` | Table A2 — windowed sample sizes per pass |
| `class_distribution.csv` | Table A3 — class counts per pass |
| `predictive_performance.csv` | Table A4 — predictive performance across windows |
| `backtesting_performance.csv` | Table A5 — backtesting performance across windows |
| `seed_stability_20day.csv` | Table A6 — seed-level performance ranges (20-day) |
| `risk_return.csv` | Figure A1 — risk-return profile data |

---

## Reproducing the Pipeline

```bash
git clone https://github.com/<username>/visual-temporal-coattention.git
cd visual-temporal-coattention
pip install -r requirements.txt
jupyter notebook notebooks/01_preprocessing_5day.ipynb
```

The notebook downloads the raw data, constructs splits and labels, and renders the chart images directly to the local filesystem. The 20-day and 60-day notebooks follow the identical structure with `window_size` set accordingly.

---

## How to Cite

If you use these materials, please cite both the paper and the archived repository.

**Paper:**

```bibtex
@article{<lastname>2026vtcoattention,
  title   = {Visual--Temporal Co-Attention for Gold ETF Trend Forecasting},
  author  = {<Last Name>, Thesan and Jeon, Jaegi},
  journal = {Applied Artificial Intelligence},
  year    = {2026}
}
```

**Repository (Zenodo archive):**

```bibtex
@misc{<lastname>2026repo,
  author    = {<Last Name>, Thesan and Jeon, Jaegi},
  title     = {{Visual--Temporal Co-Attention for Gold ETF Trend Forecasting: Code and Materials}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## License

- **Source code** (notebooks, scripts): MIT License — see [`LICENSE`](LICENSE).
- **Materials and data files** (CSVs, documentation): Creative Commons Attribution 4.0 International (CC BY 4.0) — see [`LICENSE-DATA`](LICENSE-DATA).

The raw GLD OHLCV data are property of their respective providers and are not redistributed here.
