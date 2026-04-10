# Alternative Data in Quantitative Trading
### NLP Sentiment Signal → Stock Return Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange)](.)
[![Institution](https://img.shields.io/badge/SIT%20Pune-B.Tech%20AI%2FML-navy)](.)

---

## What This Project Does

This project builds an end-to-end **alternative data pipeline** that:

1. Scrapes public Reddit posts from financial subreddits (r/wallstreetbets, r/investing, r/stocks)
2. Scores each post using **FinBERT** — a BERT model fine-tuned on financial text
3. Constructs a **daily weighted sentiment signal** per stock ticker
4. Tests whether that signal statistically predicts **next-day stock returns**
5. Backtests a simple long/short strategy driven entirely by sentiment
6. Runs a full **time series analysis** (stationarity, ACF, ARIMA forecasting) on the signal
7. Includes an **ethics audit** examining the consent, privacy, and fairness implications of using Reddit data for institutional trading

The project demonstrates the complete lifecycle of an alternative data signal — from raw scraping to statistical validation — and critically examines whether this practice is ethically justified.

---

## Project Structure

```
alt-data-quant-research/
│
├── alt_data_sentiment.py       # Main pipeline (12 cells, run in Google Colab)
├── README.md                   # This file
├── DOCUMENTATION.md            # Full technical + mathematical documentation
│
├── outputs/                    # Generated when you run the notebook
│   ├── alt_data_research.png   # 4-panel research plot
│   └── time_series_analysis.png # 3-panel time series plot
│
└── requirements.txt            # Python dependencies
```

---

## Quickstart

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `alt_data_sentiment.py` or paste each cell manually
3. Run **Cell 1** first (installs all dependencies — takes ~3 minutes)
4. Run **Cell 3B** if you do not have Reddit API credentials (uses realistic sample data)
5. Run **Cells 4 through 12** in order

### Option B — Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/alt-data-quant-research
cd alt-data-quant-research
pip install -r requirements.txt
jupyter notebook
```

---

## Dependencies

```
praw>=7.7.0          # Reddit API scraping
transformers>=4.35   # FinBERT (HuggingFace)
torch>=2.0           # PyTorch backend for FinBERT
yfinance>=0.2.28     # Stock price data
pandas>=2.0          # Data manipulation
numpy>=1.24          # Numerical computation
matplotlib>=3.7      # Visualisation
seaborn>=0.12        # Statistical plots
scipy>=1.11          # Pearson correlation, OLS regression
statsmodels>=0.14    # ARIMA, ADF test, ACF/PACF
mlfinlab>=0.16       # Lopez de Prado's financial ML library
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Reddit API Setup

1. Go to [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
2. Click **"create another app"**
3. Fill in:
   - Name: `altdata_research_bot`
   - Type: `script`
   - Redirect URI: `http://localhost:8080`
4. Copy your **Client ID** (under the app name) and **Client Secret**
5. In Cell 3A, replace:
```python
REDDIT_CLIENT_ID     = "YOUR_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
USE_LIVE_REDDIT      = True
```

If you do not have credentials yet, Cell 3B generates realistic sample data so the full pipeline still runs.

---

## Pipeline Overview

| Cell | Purpose | Key Output |
|------|---------|-----------|
| 1 | Install dependencies | All packages ready |
| 2 | Imports and plot styling | Dark-theme matplotlib config |
| 3A | Live Reddit scraping (PRAW) | Raw DataFrame of posts |
| 3B | Sample data fallback | Simulated 250 posts × 5 tickers |
| 4 | FinBERT sentiment scoring | compound score ∈ [−1, +1] per post |
| 5 | yfinance price download | OHLCV + forward returns |
| 6 | Daily signal aggregation | Weighted sentiment score per (ticker, day) |
| 7 | Signal-return merge | Aligned panel DataFrame |
| 8 | Statistical tests | Pearson r, quintile returns, OLS regression |
| 9 | 4-panel visualisation | `alt_data_research.png` |
| 10 | Long/short backtest | Strategy vs buy-and-hold returns |
| 11 | Ethics audit | Contextual integrity analysis table |
| 12 | Time series analysis | ADF test, ACF plot, ARIMA(1,0,1) forecast |

---

## Key Results

Running on sample data, the pipeline produces:

- **Sentiment distribution** across positive / negative / neutral for all 5 tickers
- **Quintile return table** — sorted from most bearish to most bullish sentiment days
- **OLS regression** — slope, intercept, R², and p-value for sentiment → return prediction
- **Backtest table** — strategy return vs buy-and-hold, trade count, hit rate
- **ADF stationarity test** result on the sentiment signal
- **5-day ARIMA forecast** with 95% confidence intervals

---

## Tickers Covered

`GME` · `NVDA` · `TSLA` · `AAPL` · `AMD`

---

## Mathematics Used

| Concept | Formula |
|---------|---------|
| Softmax | P(class_i) = exp(z_i) / Σ exp(z_j) |
| Compound score | compound = P(positive) − P(negative) |
| Weighted sentiment | S_t = Σ(w_i × c_i) / Σ w_i |
| Log return | log_r = ln(P_t / P_{t−1}) |
| Pearson r | r = Σ[(S−S̄)(r−r̄)] / sqrt[Σ(S−S̄)² × Σ(r−r̄)²] |
| OLS slope | β̂ = Σ[(S_t−S̄)(r_t−r̄)] / Σ(S_t−S̄)² |
| Sharpe Ratio | (μ − r_f) / σ × sqrt(252) |
| ARIMA(1,0,1) | S_t = μ + φ₁S_{t−1} + θ₁ε_{t−1} + ε_t |

Full derivations with theory in [DOCUMENTATION.md](DOCUMENTATION.md)

---

## Ethics Audit Summary

| Data Type | Legal Risk | Ethics Risk |
|-----------|-----------|------------|
| Reddit NLP (this project) | Low | Low–Medium |
| Satellite imagery (public) | Low | Low |
| Credit card transactions | Medium | High |
| Smartphone geolocation | High (CCPA) | Very High |
| Web scraping without ToS | High (CFAA) | High |

The core ethical tension: Reddit users post opinions expecting other retail investors to read them. When those opinions are aggregated by an ML pipeline to generate institutional trading signals, their data flows into a context they never consented to — a violation of **contextual integrity** (Nissenbaum, 2004).

---

## Roadmap / Next Steps

- [ ] Fine-tune FinBERT on r/wallstreetbets-specific vocabulary
- [ ] Replace time bars with **dollar bars** (López de Prado Chapter 2)
- [ ] Apply **triple-barrier labelling** for better ML target construction
- [ ] Add **Deflated Sharpe Ratio** to account for multiple testing
- [ ] Implement **Combinatorial Purged Cross-Validation (CPCV)**
- [ ] Extend to NSE/BSE mid-cap tickers using Indian financial subreddits
- [ ] Integrate with **Zerodha Kite API** for live paper trading

---

## Academic References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Chan, E. (2009). *Quantitative Trading*. Wiley.
- Nissenbaum, H. (2004). Privacy as Contextual Integrity. *Washington Law Review*, 79(1).
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with BERT. *arXiv:1908.10063*.
- Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds. *JFE*.
- Dickey, D. & Fuller, W. (1979). Distribution of estimators for autoregressive time series. *JASA*.

---

## Author

**Khullar** — B.Tech AI/ML, SIT Pune  
Research focus: Quantitative trading, alternative data, NLP signal construction  
Target: Quant Research roles at Jane Street · Citadel · Optiver · IMC

---

## License

MIT License — free to use for academic and research purposes.  
If you use this pipeline in your own research, please cite this repository.
