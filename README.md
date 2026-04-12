# Alternative Data in Quantitative Trading

> **NLP Sentiment Signal → Stock Return Prediction**  
> Ethical AI Case Study · CO3 · CO4 · CO5  
> B.Tech AI/ML · SIT Pune · 2025-26

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://python.org)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00)](https://colab.research.google.com)
[![FinBERT](https://img.shields.io/badge/Model-FinBERT-FF6F00)](https://huggingface.co/ProsusAI/finbert)
[![License](https://img.shields.io/badge/License-MIT-10b981)](LICENSE)

---

## What This Project Does

This repository implements a complete **alternative data research pipeline** that:

1. Scrapes public Reddit posts from financial subreddits using the PRAW API
2. Scores each post with **FinBERT** — a BERT transformer fine-tuned on financial text
3. Constructs a daily **importance-weighted sentiment signal** per stock ticker
4. Tests whether that signal statistically predicts **next-day stock returns**
5. Backtests a long/short strategy and measures **Information Coefficient (IC)**
6. Runs a full **time series analysis**: ADF stationarity, ACF, ARIMA(1,0,1) forecast
7. Includes a rigorous **ethics audit** using Nissenbaum's contextual integrity framework

---

## Repository Structure

```
alt-data-quant-research/
│
├── alt_data_sentiment_v2.py         # Main pipeline (17 cells, Google Colab)
├── README.md                        # This file
│
├── docs/
│   ├── CA3_Report.docx              # Full academic report (20–23 pages)
│   ├── CA3_Presentation.pptx        # 10-slide presentation
│   ├── Alt_Data_Documentation.docx  # Technical + math documentation
│   └── Alt_Data_Mathematics.docx    # Formula reference sheet
│
└── outputs/
    ├── alt_data_research.png        # 4-panel research overview
    ├── time_series_analysis.png     # ADF + ACF + ARIMA
    ├── sentiment_vs_price.png       # Visual correlation proof
    ├── sentiment_heatmap.png        # Cross-ticker correlation matrix
    ├── equity_curves.png            # 5-ticker backtest equity curves
    └── ic_analysis.png              # IC / ICIR analysis
```

---

## Pipeline Architecture

```
Reddit API (PRAW)
      │
      ▼
 Post Titles ──► FinBERT ──► P(pos), P(neg), P(neu)
                                    │
                             compound = P(pos) - P(neg)
                                    │
                        weight: w_i = √(score_i + 1)
                                    │
                   Daily Signal: S_t = Σ(w_i × c_i) / Σ(w_i)
                                    │
         ┌──────────────────────────┼──────────────────────┐
         ▼                          ▼                      ▼
   Statistical Tests           Backtest              Time Series
   Pearson r, Spearman         Long  if S_t > 0.10   ADF stationarity
   Kendall τ, OLS              Short if S_t < -0.10  ACF, ARIMA(1,0,1)
   Quintile analysis           Sharpe Ratio           IC / ICIR
```

---

## Quickstart

### No Reddit credentials (run tonight)
1. Open [Google Colab](https://colab.research.google.com)
2. Paste each cell from `alt_data_sentiment_v2.py`
3. Run Cell 1 → skip 3A → run Cell 3B → run Cells 4–17

### With Reddit credentials
1. Get free credentials at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. In Cell 3A set `USE_LIVE_REDDIT = True` and paste your credentials
3. Run Cell 3A, skip 3B, run Cells 4–17

---

## Cell Reference

| Cell | What it does | Output |
|------|-------------|--------|
| 1–2 | Install packages, imports, config | Environment ready |
| 3A | Live Reddit scraping (PRAW API) | Raw post DataFrame |
| 3B | Sample data fallback (seed=42) | 250 synthetic posts |
| 4 | FinBERT sentiment scoring | compound ∈ [−1, +1] per post |
| 5 | yfinance OHLCV + forward returns | price_data dict |
| 6 | Weighted daily aggregation | daily_sentiment DataFrame |
| 7 | Merge signal + price panel | signal_df |
| 8 | Pearson r, quintiles, OLS regression | Printed statistics |
| 9 | 4-panel research visualisation | alt_data_research.png |
| 10 | Long/short backtest | Backtest results table |
| 11 | Ethics audit | Contextual integrity table |
| 12 | ADF + ACF + ARIMA(1,0,1) | time_series_analysis.png |
| 13 | Sentiment vs price overlay | sentiment_vs_price.png |
| 14 | Pearson + Spearman + Kendall | All three correlation tests |
| 15 | Cross-ticker heatmap | sentiment_heatmap.png |
| **16** | **IC / ICIR (hedge fund standard)** | ic_analysis.png |
| 17 | 5-ticker equity curves | equity_curves.png |

---

## Key Mathematics

| Concept | Formula |
|---------|---------|
| Softmax | `P(i) = exp(z_i) / Σ exp(z_j)` |
| Compound score | `c = P(positive) - P(negative)` |
| Post weight | `w_i = √(score_i + 1)` |
| Daily signal | `S_t = Σ(w_i·c_i) / Σw_i` |
| Forward return | `r_{t+1} = (P_{t+1} - P_t) / P_t` |
| Pearson r | `Σ[(S-S̄)(r-r̄)] / √[Σ(S-S̄)²·Σ(r-r̄)²]` |
| OLS slope | `β̂ = Σ[(S_t-S̄)(r_t-r̄)] / Σ(S_t-S̄)²` |
| Sharpe Ratio | `(μ - r_f) / σ × √252` |
| ARIMA(1,0,1) | `S_t = μ + φ₁S_{t-1} + θ₁ε_{t-1} + ε_t` |
| IC | `Spearman(rank(S_t), rank(r_{t+1}))` |
| ICIR | `mean(IC) / std(IC)` |

Full derivations: [docs/Alt_Data_Documentation.docx](docs/Alt_Data_Documentation.docx)

---

## Why Non-Significant on Sample Data?

Cell 14 shows p > 0.05 for all tickers on synthetic data. **This is correct.**
No real relationship was built into the random sample. A model that shows
significance on pure noise is broken — ours correctly finds nothing.

On live Reddit data for high-attention stocks, literature reports:
- Bollen et al. (2011): Twitter mood predicts DJIA 3 days ahead
- Chen et al. (2014): Seeking Alpha → significant abnormal returns
- Expected r ≈ 0.10–0.25 for meme stocks like GME/NVDA

---

## Ethics Summary

**Contextual Integrity (Nissenbaum, 2004):** Reddit users post for community
discussion. Their data flows to institutional trading profit without consent,
disclosure, or compensation. Technically legal under Mosaic Theory — ethically
problematic under contextual integrity.

| Data Type | Legal Risk | Ethics Risk |
|-----------|-----------|------------|
| Reddit NLP (this project) | Low | Low–Medium |
| Satellite imagery | Low | Low |
| Credit card transactions | Medium | High |
| Smartphone geolocation | High | Very High |

**Recommendations:** Differential privacy · FinBERT domain adaptation ·
Regulatory disclosure · Open-source publication

---

## Academic References (selected)

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Araci, D. (2019). FinBERT. arXiv:1908.10063.
- Nissenbaum, H. (2004). Privacy as Contextual Integrity. *Washington Law Review*.
- Bollen, J. et al. (2011). Twitter mood predicts the stock market.
- Devlin, J. et al. (2019). BERT. NAACL-HLT.
- Zuboff, S. (2019). *The Age of Surveillance Capitalism*.

Full 34-reference list in [docs/CA3_Report.docx](docs/CA3_Report.docx)

---

## Author

**Khullar** — B.Tech AI/ML, SIT Pune  
QuantLab: Multi-semester factor model for Indian equities  
Target: Quant Research — Jane Street · Citadel · Optiver

---

*MIT License — free for academic and research use.*
