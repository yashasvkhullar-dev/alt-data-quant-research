# 📊 Alternative Data in Quantitative Trading
### NLP Sentiment Signal from Reddit → Stock Return Prediction

> **Course:** AI Systems Engineering and Ethics (AISEE) — CA3  
> **Authors:** Yashasv Khullar · Vibhuti Patil · Uddish Agarwal · Tanvi Agrawal  
> **Institution:** MIT School of Engineering  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Explained Step by Step](#3-pipeline-explained-step-by-step)
4. [FinBERT Model Deep Dive](#4-finbert-model-deep-dive)
5. [Signal Engineering](#5-signal-engineering)
6. [Statistical Analysis Methods](#6-statistical-analysis-methods)
7. [Backtesting Engine](#7-backtesting-engine)
8. [Extended Analysis (New)](#8-extended-analysis-new)
9. [Ethical Implications](#9-ethical-implications)
10. [How to Run](#10-how-to-run)
11. [Output Files](#11-output-files)
12. [Project Structure](#12-project-structure)

---

## 1. Project Overview

This project investigates whether **alternative data** — specifically Reddit social media sentiment — contains statistically significant predictive information about short-term stock returns.

### What is Alternative Data?

Traditional quantitative trading relies on:
- Price and volume history
- Financial statements (earnings, balance sheets)
- Macroeconomic indicators

**Alternative data** extends this with non-traditional sources:

| Source | Example |
|---|---|
| Social media | Reddit posts, Twitter/X mentions |
| Satellite imagery | Retail parking lot density |
| Credit card data | Consumer spending per sector |
| Web scraping | Job postings, product reviews |
| Mobile location | Foot traffic to stores |

This project focuses on **Reddit sentiment** because it is high-frequency, publicly available, and demonstrably moves prices (see: GameStop 2021 short squeeze).

### Research Questions

1. Does Reddit sentiment correlate with next-day stock returns?
2. Is the relationship statistically significant across Pearson, Spearman, and Kendall tests?
3. Can an ARIMA model forecast sentiment time series?
4. Does a sentiment-driven long/short strategy outperform buy-and-hold on a risk-adjusted basis (Sharpe ratio)?
5. Do sentiment signals Granger-cause returns, or do returns drive sentiment?

### Tickers Studied

`GME` · `NVDA` · `TSLA` · `AAPL` · `AMD`

---

## 2. System Architecture

The pipeline follows a classic **Extract → Transform → Load → Analyze** pattern adapted for financial machine learning.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                      │
│                                                                           │
│   ┌──────────────────┐          ┌───────────────────────┐               │
│   │  Reddit API       │          │  Yahoo Finance API     │               │
│   │  (PRAW library)   │          │  (yfinance library)    │               │
│   │                   │          │                        │               │
│   │  Posts, scores,   │          │  OHLCV price data,     │               │
│   │  comments,        │          │  volume, adjusted      │               │
│   │  upvote ratios    │          │  close prices          │               │
│   └────────┬─────────┘          └───────────┬────────────┘               │
│            │                                │                             │
└────────────┼────────────────────────────────┼─────────────────────────────┘
             │                                │
             ▼                                ▼
┌────────────────────────┐      ┌─────────────────────────────┐
│  STEP 1: Collection     │      │  STEP 4: Price Feature Eng. │
│                         │      │                              │
│  • Live scraping (PRAW) │      │  return_1d  = Δclose (t+1)  │
│  • OR synthetic data    │      │  return_2d  = Δclose (t+2)  │
│    generator (default)  │      │  return_5d  = Δclose (t+5)  │
│                         │      │  log_return = ln(Pt/Pt-1)   │
│  Output:                │      │  volatility = 5d rolling σ  │
│  raw_df (250+ rows)     │      │  volume_rel = vol/20d avg   │
└──────────┬──────────────┘      └──────────────┬──────────────┘
           │                                     │
           ▼                                     │
┌──────────────────────────────────────────┐     │
│  STEP 2: NLP Sentiment Scoring (FinBERT) │     │
│                                          │     │
│  Input: Post title text                  │     │
│                                          │     │
│  ┌─────────────────────────────────┐     │     │
│  │ BERT Tokenizer                  │     │     │
│  │   ↓                             │     │     │
│  │ 12-layer Transformer Encoder    │     │     │
│  │   ↓                             │     │     │
│  │ [CLS] representation            │     │     │
│  │   ↓                             │     │     │
│  │ Linear classifier head          │     │     │
│  │   ↓                             │     │     │
│  │ Softmax → [P_pos, P_neg, P_neu] │     │     │
│  └─────────────────────────────────┘     │     │
│                                          │     │
│  Output per post:                        │     │
│    label    : positive / negative /      │     │
│               neutral                    │     │
│    compound : P_positive − P_negative    │     │
│               range: [−1, +1]            │     │
└──────────┬───────────────────────────────┘     │
           │                                     │
           ▼                                     │
┌──────────────────────────────────────────┐     │
│  STEP 3: Daily Signal Aggregation        │     │
│                                          │     │
│  Group by (ticker, date)                 │     │
│                                          │     │
│  Weighted avg sentiment:                 │     │
│    weight_i = √(score_i + 1)             │     │
│    S_daily  = Σ(w_i × compound_i)/Σw_i  │     │
│                                          │     │
│  Additional features:                    │     │
│    post_count        (attention signal)  │     │
│    avg_upvote_ratio  (consensus)         │     │
│    total_comments    (engagement depth)  │     │
│    sentiment_std     (intra-day noise)   │     │
│    bull_ratio        (bullish %)         │     │
│    sentiment_momentum (day-over-day Δ)   │     │
│    volume_zscore     (abnormal activity) │     │
└──────────┬───────────────────────────────┘     │
           │                                     │
           └─────────────┬───────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │  STEP 5: Signal Merge        │
           │                              │
           │  Inner join on (ticker, date)│
           │                              │
           │  Output: signal_df           │
           │  (all features unified)      │
           └──────────┬──────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────────┐
│ STEP 6:      │ │ STEP 7:  │ │ STEP 8: Backtest      │
│ Statistical  │ │ Time     │ │                        │
│ Analysis     │ │ Series   │ │ Long/Short strategy:   │
│              │ │          │ │  S > +0.10 → BUY       │
│ • Pearson r  │ │ • ADF    │ │  S < −0.10 → SELL      │
│ • Spearman r │ │   test   │ │  else → FLAT           │
│ • Kendall τ  │ │          │ │                        │
│ • OLS simple │ │ • ARIMA  │ │ Metrics:               │
│ • OLS w/ctrl │ │  (1,0,1) │ │  Sharpe ratio          │
│ • Quintile   │ │  5-day   │ │  Max drawdown          │
│   analysis   │ │  fcast   │ │  Win rate              │
│              │ │          │ │  Profit factor         │
│              │ │ • Granger│ │  Trade count           │
│              │ │  causal- │ │                        │
│              │ │  ity     │ └──────────────────────┬─┘
└──────────────┘ └──────────┘                        │
                                                     │
                         ┌───────────────────────────┘
                         │
                         ▼
            ┌───────────────────────────┐
            │  STEP 9: Extended Analysis │
            │                            │
            │  • Rolling 15-day corr.    │
            │  • Cross-ticker heatmap    │
            │  • Dispersion vs vol.      │
            │  • Bull ratio time series  │
            └───────────────────────────┘
```

---

## 3. Pipeline Explained Step by Step

### Step 1 — Data Collection

**Live mode** (requires Reddit API credentials):
```python
# PRAW: Python Reddit API Wrapper
reddit = praw.Reddit(client_id=..., client_secret=..., user_agent=...)
posts = reddit.subreddit("wallstreetbets").search("NVDA", limit=30, sort="new")
```
Each post yields: `title`, `selftext`, `score` (upvotes), `num_comments`, `upvote_ratio`, `created_utc`.

**Simulation mode** (default — no credentials needed):
- 50 posts × 5 tickers = 250 synthetic posts
- Titles drawn from a curated list of realistic Reddit language per ticker
- Timestamps uniformly distributed over last 60 days
- Scores log-normally distributed to mimic Reddit's power-law engagement

### Step 2 — FinBERT Sentiment Scoring

See [Section 4](#4-finbert-model-deep-dive) for full explanation.

### Step 3 — Daily Signal Aggregation

Raw posts are noisy. We aggregate to a daily granularity with engagement-weighted averaging:

```
weight_i    = √(upvotes_i + 1)         # viral posts count more
S_daily     = Σ(weight_i × compound_i) / Σ(weight_i)
```

This design choice prevents low-karma spam posts from diluting the signal — a single post with 5,000 upvotes receives ~71× more weight than a post with 0 upvotes.

**Additional engineered features:**

| Feature | Meaning | Use |
|---|---|---|
| `post_count` | Number of posts per day | Attention / crowd interest proxy |
| `avg_upvote_ratio` | Community agreement level | Consensus strength filter |
| `sentiment_std` | Intra-day sentiment spread | Uncertainty / disagreement signal |
| `bull_ratio` | % of posts classified positive | Narrative direction |
| `sentiment_momentum` | Day-over-day Δ sentiment | Regime change detection |
| `volume_zscore` | Standardised post count | Abnormal crowd activity flag |

### Step 4 — Market Data

yfinance downloads OHLCV data and we compute forward-looking return windows:

```python
return_1d  = close[t+1] / close[t] − 1   # predict this (main target)
return_2d  = close[t+2] / close[t] − 1
return_5d  = close[t+5] / close[t] − 1
log_return = ln(close[t] / close[t−1])
volatility = 5-day rolling σ(log_return) # realized vol proxy
volume_rel = volume[t] / MA20(volume)     # abnormal trading volume
```

### Step 5 — Merge

Inner join on `(ticker, date)`. Days where sentiment data or price data is missing are dropped. This is conservative but avoids lookahead bias.

---

## 4. FinBERT Model Deep Dive

### What is FinBERT?

FinBERT (Araci, 2019) is a **BERT-base** model fine-tuned on ~10,000 financial sentences from Reuters news, financial analyst reports, and earnings call transcripts. It outperforms generic BERT on financial sentiment classification.

### Architecture

```
Input Text: "NVDA smashes earnings again"
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ WordPiece Tokenizer                                             │
│                                                                  │
│  [CLS] NV ##DA smashes earnings again [SEP]                     │
│    ↓     ↓    ↓      ↓        ↓      ↓                          │
│  101  1319 3204  25362     6732  2153  102   (token IDs)        │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────┐
│ 12-Layer Transformer Encoder (BERT-base, 110M parameters)       │
│                                                                  │
│  Layer 1:  Multi-Head Self-Attention (12 heads, d_model=768)    │
│  Layer 2:  Add & Norm → Feed-Forward (3072 hidden) → Add & Norm │
│  ...                                                             │
│  Layer 12: Final contextual token representations               │
└────────────────────┬───────────────────────────────────────────┘
                     │  [CLS] token representation  (768-dim vector)
                     ▼
┌────────────────────────────────────────────────────────────────┐
│ Fine-tuned Classification Head                                   │
│                                                                  │
│  Linear(768 → 3) + Softmax                                      │
│                                                                  │
│  Output: [P_positive, P_negative, P_neutral]                    │
│  e.g.:   [  0.82,       0.04,      0.14   ]                    │
└────────────────────────────────────────────────────────────────┘
                     │
                     ▼
  compound = P_positive − P_negative = 0.82 − 0.04 = +0.78
```

### Why compound score?

`compound = P_positive − P_negative` maps the output to **[−1, +1]**:
- `+1.0` = pure positive (e.g., "NVDA crushes earnings, stock rips")
- `−1.0` = pure negative (e.g., "TSLA delivery disaster, biggest miss ever")
- `~0.0` = neutral or balanced (e.g., "AMD announces new chip lineup")

This is more expressive than a 3-class label because it preserves the confidence gradient.

### Limitations

- Trained on professional financial text; Reddit slang ("moon", "🚀", "apes") may be misclassified
- Max input length = 512 tokens; very long posts are truncated to 128 (our setting)
- Sarcasm and irony are frequently misclassified by any BERT model
- No temporal context — each post scored in isolation

---

## 5. Signal Engineering

### Weighting Rationale

```
Why √(score + 1) and not just score?

Raw score distribution on Reddit is heavily right-skewed:
  50% of posts  → score  10–100
  49% of posts  → score  100–5000
   1% of posts  → score  5000–100,000+

Using raw scores would give viral posts 1000× the weight of normal posts,
which is too aggressive. Using log(score) is too conservative.
√(score + 1) gives a balanced middle ground:

  score = 0       → weight = 1.0
  score = 100     → weight = 10.0
  score = 1000    → weight = 31.6
  score = 10000   → weight = 100.1
```

### Sentiment Momentum

`sentiment_momentum = S_daily(t) − S_daily(t−1)`

This captures **narrative shifts** — e.g., sentiment going from −0.3 to +0.2 in one day (momentum = +0.5) is a potentially stronger signal than a steady positive sentiment of +0.3.

### Volume Z-score

```
volume_zscore = (post_count − μ_post_count) / σ_post_count
```

Days with `volume_zscore > 2.0` indicate unusual crowd attention — potentially more signal-rich periods.

---

## 6. Statistical Analysis Methods

### A. Correlation Tests

Three tests used in parallel to triangulate strength of relationship:

| Test | Assumption | What it measures |
|---|---|---|
| **Pearson r** | Linear relationship, normality | Linear correlation coefficient |
| **Spearman ρ** | Monotonic (not necessarily linear) | Rank correlation — more robust to outliers |
| **Kendall τ** | No distributional assumptions | Concordance — fraction of concordant pairs |

**Interpretation rule:** If all three agree (all significant or all insignificant), the result is robust.

### B. OLS Regression

**Simple:** `return_1d ~ sentiment_score`

**Extended (with controls):**
```
return_1d ~ β₀ + β₁·sentiment_score 
                + β₂·volume_zscore
                + β₃·volatility
                + β₄·sentiment_momentum 
                + ε
```

The extended model tests whether sentiment has incremental predictive value *beyond* what volume and volatility already explain — a stronger test of signal quality.

### C. Quintile Analysis

Sort all (ticker, day) observations by `sentiment_score` → divide into 5 equal buckets → compute mean next-day return per bucket.

```
Ideal pattern (if signal works):
Q1 (most bearish) ──→ lowest mean return   (short candidate)
Q2               ──→ below-average return
Q3 (neutral)     ──→ near-zero return
Q4               ──→ above-average return
Q5 (most bullish) ──→ highest mean return  (long candidate)

Monotonic increase Q1→Q5 = strong signal
```

---

## 7. Backtesting Engine

### Strategy Rules

```python
position = +1  if sentiment_score >  +0.10   # LONG
position = -1  if sentiment_score <  -0.10   # SHORT
position =  0  otherwise                      # FLAT (no trade)
```

The ±0.10 threshold filters out weak/noisy signals, reducing unnecessary trading.

### Performance Metrics Explained

| Metric | Formula | What a good result looks like |
|---|---|---|
| **Total Return** | (final equity / initial equity) − 1 | Strategy > Buy & Hold |
| **Sharpe Ratio** | (μ_daily / σ_daily) × √252 | > 1.0 is considered good |
| **Max Drawdown** | min[(equity − peak equity) / peak equity] | Closer to 0% is better |
| **Win Rate** | Winning trades / Total trades | > 50% ideally |
| **Avg Win** | Mean return on winning days | Should exceed avg loss in magnitude |
| **Avg Loss** | Mean return on losing days | Should be smaller than avg win |
| **Profit Factor** | Avg Win / \|Avg Loss\| | > 1.0 means profitable in aggregate |
| **Trade Count** | Days with non-zero position | Higher = more signal utilised |

### Sharpe Ratio Intuition

```
Sharpe = annualised return / annualised volatility

Sharpe < 0    → worse than risk-free (terrible)
0 < Sharpe < 1 → earns return but with poor risk/reward
1 < Sharpe < 2 → solid risk-adjusted performance
Sharpe > 2    → exceptional (rare in live trading)
```

---

## 8. Extended Analysis (New)

Four additional analyses that draw deeper conclusions from the data:

### A. Rolling 15-Day Correlation

Instead of a single static correlation, we compute a rolling window to see:
- Does the sentiment-return relationship strengthen before earnings?
- Is correlation regime-dependent (high volatility vs low volatility markets)?
- Which ticker shows the most stable relationship over time?

### B. Cross-Ticker Sentiment Correlation Heatmap

Correlation matrix of daily sentiment scores across all 5 tickers.

**Interpretation:**
- High correlation between NVDA and AMD → AI chip narrative affects both simultaneously
- Low correlation between GME and AAPL → driven by different communities and catalysts
- This is useful for portfolio construction: low inter-ticker sentiment correlation = better diversification

### C. Sentiment Dispersion vs Realised Volatility

`sentiment_std` = standard deviation of compound scores within a single day.

**Hypothesis:** High intra-day sentiment dispersion (community disagreement) should predict higher subsequent realised price volatility.

This would be useful for **options traders** who trade volatility (vega), not direction.

### D. Bull Ratio Over Time

`bull_ratio` = fraction of posts classified as positive on a given day.

Plotting this over time reveals:
- Persistent bullish periods (trend confirmation)
- Sharp drops in bull ratio (narrative reversal signal)
- Divergence from price: price rising while bull ratio falling = potential topping signal

### E. Granger Causality

Tests whether past values of sentiment help predict future returns *beyond what past returns alone explain*.

```
H₀: Sentiment does NOT Granger-cause returns
H₁: Sentiment adds predictive power (p < 0.05 → reject H₀)
```

Note: Granger causality ≠ true causality. It measures **predictive precedence**, not mechanism.

---

## 9. Ethical Implications

This project was submitted as part of an ethics analysis. Key issues identified:

### Privacy and Consent

Reddit users do not consent to their posts being used in financial modelling. While technically public, this represents a **contextual integrity violation** — posts written for community discussion are repurposed for profit extraction.

### Information Asymmetry

Alternative data is expensive. Only large hedge funds can afford:
- Commercial Reddit data providers (prices: $50k–$500k/year)
- GPU infrastructure to run models like FinBERT at scale
- Quant teams to interpret signals

This creates **structural information inequality** in markets, disadvantaging retail investors.

### Bias

- Reddit's user base skews young, male, US-based, and tech-oriented
- GME/meme stock communities overrepresent high-volatility, short-term speculation
- NVDA/AMD discussions reflect tech-industry insiders
- This **demographic bias** propagates into the model's signals

### Market Manipulation Risk

If sentiment signals become widely known and traded:
- Bad actors can post coordinated bullish content to drive prices up
- "Pump and dump" via social media becomes algorithmically amplifiable
- This creates **feedback loops** between AI systems and social platforms

### Responsible AI Recommendations

1. **Transparency:** Disclose when social media data is used in trading systems
2. **Bias audits:** Regularly test whether model outputs systematically disadvantage certain stocks/sectors
3. **Regulatory compliance:** Comply with SEC guidelines on alternative data usage (Rule 10b-5)
4. **Data minimisation:** Avoid processing personally identifiable information
5. **Consent frameworks:** Support initiatives for user data ownership and compensation

---

## 10. How to Run

### Prerequisites

```bash
git clone https://github.com/YOUR_USERNAME/alt-data-quant-trading.git
cd alt-data-quant-trading
pip install -r requirements.txt
```

### Quick Start (Simulated Data, No Credentials Needed)

```bash
python src/pipeline.py
```

This runs the full pipeline with:
- Synthetic Reddit data (250 posts)
- Keyword-based mock sentiment (no FinBERT download)
- Real price data from yfinance
- All 4 figures saved to `outputs/`

### With Real FinBERT (GPU recommended)

```python
from src.pipeline import main

main(
    use_live_reddit=False,   # keep simulated data
    use_finbert=True          # enable real FinBERT NLP
)
```

FinBERT downloads ~420 MB on first run and is cached to `~/.cache/huggingface/`.

### With Live Reddit Data

1. Create a Reddit API app at https://www.reddit.com/prefs/apps
2. Note your `client_id` and `client_secret`

```python
from src.pipeline import main

main(
    use_live_reddit=True,
    use_finbert=True,
    reddit_client_id="YOUR_CLIENT_ID",
    reddit_client_secret="YOUR_SECRET"
)
```

### Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GJ9SBGDYe5DM67rG7qWM2KFVqYNvySO7)

The original notebook is hosted on Colab. The Colab environment provides free GPU, which significantly speeds up FinBERT inference.

---

## 11. Output Files

After running, `outputs/` will contain:

| File | Description |
|---|---|
| `figure1_main_dashboard.png` | 4-panel: sentiment timeline, quintile returns, scatter, post volume |
| `figure2_time_series.png` | 3-panel: raw signal + ACF + ARIMA forecast for NVDA |
| `figure3_backtest.png` | 2-row: equity curves + Sharpe/drawdown/win-rate bar charts |
| `figure4_extended.png` | 4-panel: rolling corr, heatmap, dispersion vs vol, bull ratio |

---

## 12. Project Structure

```
alt-data-quant-trading/
│
├── README.md                         ← You are here
├── requirements.txt                  ← Python dependencies
├── .gitignore
│
├── src/
│   └── pipeline.py                   ← Full end-to-end pipeline (all steps)
│
├── docs/
│   ├── methodology.md                ← Academic methodology writeup
│   └── ethical_analysis.md           ← Full ethics report
│
├── outputs/                          ← Generated figures (git-ignored)
│   └── .gitkeep
│
└── .github/
    └── CITATION.md                   ← How to cite this work
```

---

## References

1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *arXiv:1810.04805*
3. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
4. Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance, 62*(3), 1139–1168.
5. Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science, 2*(1), 1–8.
6. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance, 66*(1), 35–65.

---

*For academic correspondence or questions about the research methodology, please open a GitHub Issue.*
