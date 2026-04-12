# Methodology

## Research Design

This study follows a **quantitative, empirical** research design. The methodology is structured to test whether Reddit-derived NLP sentiment signals contain information about near-term stock returns.

## Data

### Reddit Data
- **Source:** r/wallstreetbets, r/investing, r/stocks (three subreddits)
- **Tickers:** GME, NVDA, TSLA, AAPL, AMD
- **Time window:** 60-day rolling window
- **Fields collected:** post title, body, score (upvotes), comment count, upvote ratio, UTC timestamp

**Note:** For this study, synthetic data was generated to simulate realistic Reddit posts due to API rate-limit constraints during development. The pipeline fully supports live Reddit scraping via PRAW when credentials are supplied.

### Price Data
- **Source:** Yahoo Finance (via yfinance)
- **Granularity:** Daily OHLCV
- **Time window:** 65 days (5 extra days for return computation lookback)

## NLP Model

**FinBERT** (ProsusAI, 2019) — BERT-base fine-tuned on 10,000+ financial sentences from Reuters, financial blogs, and analyst reports.

- **Input:** Post title (max 128 tokens, truncated)
- **Output:** P(positive), P(negative), P(neutral)
- **Compound score:** compound = P(positive) − P(negative) ∈ [−1, +1]

## Signal Aggregation

Daily sentiment aggregated using engagement-weighted mean:

```
S_t = Σ_i [√(score_i + 1) × compound_i] / Σ_i [√(score_i + 1)]
```

Additional features: post_count, sentiment_std, bull_ratio, sentiment_momentum, volume_zscore.

## Statistical Tests

### Correlation Analysis
Three non-parametric tests used to triangulate signal validity:
- Pearson r (linear, assumes normality)
- Spearman ρ (monotonic, rank-based)
- Kendall τ (concordance-based, most robust)

Significance threshold: α = 0.05 (two-tailed).

### Regression
- Simple OLS: `return_1d ~ sentiment_score`
- Extended OLS: `return_1d ~ sentiment + volume_zscore + volatility + sentiment_momentum`

### Quintile Analysis
Observations ranked by daily sentiment → divided into 5 equal groups → mean return per quintile computed.

### Time Series
- ADF test for stationarity (KPSS used as robustness check)
- ARIMA(1,0,1) for 5-day sentiment forecasting
- Granger causality test (lag = 1, 2, 3 days)
- Rolling 15-day Pearson correlation

## Backtesting

**Strategy:** Long/short signal-driven strategy with flat periods.
- Long threshold: compound > +0.10
- Short threshold: compound < −0.10
- Benchmark: Buy and Hold

**Evaluation metrics:**
- Total return
- Annualised Sharpe ratio (rf = 0)
- Maximum drawdown
- Win rate
- Profit factor
- Trade count

**Important caveat:** This is a **simulation only**. Transaction costs, slippage, market impact, and short-selling constraints are not modelled. Real-world implementation would degrade these results.

## Limitations

1. **Sample size:** 250 synthetic posts over 60 days is small for drawing definitive conclusions. A production system would use millions of posts.
2. **Synthetic data:** Mock sentiment does not capture true FinBERT classification on real Reddit text.
3. **Survivorship bias:** The 5 tickers chosen are all well-known, actively traded stocks. The signal may not generalise to smaller caps.
4. **Look-ahead bias check:** Return features use `.shift(-n)` to ensure no future data leaks into signal computation.
5. **Transaction costs:** Not modelled; Sharpe ratios would decrease in practice.
6. **Reddit data quality:** Bots, coordinated campaigns, and low-quality posts add noise.
