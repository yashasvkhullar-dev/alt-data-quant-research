# -*- coding: utf-8 -*-
"""
pipeline.py — End-to-end Alternative Data Trading Research Pipeline
=======================================================================
Author  : Yashasv Khullar et
Project : Alternative Data in Quantitative Trading (CA3 — AISEE)
Model   : FinBERT Sentiment → Stock Return Prediction

Pipeline steps:
  1. Data Collection   — Reddit scraping (live or simulated)
  2. Sentiment Scoring — FinBERT (ProsusAI/finbert)
  3. Signal Building   — Weighted daily sentiment aggregation
  4. Market Data       — yfinance OHLCV
  5. Statistical Tests — Pearson / Spearman / Kendall / OLS / Quintile
  6. Time Series       — ADF stationarity + ARIMA(1,0,1) forecast
  7. Backtesting       — Equity curve, Sharpe, Drawdown, Win Rate
  8. Extended Analysis — Rolling correlation, Granger causality,
                         Cross-ticker heatmap, Market regime detection
  9. Visualizations    — 4 multi-panel figures saved as PNG
"""

# ── 0. Dependencies ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau

import yfinance as yf

# ── Plotting theme (GitHub dark palette) ─────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
})

PALETTE = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
TICKERS    = ["GME", "NVDA", "TSLA", "AAPL", "AMD"]
SUBREDDITS = ["wallstreetbets", "investing", "stocks"]
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def scrape_reddit_live(ticker: str, subreddit_name: str,
                       client_id: str, client_secret: str,
                       limit: int = 30) -> pd.DataFrame:
    """
    Live Reddit scraping via PRAW.
    Set USE_LIVE_REDDIT = True and supply real credentials to activate.

    Parameters
    ----------
    ticker         : Stock ticker symbol (e.g. "NVDA")
    subreddit_name : Target subreddit (e.g. "wallstreetbets")
    client_id      : Reddit API client ID
    client_secret  : Reddit API secret
    limit          : Max posts to retrieve per query

    Returns
    -------
    pd.DataFrame with columns: ticker, subreddit, title, selftext,
                                score, num_comments, upvote_ratio, created_utc
    """
    import praw
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="altdata_research_bot",
    )
    records = []
    for post in reddit.subreddit(subreddit_name).search(ticker, limit=limit, sort="new"):
        records.append({
            "ticker":       ticker,
            "subreddit":    subreddit_name,
            "title":        post.title,
            "selftext":     post.selftext[:300],
            "score":        post.score,
            "num_comments": post.num_comments,
            "upvote_ratio": post.upvote_ratio,
            "created_utc":  datetime.utcfromtimestamp(post.created_utc),
        })
    return pd.DataFrame(records)


def generate_sample_data(tickers=TICKERS, n_posts_per_ticker=50,
                         days_back=60) -> pd.DataFrame:
    """
    Generate realistic synthetic Reddit post data.
    Used when live Reddit credentials are unavailable.

    Each ticker has a mix of bullish / bearish / neutral titles.
    Post volume, scores, and timestamps are randomized to mimic real data.

    Parameters
    ----------
    tickers               : List of ticker symbols
    n_posts_per_ticker    : Number of posts to simulate per ticker
    days_back             : How many days of history to simulate

    Returns
    -------
    pd.DataFrame sorted by created_utc
    """
    sample_titles = {
        "GME":  ["GME short squeeze incoming again",
                 "GameStop fundamentals don't justify this price",
                 "GME to the moon 🚀🚀", "Why I'm holding GME long term",
                 "GME earnings disaster", "Ryan Cohen moves the needle",
                 "Retail vs institutions — GME round 2?"],
        "NVDA": ["NVDA is the most important AI company",
                 "Nvidia valuation is insane right now",
                 "Bought NVDA calls, feeling good",
                 "NVDA smashes earnings again", "GPU shortage lifts NVDA",
                 "Blackwell chip demand exploding", "NVDA data center beats"],
        "TSLA": ["Tesla delivery numbers disappointing",
                 "Elon selling TSLA again???", "TSLA bull case 2024",
                 "Tesla FSD actually works now", "Why TSLA is overvalued",
                 "Cybertruck demand softer than expected",
                 "Tesla energy division underrated"],
        "AAPL": ["Apple Vision Pro flop?", "AAPL quietly dominates services",
                 "iPhone 16 demand looks weak", "Apple moat unbreakable",
                 "AAPL buyback machine continues", "India manufacturing = bull",
                 "Apple AI features disappoint"],
        "AMD":  ["AMD gaining on Intel fast", "Ryzen 9000 benchmarks insane",
                 "AMD vs NVDA in AI chips — AMD underdog story",
                 "Bought AMD before earnings", "AMD undervalued vs Nvidia",
                 "MI300X getting enterprise traction", "AMD server share rising"],
    }

    base_date = datetime.now() - timedelta(days=days_back)
    records   = []

    for ticker in tickers:
        for _ in range(n_posts_per_ticker):
            post_date = base_date + timedelta(
                days=random.randint(0, days_back - 2),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            records.append({
                "ticker":       ticker,
                "subreddit":    random.choice(SUBREDDITS),
                "title":        random.choice(sample_titles[ticker]),
                "selftext":     "",
                "score":        random.randint(10, 5000),
                "num_comments": random.randint(5, 500),
                "upvote_ratio": round(random.uniform(0.55, 0.98), 2),
                "created_utc":  post_date,
            })

    df = pd.DataFrame(records).sort_values("created_utc").reset_index(drop=True)
    print(f"[DATA] Simulated {len(df)} posts across {df['ticker'].nunique()} tickers "
          f"spanning {days_back} days.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SENTIMENT SCORING (FinBERT)
# ══════════════════════════════════════════════════════════════════════════════

def load_finbert():
    """
    Load FinBERT tokenizer and model from HuggingFace.
    FinBERT = BERT fine-tuned on ~10 000 financial sentences (Reuters, etc.)

    Returns
    -------
    (tokenizer, model) tuple ready for inference
    """
    from transformers import BertTokenizer, BertForSequenceClassification
    FINBERT_MODEL = "ProsusAI/finbert"
    print(f"[MODEL] Loading {FINBERT_MODEL} (~420 MB one-time download)...")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
    model     = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    print("[MODEL] FinBERT ready.")
    return tokenizer, model


def score_sentiment(text: str, tokenizer, model) -> dict:
    """
    Run FinBERT on a single text string.

    Label mapping:
      0 → positive   1 → negative   2 → neutral

    Parameters
    ----------
    text      : Raw post title or body text
    tokenizer : FinBERT BertTokenizer
    model     : FinBERT BertForSequenceClassification

    Returns
    -------
    dict with keys: label (str), positive (float), negative (float), neutral (float)
    """
    import torch
    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    text = str(text).strip()
    if not text:
        return {"label": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()

    return {
        "label":    LABEL_MAP[int(torch.argmax(logits))],
        "positive": round(probs[0], 4),
        "negative": round(probs[1], 4),
        "neutral":  round(probs[2], 4),
    }


def apply_finbert(raw_df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """Score every post title and attach sentiment columns."""
    print(f"[SENTIMENT] Scoring {len(raw_df)} posts with FinBERT...")
    results = raw_df["title"].apply(lambda t: score_sentiment(t, tokenizer, model))
    sentiment_df = pd.DataFrame(results.tolist())
    scored = pd.concat([raw_df.reset_index(drop=True), sentiment_df], axis=1)
    scored["compound"] = scored["positive"] - scored["negative"]   # range: −1 to +1
    print(f"[SENTIMENT] Done. Label distribution:\n{scored['label'].value_counts()}\n")
    return scored


def mock_sentiment(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based mock sentiment for fast testing without GPU / large download.
    Assigns sentiment based on keyword heuristics in the post title.
    Useful for quick iteration during development.
    """
    BULLISH_WORDS = {"moon", "bull", "calls", "buy", "beats", "dominates",
                     "smashes", "gaining", "insane", "important", "traction",
                     "rising", "unbreakable", "exploding", "underdog"}
    BEARISH_WORDS = {"flop", "disaster", "disappointing", "overvalued", "weak",
                     "selling", "softer", "down", "drop", "fail", "disappoint",
                     "loss", "crash"}

    def _label(title: str) -> dict:
        words = set(title.lower().split())
        pos_hits = len(words & BULLISH_WORDS)
        neg_hits = len(words & BEARISH_WORDS)
        if pos_hits > neg_hits:
            p, n, u = 0.70, 0.10, 0.20
            label = "positive"
        elif neg_hits > pos_hits:
            p, n, u = 0.10, 0.70, 0.20
            label = "negative"
        else:
            p, n, u = 0.20, 0.20, 0.60
            label = "neutral"
        # Add small noise
        noise = np.random.uniform(-0.08, 0.08, 3)
        p, n, u = max(0, p + noise[0]), max(0, n + noise[1]), max(0, u + noise[2])
        total = p + n + u
        return {"label": label, "positive": round(p/total, 4),
                "negative": round(n/total, 4), "neutral": round(u/total, 4)}

    results  = raw_df["title"].apply(_label)
    sent_df  = pd.DataFrame(results.tolist())
    scored   = pd.concat([raw_df.reset_index(drop=True), sent_df], axis=1)
    scored["compound"] = scored["positive"] - scored["negative"]
    print("[SENTIMENT] Mock sentiment applied (keyword heuristic).")
    print(scored["label"].value_counts(), "\n")
    return scored


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DAILY SIGNAL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def build_daily_signal(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-post sentiment into a single daily signal per ticker.

    Weighting scheme:
        weight_i = sqrt(score_i + 1)      ← viral posts carry more signal
        daily_sentiment = Σ(weight_i × compound_i) / Σ(weight_i)

    Additional features computed per (ticker, date):
        - post_count       : raw post volume (attention proxy)
        - avg_upvote_ratio : community consensus strength
        - total_comments   : engagement depth
        - sentiment_momentum: day-over-day change in sentiment (new)
        - volume_zscore    : standardised post volume (new)

    Returns
    -------
    pd.DataFrame indexed by (ticker, date) with signal columns
    """
    scored_df = scored_df.copy()
    scored_df["date"] = pd.to_datetime(scored_df["created_utc"]).dt.normalize()

    def _agg(group):
        weights = np.sqrt(group["score"].clip(lower=0) + 1)
        weighted_compound = np.average(group["compound"], weights=weights)
        return pd.Series({
            "sentiment_score":   round(weighted_compound, 4),
            "post_count":        len(group),
            "avg_upvote_ratio":  round(group["upvote_ratio"].mean(), 4),
            "total_comments":    group["num_comments"].sum(),
            "sentiment_std":     round(group["compound"].std(), 4),   # intra-day dispersion
            "bull_ratio":        round((group["label"] == "positive").mean(), 4),
        })

    daily = (
        scored_df
        .groupby(["ticker", "date"])
        .apply(_agg)
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # Sentiment momentum (1-day lag difference) — captures shifts in narrative
    daily = daily.sort_values(["ticker", "date"])
    daily["sentiment_momentum"] = (
        daily.groupby("ticker")["sentiment_score"].diff()
    )

    # Volume z-score (per ticker) — how unusual is today's activity?
    daily["volume_zscore"] = (
        daily.groupby("ticker")["post_count"]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    )

    print(f"[SIGNAL] Daily sentiment constructed: {len(daily)} (ticker, date) pairs.")
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_price_data(tickers=TICKERS, days_back=65) -> dict:
    """
    Download OHLCV data via yfinance and compute return features.

    Features added per ticker:
        return_1d   : next-day forward return (t+1 close / t close − 1)
        return_2d   : 2-day forward return
        return_5d   : 5-day forward return (new)
        log_return  : log(close_t / close_{t-1})
        volatility  : 5-day rolling std of log returns (realised vol proxy)
        volume_rel  : volume / 20-day avg volume (abnormal volume signal)

    Returns
    -------
    dict {ticker: pd.DataFrame}
    """
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end   = datetime.now().strftime("%Y-%m-%d")
    print(f"[PRICE] Downloading OHLCV {start} → {end} ...")
    price_data = {}

    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df.index = pd.to_datetime(df.index)

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df["return_1d"]    = df["Close"].pct_change(1).shift(-1)
        df["return_2d"]    = df["Close"].pct_change(2).shift(-2)
        df["return_5d"]    = df["Close"].pct_change(5).shift(-5)
        df["log_return"]   = np.log(df["Close"] / df["Close"].shift(1))
        df["volatility"]   = df["log_return"].rolling(5).std()
        df["volume_rel"]   = df["Volume"] / df["Volume"].rolling(20).mean()

        price_data[ticker] = df
        print(f"  {ticker}: {len(df)} trading days")

    print()
    return price_data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SIGNAL MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_signals(daily_sentiment: pd.DataFrame,
                  price_data: dict,
                  tickers=TICKERS) -> pd.DataFrame:
    """
    Inner-join daily sentiment with price data on (ticker, date).

    Returns
    -------
    pd.DataFrame with all sentiment + price features per (ticker, date)
    """
    parts = []
    for ticker in tickers:
        sent = daily_sentiment[daily_sentiment["ticker"] == ticker].copy()
        price = (price_data[ticker]
                 .reset_index()
                 .rename(columns={"Date": "date", "Close": "close",
                                  "Volume": "volume"}))
        price["date"] = pd.to_datetime(price["date"])
        merged = pd.merge(sent, price[["date", "close", "volume",
                                       "return_1d", "return_2d", "return_5d",
                                       "log_return", "volatility", "volume_rel"]],
                          on="date", how="inner")
        merged["ticker"] = ticker
        parts.append(merged)

    signal_df = (pd.concat(parts, ignore_index=True)
                 .dropna(subset=["return_1d", "sentiment_score"]))
    print(f"[MERGE] Combined dataset: {len(signal_df)} rows across "
          f"{signal_df['ticker'].nunique()} tickers.\n")
    return signal_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_analysis(signal_df: pd.DataFrame) -> dict:
    """
    Run comprehensive statistical tests to evaluate whether sentiment
    has predictive power over next-day stock returns.

    Tests performed:
        A. Pearson / Spearman / Kendall correlation (per ticker)
        B. OLS regression: return ~ sentiment
        C. Quintile analysis: mean return per sentiment bucket
        D. Extended OLS with controls: return ~ sentiment + volume_zscore + volatility

    Returns
    -------
    dict with keys: correlations, ols_simple, quintile, ols_extended
    """
    from scipy.stats import linregress

    print("=" * 65)
    print("  STATISTICAL ANALYSIS: SENTIMENT → NEXT-DAY RETURN")
    print("=" * 65)

    # ── A. Three-way correlation per ticker ──────────────────────────────────
    print("\nA. Correlation Tests (per ticker):\n")
    corr_records = []
    for ticker in TICKERS:
        sub = signal_df[signal_df["ticker"] == ticker].dropna(
              subset=["sentiment_score", "return_1d"])
        if len(sub) < 5:
            continue
        s, r = sub["sentiment_score"].values, sub["return_1d"].values
        p_r, p_p   = pearsonr(s, r)
        sp_r, sp_p = spearmanr(s, r)
        k_r, k_p   = kendalltau(s, r)
        print(f"  ${ticker:5s}  Pearson r={p_r:+.3f} p={p_p:.3f}  "
              f"Spearman r={sp_r:+.3f} p={sp_p:.3f}  "
              f"Kendall τ={k_r:+.3f} p={k_p:.3f}")
        corr_records.append(dict(ticker=ticker, pearson_r=p_r, pearson_p=p_p,
                                  spearman_r=sp_r, spearman_p=sp_p,
                                  kendall_tau=k_r, kendall_p=k_p))
    correlations = pd.DataFrame(corr_records)

    # ── B. Simple OLS: return ~ sentiment ────────────────────────────────────
    slope, intercept, r_val, p_val, std_err = linregress(
        signal_df["sentiment_score"], signal_df["return_1d"])
    ols_simple = dict(slope=slope, intercept=intercept,
                      r_squared=r_val**2, p_value=p_val, std_err=std_err)
    print(f"\nB. OLS (pooled):  slope={slope:.5f}  R²={r_val**2:.4f}  "
          f"p={p_val:.4f}")

    # ── C. Quintile analysis ─────────────────────────────────────────────────
    signal_df = signal_df.copy()
    signal_df["quintile"] = pd.qcut(
        signal_df["sentiment_score"], q=5,
        labels=["Q1\n(most neg)", "Q2", "Q3", "Q4", "Q5\n(most pos)"])
    quintile = (signal_df.groupby("quintile", observed=True)["return_1d"]
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "Mean Return",
                                 "std": "Std Dev", "count": "Obs"}))
    quintile["Mean Return"] = (quintile["Mean Return"] * 100).round(3)
    quintile["Std Dev"]     = (quintile["Std Dev"] * 100).round(3)
    print(f"\nC. Quintile Returns:\n{quintile.to_string()}")

    # ── D. Extended OLS with controls ────────────────────────────────────────
    try:
        import statsmodels.api as sm
        X_cols = ["sentiment_score", "volume_zscore", "volatility",
                  "sentiment_momentum"]
        sub_ext = signal_df.dropna(subset=X_cols + ["return_1d"])
        X = sm.add_constant(sub_ext[X_cols].astype(float))
        y = sub_ext["return_1d"].astype(float)
        ols_ext_res = sm.OLS(y, X).fit()
        print(f"\nD. Extended OLS (with controls):\n{ols_ext_res.summary2().tables[1]}")
        ols_extended = ols_ext_res
    except Exception as e:
        print(f"\nD. Extended OLS skipped: {e}")
        ols_extended = None

    return dict(correlations=correlations, ols_simple=ols_simple,
                quintile=quintile, ols_extended=ols_extended,
                signal_df_with_quintile=signal_df)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — TIME SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_time_series_analysis(signal_df: pd.DataFrame,
                             focus_ticker: str = "NVDA") -> dict:
    """
    Time-series analysis of the sentiment signal for a chosen ticker.

    Tests:
        1. ADF stationarity test (auto-differences if needed)
        2. ARIMA(1,0,1) 5-day forecast
        3. Granger causality: does sentiment Granger-cause returns? (new)
        4. Rolling 30-day correlation between sentiment and return (new)

    Parameters
    ----------
    signal_df    : merged signal dataframe
    focus_ticker : ticker to analyse in detail (default "NVDA")

    Returns
    -------
    dict with keys: ts, adf_results, arima_model, forecast, granger, rolling_corr
    """
    from statsmodels.tsa.stattools  import adfuller, grangercausalitytests
    from statsmodels.tsa.arima.model import ARIMA

    print("=" * 65)
    print(f"  TIME SERIES ANALYSIS — ${focus_ticker} Sentiment Signal")
    print("=" * 65)

    # Prepare daily sentiment series (resample to calendar days, ffill gaps)
    ts = (signal_df[signal_df["ticker"] == focus_ticker]
          .sort_values("date")
          .set_index("date")["sentiment_score"]
          .resample("D").mean()
          .ffill()
          .dropna())

    print(f"\n  Series length : {len(ts)} obs  "
          f"({ts.index[0].date()} → {ts.index[-1].date()})")

    # ADF test
    adf_stat, adf_p, _, _, crit_vals, _ = adfuller(ts, autolag="AIC")
    stationary = adf_p < 0.05
    print(f"\n  ADF: stat={adf_stat:.4f}  p={adf_p:.4f}  "
          f"{'STATIONARY ✓' if stationary else 'NON-STATIONARY — differencing'}")

    ts_model = ts if stationary else ts.diff().dropna()

    # ARIMA(1,0,1)
    arima = ARIMA(ts_model, order=(1, 0, 1)).fit()
    fc    = arima.get_forecast(steps=5)
    fcast = fc.predicted_mean
    conf  = fc.conf_int(alpha=0.05)
    print(f"\n  ARIMA(1,0,1) AIC={arima.aic:.2f}  BIC={arima.bic:.2f}")
    print("  5-day forecast:")
    for i, (idx, val) in enumerate(fcast.items()):
        lo, hi = conf.iloc[i, 0], conf.iloc[i, 1]
        print(f"    Day {i+1}: {val:+.4f}  CI=[{lo:.4f}, {hi:.4f}]")

    # Granger causality: does sentiment Granger-cause next-day return?
    granger_result = None
    try:
        gc_sub = (signal_df[signal_df["ticker"] == focus_ticker]
                  .sort_values("date")[["sentiment_score", "return_1d"]]
                  .dropna())
        gc_data = gc_sub[["return_1d", "sentiment_score"]].values
        if len(gc_data) >= 10:
            gc_res = grangercausalitytests(gc_data, maxlag=3, verbose=False)
            gc_p   = {lag: list(res[0].values())[0][1]
                      for lag, res in gc_res.items()}
            print(f"\n  Granger causality (sentiment → return) p-values by lag:")
            for lag, p in gc_p.items():
                sig = "✓ sig" if p < 0.05 else "— not sig"
                print(f"    Lag {lag}: p={p:.4f} {sig}")
            granger_result = gc_p
    except Exception as e:
        print(f"\n  Granger causality skipped: {e}")

    # Rolling 30-day correlation
    all_tickers_roll = []
    for ticker in TICKERS:
        sub = (signal_df[signal_df["ticker"] == ticker]
               .sort_values("date")
               .set_index("date")[["sentiment_score", "return_1d"]]
               .dropna())
        roll_corr = (sub["sentiment_score"]
                     .rolling(15)
                     .corr(sub["return_1d"])
                     .dropna())
        roll_corr.name = ticker
        all_tickers_roll.append(roll_corr)
    rolling_corr = pd.concat(all_tickers_roll, axis=1)

    return dict(ts=ts, ts_model=ts_model,
                adf=dict(stat=adf_stat, p=adf_p, stationary=stationary),
                arima_model=arima, forecast=fcast, conf_int=conf,
                granger=granger_result, rolling_corr=rolling_corr)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — BACKTESTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(signal_df: pd.DataFrame,
                 long_threshold:  float = 0.10,
                 short_threshold: float = -0.10,
                 tickers=TICKERS) -> pd.DataFrame:
    """
    Simulate a long/short sentiment-driven trading strategy.

    Rules:
        sentiment >  long_threshold   → go LONG  (position = +1)
        sentiment < short_threshold   → go SHORT (position = −1)
        otherwise                     → flat      (position =  0)

    Metrics computed per ticker:
        total_return     : cumulative strategy return
        sharpe_ratio     : annualised Sharpe (252 trading days, rf=0)
        max_drawdown     : maximum peak-to-trough loss
        win_rate         : fraction of trades that were profitable
        avg_win          : average return on winning trades
        avg_loss         : average return on losing trades
        profit_factor    : |avg_win| / |avg_loss|
        trade_count      : number of non-zero position days

    Parameters
    ----------
    long_threshold  : sentiment score above which we go long
    short_threshold : sentiment score below which we go short

    Returns
    -------
    pd.DataFrame indexed by ticker with all performance metrics
    """
    print("=" * 65)
    print("  BACKTEST: Sentiment Strategy vs Buy & Hold")
    print("=" * 65)

    rows = []
    for ticker in tickers:
        sub = (signal_df[signal_df["ticker"] == ticker]
               .copy()
               .sort_values("date"))

        sub["position"]  = np.where(sub["sentiment_score"] >  long_threshold,  1,
                           np.where(sub["sentiment_score"] < short_threshold, -1, 0))
        sub["strat_ret"] = sub["position"] * sub["return_1d"]

        cum_strat = (1 + sub["strat_ret"]).cumprod()
        cum_bh    = (1 + sub["return_1d"]).cumprod()

        total_ret  = cum_strat.iloc[-1] - 1
        bh_ret     = cum_bh.iloc[-1]    - 1

        # Sharpe ratio (annualised, assume rf=0)
        daily_mean  = sub["strat_ret"].mean()
        daily_std   = sub["strat_ret"].std()
        sharpe      = (daily_mean / (daily_std + 1e-9)) * np.sqrt(252)

        # Maximum drawdown
        roll_max  = cum_strat.cummax()
        drawdown  = (cum_strat - roll_max) / roll_max
        max_dd    = drawdown.min()

        # Win rate / avg win / avg loss
        active = sub[sub["position"] != 0]["strat_ret"]
        wins   = active[active > 0]
        losses = active[active < 0]
        win_rate    = len(wins) / max(len(active), 1)
        avg_win     = wins.mean()  if len(wins)   > 0 else 0.0
        avg_loss    = losses.mean() if len(losses) > 0 else 0.0
        profit_factor = abs(avg_win) / (abs(avg_loss) + 1e-9)

        rows.append(dict(
            ticker=ticker,
            total_return=round(total_ret * 100, 2),
            bh_return=round(bh_ret * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown=round(max_dd * 100, 2),
            win_rate=round(win_rate * 100, 1),
            avg_win_pct=round(avg_win * 100, 3),
            avg_loss_pct=round(avg_loss * 100, 3),
            profit_factor=round(profit_factor, 3),
            trade_count=int((sub["position"] != 0).sum()),
        ))
        print(f"  ${ticker:5s}  TotalRet={total_ret*100:+.2f}%  "
              f"BH={bh_ret*100:+.2f}%  Sharpe={sharpe:.2f}  "
              f"MaxDD={max_dd*100:.2f}%  WinRate={win_rate*100:.0f}%")

    metrics = pd.DataFrame(rows).set_index("ticker")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_main_dashboard(signal_df: pd.DataFrame,
                        stats_results: dict,
                        save_path: str = "outputs/figure1_main_dashboard.png"):
    """
    Figure 1 — Main 4-panel research dashboard.
        Panel A : Sentiment timeline (3-day rolling avg per ticker)
        Panel B : Quintile mean returns bar chart
        Panel C : Scatter sentiment vs next-day return + OLS line
        Panel D : Daily Reddit post volume (stacked bar)
    """
    ols   = stats_results["ols_simple"]
    qdf   = stats_results["quintile"]
    sdf   = stats_results["signal_df_with_quintile"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Alternative Data Research: Reddit Sentiment → Stock Returns",
                 fontsize=16, fontweight="bold", color="#58a6ff", y=1.01)

    # A — Sentiment timeline
    ax1 = axes[0, 0]
    for i, ticker in enumerate(TICKERS):
        sub = signal_df[signal_df["ticker"] == ticker].sort_values("date")
        ax1.plot(sub["date"], sub["sentiment_score"].rolling(3).mean(),
                 label=ticker, color=PALETTE[i], linewidth=1.8, alpha=0.9)
    ax1.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title("A — Daily Sentiment (3-day rolling avg)", color="#58a6ff", fontweight="bold")
    ax1.set_ylabel("Sentiment  (−1 → +1)")
    ax1.legend(fontsize=8, framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # B — Quintile returns
    ax2 = axes[0, 1]
    ql = ["Q1\n(bearish)", "Q2", "Q3", "Q4", "Q5\n(bullish)"]
    bar_colors = ["#f78166", "#ffa657", "#8b949e", "#56d364", "#3fb950"]
    vals = qdf["Mean Return"].values
    bars = ax2.bar(ql, vals, color=bar_colors, width=0.6, edgecolor="#21262d")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + np.sign(val) * 0.001,
                 f"{val:+.3f}%", ha="center", va="bottom", fontsize=9, color="#e6edf3")
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_title("B — Avg Next-Day Return by Sentiment Quintile", color="#58a6ff", fontweight="bold")
    ax2.set_ylabel("Mean Return (%)")
    ax2.grid(True, alpha=0.3, axis="y")

    # C — Scatter + OLS
    ax3 = axes[1, 0]
    for i, ticker in enumerate(TICKERS):
        sub = signal_df[signal_df["ticker"] == ticker]
        ax3.scatter(sub["sentiment_score"], sub["return_1d"] * 100,
                    label=ticker, alpha=0.6, s=30, color=PALETTE[i])
    x_line = np.linspace(signal_df["sentiment_score"].min(),
                         signal_df["sentiment_score"].max(), 100)
    y_line = ols["slope"] * x_line + ols["intercept"]
    ax3.plot(x_line, y_line * 100, color="#f0f6fc", linewidth=1.5,
             linestyle="--", label=f"OLS (R²={ols['r_squared']:.3f})")
    ax3.set_title(f"C — Sentiment vs Next-Day Return (R²={ols['r_squared']:.3f})",
                  color="#58a6ff", fontweight="bold")
    ax3.set_xlabel("Daily Sentiment Score")
    ax3.set_ylabel("Next-Day Return (%)")
    ax3.legend(fontsize=7, framealpha=0.3)
    ax3.grid(True, alpha=0.3)

    # D — Post volume stacked bar
    ax4 = axes[1, 1]
    vpivot = (signal_df.pivot_table(index="date", columns="ticker",
                                    values="post_count", aggfunc="sum")
              .fillna(0))
    bottom = np.zeros(len(vpivot))
    for i, ticker in enumerate(TICKERS):
        if ticker in vpivot.columns:
            ax4.bar(vpivot.index, vpivot[ticker].values,
                    bottom=bottom, label=ticker, color=PALETTE[i], alpha=0.85)
            bottom += vpivot[ticker].values
    ax4.set_title("D — Reddit Post Volume (Attention Signal)", color="#58a6ff", fontweight="bold")
    ax4.set_ylabel("Number of Posts")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30)
    ax4.legend(fontsize=8, framealpha=0.3)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_time_series(ts_results: dict,
                     save_path: str = "outputs/figure2_time_series.png"):
    """
    Figure 2 — Time-series analysis (3 panels):
        Panel A : Raw sentiment + 7-day rolling mean + fill
        Panel B : ACF plot (autocorrelation function)
        Panel C : ARIMA(1,0,1) 5-day forecast with 95% CI
    """
    from statsmodels.graphics.tsaplots import plot_acf

    ts       = ts_results["ts_model"]
    fcast    = ts_results["forecast"]
    conf     = ts_results["conf_int"]
    arima    = ts_results["arima_model"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("Time Series Analysis — NVDA Reddit Sentiment",
                 fontsize=14, fontweight="bold", color="#58a6ff")

    # A
    ax = axes[0]
    ax.plot(ts.index, ts.values, color="#58a6ff", linewidth=1.2, alpha=0.7, label="Daily")
    ax.plot(ts.index, ts.rolling(7).mean(), color="#ffa657", linewidth=2.0, label="7-day MA")
    ax.fill_between(ts.index, 0, ts.values, where=ts.values > 0, color="#3fb950", alpha=0.15)
    ax.fill_between(ts.index, 0, ts.values, where=ts.values < 0, color="#f78166", alpha=0.15)
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_title("A — Sentiment + 7-day Rolling Mean", color="#8b949e", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # B
    ax2 = axes[1]
    plot_acf(ts, lags=min(20, len(ts) // 2 - 1), ax=ax2, color="#58a6ff",
             title="B — Autocorrelation Function (ACF)", zero=False, alpha=0.05)
    ax2.set_xlabel("Lag (days)")
    ax2.title.set_color("#8b949e"); ax2.title.set_fontsize(10)
    ax2.grid(True, alpha=0.3)

    # C
    ax3 = axes[2]
    ax3.plot(ts.index, ts.values, color="#58a6ff", linewidth=1.2, alpha=0.7, label="Historical")
    fc_idx = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=5, freq="D")
    ax3.plot(fc_idx, fcast.values, color="#ffa657", linewidth=2.0,
             marker="o", markersize=5, label="ARIMA Forecast")
    ax3.fill_between(fc_idx, conf.iloc[:, 0], conf.iloc[:, 1],
                     color="#ffa657", alpha=0.2, label="95% CI")
    ax3.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax3.set_title(f"C — ARIMA(1,0,1) 5-Day Forecast  (AIC={arima.aic:.1f})",
                  color="#8b949e", fontsize=10)
    ax3.legend(fontsize=8, framealpha=0.3)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_backtest_and_metrics(signal_df: pd.DataFrame,
                              metrics: pd.DataFrame,
                              save_path: str = "outputs/figure3_backtest.png"):
    """
    Figure 3 — Backtesting & performance metrics (2 rows):
        Row 1 : Equity curves for each ticker (5 subplots)
        Row 2 : Sharpe, Max Drawdown, Win Rate bar charts (3 subplots)
    """
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Backtest: Sentiment Strategy vs Buy & Hold — Performance Metrics",
                 fontsize=14, fontweight="bold", color="#58a6ff")

    # Row 1 — Equity curves
    for i, ticker in enumerate(TICKERS):
        ax = fig.add_subplot(2, 5, i + 1)
        sub = (signal_df[signal_df["ticker"] == ticker]
               .copy().sort_values("date"))
        sub["position"]  = np.where(sub["sentiment_score"] >  0.10,  1,
                           np.where(sub["sentiment_score"] < -0.10, -1, 0))
        sub["strat_ret"] = sub["position"] * sub["return_1d"]
        cum_strat = (1 + sub["strat_ret"]).cumprod()
        cum_bh    = (1 + sub["return_1d"]).cumprod()

        ax.plot(sub["date"].values, cum_bh.values,
                color="#8b949e", linewidth=1.5, linestyle="--",
                label="B&H", alpha=0.8)
        ax.plot(sub["date"].values, cum_strat.values,
                color=PALETTE[i], linewidth=2.2, label="Strategy")
        ax.axhline(1.0, color="#8b949e", linewidth=0.6, linestyle=":")
        ax.set_title(f"${ticker}", color=PALETTE[i], fontweight="bold")
        ax.legend(fontsize=6, framealpha=0.3)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#161b22")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, fontsize=6)

    # Row 2 — Metric bars
    metric_configs = [
        ("sharpe_ratio",  "Sharpe Ratio (annualised)", "#58a6ff"),
        ("max_drawdown",  "Max Drawdown (%)",           "#f78166"),
        ("win_rate",      "Win Rate (%)",                "#3fb950"),
    ]
    for j, (col, title, color) in enumerate(metric_configs):
        ax = fig.add_subplot(2, 5, 6 + j)
        vals   = metrics[col].values
        tickers = metrics.index.tolist()
        bars = ax.bar(tickers, vals, color=color, alpha=0.8, edgecolor="#21262d")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + np.sign(val) * 0.3,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#e6edf3")
        ax.set_title(title, color="#8b949e", fontsize=9)
        ax.set_facecolor("#161b22")
        ax.grid(True, alpha=0.25, axis="y")

    # Profit factor plot
    ax = fig.add_subplot(2, 5, 9)
    pf_vals = metrics["profit_factor"].values
    bars = ax.bar(metrics.index.tolist(), pf_vals,
                  color="#d2a8ff", alpha=0.8, edgecolor="#21262d")
    ax.axhline(1.0, color="#8b949e", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, pf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#e6edf3")
    ax.set_title("Profit Factor", color="#8b949e", fontsize=9)
    ax.set_facecolor("#161b22")
    ax.grid(True, alpha=0.25, axis="y")

    # Trade count
    ax = fig.add_subplot(2, 5, 10)
    tc_vals = metrics["trade_count"].values
    bars = ax.bar(metrics.index.tolist(), tc_vals,
                  color="#ffa657", alpha=0.8, edgecolor="#21262d")
    for bar, val in zip(bars, tc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(int(val)), ha="center", va="bottom", fontsize=8, color="#e6edf3")
    ax.set_title("Trade Count (signal days)", color="#8b949e", fontsize=9)
    ax.set_facecolor("#161b22")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_extended_analysis(signal_df: pd.DataFrame,
                           ts_results: dict,
                           save_path: str = "outputs/figure4_extended.png"):
    """
    Figure 4 — Extended analysis (new insights):
        Panel A : Rolling 15-day correlation: sentiment vs return (per ticker)
        Panel B : Cross-ticker sentiment correlation heatmap
        Panel C : Sentiment dispersion (intra-day std) vs realized volatility
        Panel D : Bull ratio over time (fraction of bullish posts per day)
    """
    rolling_corr = ts_results["rolling_corr"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Extended Analysis: Deeper Signal Insights",
                 fontsize=16, fontweight="bold", color="#58a6ff", y=1.01)

    # A — Rolling correlation
    ax1 = axes[0, 0]
    for i, ticker in enumerate(TICKERS):
        if ticker in rolling_corr.columns:
            ax1.plot(rolling_corr.index, rolling_corr[ticker],
                     label=ticker, color=PALETTE[i], linewidth=1.6, alpha=0.85)
    ax1.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)
    ax1.axhline(0.3, color="#3fb950", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.axhline(-0.3, color="#f78166", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_title("A — Rolling 15-day Sentiment↔Return Correlation",
                  color="#58a6ff", fontweight="bold")
    ax1.set_ylabel("Pearson r")
    ax1.legend(fontsize=8, framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # B — Cross-ticker sentiment heatmap
    ax2 = axes[0, 1]
    pivot = signal_df.pivot_table(index="date", columns="ticker",
                                   values="sentiment_score")
    corr_matrix = pivot.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, ax=ax2, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#30363d",
                annot_kws={"size": 11, "color": "#e6edf3"},
                cbar_kws={"shrink": 0.8})
    ax2.set_title("B — Cross-Ticker Sentiment Correlation",
                  color="#58a6ff", fontweight="bold")
    ax2.tick_params(colors="#e6edf3")

    # C — Sentiment dispersion vs volatility
    ax3 = axes[1, 0]
    plot_df = signal_df.dropna(subset=["sentiment_std", "volatility"])
    for i, ticker in enumerate(TICKERS):
        sub = plot_df[plot_df["ticker"] == ticker]
        ax3.scatter(sub["sentiment_std"], sub["volatility"] * 100,
                    label=ticker, alpha=0.6, s=40, color=PALETTE[i])
    ax3.set_title("C — Sentiment Dispersion vs Realized Volatility",
                  color="#58a6ff", fontweight="bold")
    ax3.set_xlabel("Intra-day Sentiment Std Dev")
    ax3.set_ylabel("5-day Realized Volatility (%)")
    ax3.legend(fontsize=8, framealpha=0.3)
    ax3.grid(True, alpha=0.3)

    # D — Bull ratio over time
    ax4 = axes[1, 1]
    for i, ticker in enumerate(TICKERS):
        sub = (signal_df[signal_df["ticker"] == ticker]
               .sort_values("date"))
        if "bull_ratio" in sub.columns:
            ax4.plot(sub["date"], sub["bull_ratio"].rolling(5).mean(),
                     label=ticker, color=PALETTE[i], linewidth=1.6, alpha=0.85)
    ax4.axhline(0.5, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.6)
    ax4.set_title("D — Bullish Post Ratio (5-day rolling avg)",
                  color="#58a6ff", fontweight="bold")
    ax4.set_ylabel("Fraction of bullish posts")
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8, framealpha=0.3)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(use_live_reddit: bool = False,
         use_finbert:     bool = False,
         reddit_client_id:     str = "",
         reddit_client_secret: str = ""):
    """
    Orchestrate the full pipeline.

    Parameters
    ----------
    use_live_reddit      : If True, scrape Reddit live (requires credentials)
    use_finbert          : If True, load FinBERT for real NLP scoring
    reddit_client_id     : PRAW client ID (needed if use_live_reddit=True)
    reddit_client_secret : PRAW secret  (needed if use_live_reddit=True)
    """
    import os
    os.makedirs("outputs", exist_ok=True)

    # 1. Data
    if use_live_reddit:
        all_posts = []
        for ticker in TICKERS:
            for sub in SUBREDDITS:
                df_sub = scrape_reddit_live(ticker, sub,
                                            reddit_client_id,
                                            reddit_client_secret, limit=30)
                all_posts.append(df_sub)
        raw_df = pd.concat(all_posts, ignore_index=True)
    else:
        raw_df = generate_sample_data()

    # 2. Sentiment
    if use_finbert:
        tokenizer, model = load_finbert()
        scored_df = apply_finbert(raw_df, tokenizer, model)
    else:
        scored_df = mock_sentiment(raw_df)

    # 3. Signal
    daily_sentiment = build_daily_signal(scored_df)

    # 4. Price
    price_data = fetch_price_data()

    # 5. Merge
    signal_df = merge_signals(daily_sentiment, price_data)

    # 6. Statistics
    stats_results = run_statistical_analysis(signal_df)
    signal_df = stats_results["signal_df_with_quintile"]

    # 7. Time series
    ts_results = run_time_series_analysis(signal_df)

    # 8. Backtest
    metrics = run_backtest(signal_df)
    print(f"\nPerformance Summary:\n{metrics.to_string()}\n")

    # 9. Plots
    plot_main_dashboard(signal_df, stats_results)
    plot_time_series(ts_results)
    plot_backtest_and_metrics(signal_df, metrics)
    plot_extended_analysis(signal_df, ts_results)

    print("\n[DONE] All outputs saved to outputs/ directory.")
    return signal_df, stats_results, ts_results, metrics


if __name__ == "__main__":
    main(use_live_reddit=False, use_finbert=False)
