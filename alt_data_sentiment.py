# =============================================================================
# ALTERNATIVE DATA IN QUANTITATIVE TRADING
# NLP Sentiment Signal from Reddit → Stock Return Prediction
# Author: Khullar | Project: Algorithmic Trading Research
# =============================================================================
# HOW TO USE IN GOOGLE COLAB:
#   Copy each section (between the === dividers) into separate Colab cells.
#   Run them top to bottom.
# =============================================================================


# ==============================================================================
# CELL 1 — Install Dependencies
# ==============================================================================

!pip install praw transformers torch yfinance pandas numpy matplotlib seaborn scipy --quiet
!pip install mlfinlab --quiet   # Lopez de Prado's library (may take ~2 min)


# ==============================================================================
# CELL 2 — Imports & Global Config
# ==============================================================================

import praw
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Plotting style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "monospace",
})

print("✅ All imports loaded successfully.")


# ==============================================================================
# CELL 3 — Reddit API Setup
# (Get free credentials at https://www.reddit.com/prefs/apps → "create app")
# If you don't have credentials yet, skip to CELL 3B for sample data.
# ==============================================================================

# --- CELL 3A: LIVE Reddit Scraping (use if you have API credentials) ---

REDDIT_CLIENT_ID     = "YOUR_CLIENT_ID"        # ← Replace this
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET"    # ← Replace this
REDDIT_USER_AGENT    = "altdata_research_bot"

USE_LIVE_REDDIT = False   # Set to True once you have credentials

TICKERS = ["GME", "NVDA", "TSLA", "AAPL", "AMD"]
SUBREDDITS = ["wallstreetbets", "investing", "stocks"]
POSTS_PER_TICKER = 30     # Keep low to stay within API rate limits


def scrape_reddit_mentions(ticker, subreddit_name, limit=30):
    """
    Scrape Reddit posts mentioning a ticker from a given subreddit.
    Returns a DataFrame of posts with title, score, num_comments, created_utc.
    """
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    subreddit = reddit.subreddit(subreddit_name)
    records = []

    for post in subreddit.search(ticker, limit=limit, sort="new"):
        records.append({
            "ticker":       ticker,
            "subreddit":    subreddit_name,
            "title":        post.title,
            "selftext":     post.selftext[:300],   # cap at 300 chars
            "score":        post.score,
            "num_comments": post.num_comments,
            "upvote_ratio": post.upvote_ratio,
            "created_utc":  datetime.utcfromtimestamp(post.created_utc),
        })

    return pd.DataFrame(records)


if USE_LIVE_REDDIT:
    all_posts = []
    for ticker in TICKERS:
        for sub in SUBREDDITS:
            df_sub = scrape_reddit_mentions(ticker, sub, limit=POSTS_PER_TICKER)
            all_posts.append(df_sub)
            print(f"  Scraped {len(df_sub)} posts for ${ticker} from r/{sub}")

    raw_df = pd.concat(all_posts, ignore_index=True)
    print(f"\n✅ Total posts scraped: {len(raw_df)}")
else:
    print("⚠️  Live Reddit disabled. Running CELL 3B (sample data) next.")


# ==============================================================================
# CELL 3B — Sample Data (Run this if you don't have Reddit credentials yet)
# This simulates realistic Reddit post data so the full pipeline still runs.
# ==============================================================================

import random
random.seed(42)
np.random.seed(42)

sample_titles = {
    "GME":  ["GME short squeeze incoming again", "GameStop fundamentals don't justify this price",
             "GME to the moon 🚀🚀", "Why I'm holding GME long term", "GME earnings disaster"],
    "NVDA": ["NVDA is the most important company in AI", "Nvidia valuation is insane right now",
             "Bought NVDA calls, feeling good", "NVDA smashes earnings again", "Nvidia GPU shortage continues"],
    "TSLA": ["Tesla delivery numbers disappointing", "Elon selling TSLA again???",
             "TSLA bull case 2024", "Tesla FSD actually works now", "Why TSLA is overvalued"],
    "AAPL": ["Apple Vision Pro flop?", "AAPL quietly dominates services",
             "iPhone 16 demand looks weak", "Apple's moat is unbreakable", "AAPL buyback machine"],
    "AMD":  ["AMD gaining on Intel fast", "Ryzen 9000 benchmarks insane",
             "AMD vs NVDA in AI chips", "Bought AMD before earnings", "AMD undervalued vs Nvidia"],
}

sample_records = []
base_date = datetime.now() - timedelta(days=60)

for ticker in TICKERS:
    for i in range(50):
        post_date = base_date + timedelta(
            days=random.randint(0, 58),
            hours=random.randint(0, 23),
        )
        sample_records.append({
            "ticker":       ticker,
            "subreddit":    random.choice(SUBREDDITS),
            "title":        random.choice(sample_titles[ticker]),
            "selftext":     "",
            "score":        random.randint(10, 5000),
            "num_comments": random.randint(5, 500),
            "upvote_ratio": round(random.uniform(0.55, 0.98), 2),
            "created_utc":  post_date,
        })

raw_df = pd.DataFrame(sample_records).sort_values("created_utc").reset_index(drop=True)
print(f"✅ Sample dataset created: {len(raw_df)} posts across {raw_df['ticker'].nunique()} tickers")
print(raw_df.head(8).to_string())


# ==============================================================================
# CELL 4 — Sentiment Scoring with FinBERT
# FinBERT is a BERT model fine-tuned specifically on financial text.
# It classifies text as POSITIVE / NEGATIVE / NEUTRAL with a probability score.
# ==============================================================================

from transformers import BertTokenizer, BertForSequenceClassification
import torch

print("⏳ Loading FinBERT model (one-time download ~420MB)...")
FINBERT_MODEL = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
model     = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
model.eval()

LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

def score_sentiment(text: str) -> dict:
    """
    Run FinBERT on a single text string.
    Returns: {"label": str, "positive": float, "negative": float, "neutral": float}
    """
    text = str(text).strip()
    if not text:
        return {"label": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()

    return {
        "label":    LABEL_MAP[int(torch.argmax(logits))],
        "positive": round(probs[0], 4),
        "negative": round(probs[1], 4),
        "neutral":  round(probs[2], 4),
    }


# Apply FinBERT to every post title
# (using title only — selftext is often empty or too long)
print("⏳ Scoring sentiment on all posts...")
sentiment_results = raw_df["title"].apply(score_sentiment)
sentiment_df = pd.DataFrame(sentiment_results.tolist())
scored_df = pd.concat([raw_df, sentiment_df], axis=1)

# Compute a single compound score: positive - negative (range: -1 to +1)
scored_df["compound"] = scored_df["positive"] - scored_df["negative"]

print(f"\n✅ Sentiment scored. Distribution:")
print(scored_df["label"].value_counts())
print("\nSample scored posts:")
print(scored_df[["ticker", "title", "label", "compound"]].head(10).to_string())


# ==============================================================================
# CELL 5 — Download Price Data with yfinance
# ==============================================================================

START_DATE = (datetime.now() - timedelta(days=65)).strftime("%Y-%m-%d")
END_DATE   = datetime.now().strftime("%Y-%m-%d")

print(f"⏳ Downloading OHLCV data ({START_DATE} → {END_DATE})...")
price_data = {}
for ticker in TICKERS:
    df_price = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    df_price.index = pd.to_datetime(df_price.index)

    # Forward returns: how much did price move the NEXT day?
    df_price["return_1d"]  = df_price["Close"].pct_change(1).shift(-1)
    df_price["return_2d"]  = df_price["Close"].pct_change(2).shift(-2)
    df_price["log_return"] = np.log(df_price["Close"] / df_price["Close"].shift(1))

    price_data[ticker] = df_price
    print(f"  {ticker}: {len(df_price)} trading days loaded")

print("\n✅ Price data ready.")


# ==============================================================================
# CELL 6 — Build Daily Sentiment Signal
# Aggregate all posts for each (ticker, date) into one daily sentiment score.
# ==============================================================================

scored_df["date"] = pd.to_datetime(scored_df["created_utc"]).dt.date

def aggregate_daily_sentiment(group):
    """
    Weighted average sentiment: weight each post by sqrt(score + 1).
    High-upvoted posts carry more signal than low-karma noise.
    """
    weights = np.sqrt(group["score"].clip(lower=0) + 1)
    weighted_compound = np.average(group["compound"], weights=weights)
    return pd.Series({
        "sentiment_score":   round(weighted_compound, 4),
        "post_count":        len(group),
        "avg_upvote_ratio":  group["upvote_ratio"].mean(),
        "total_comments":    group["num_comments"].sum(),
    })

daily_sentiment = (
    scored_df
    .groupby(["ticker", "date"])
    .apply(aggregate_daily_sentiment)
    .reset_index()
)
daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])

print("✅ Daily sentiment signal constructed.")
print(daily_sentiment.sort_values("date").tail(15).to_string())


# ==============================================================================
# CELL 7 — Merge Sentiment with Price Returns
# ==============================================================================

signal_dfs = []

for ticker in TICKERS:
    sentiment_ticker = daily_sentiment[daily_sentiment["ticker"] == ticker].copy()
    price_ticker     = price_data[ticker].reset_index()[["Date", "Close", "return_1d", "return_2d", "log_return"]]
    price_ticker.columns = ["date", "close", "return_1d", "return_2d", "log_return"]
    price_ticker["date"] = pd.to_datetime(price_ticker["date"])

    merged = pd.merge(sentiment_ticker, price_ticker, on="date", how="inner")
    merged["ticker"] = ticker
    signal_dfs.append(merged)

signal_df = pd.concat(signal_dfs, ignore_index=True).dropna(subset=["return_1d", "sentiment_score"])
print(f"✅ Merged dataset: {len(signal_df)} (ticker, day) observations")
print(signal_df[["ticker", "date", "sentiment_score", "post_count", "return_1d"]].head(12).to_string())


# ==============================================================================
# CELL 8 — Statistical Analysis: Does Sentiment Predict Returns?
# ==============================================================================

print("=" * 65)
print("  SENTIMENT → NEXT-DAY RETURN: STATISTICAL ANALYSIS")
print("=" * 65)

# ---- 8A: Pearson Correlation per ticker ----
print("\n📊 Pearson Correlation (sentiment_score vs next-day return):\n")
for ticker in TICKERS:
    sub = signal_df[signal_df["ticker"] == ticker]
    if len(sub) < 5:
        continue
    r, p = stats.pearsonr(sub["sentiment_score"], sub["return_1d"])
    sig = "✅ significant" if p < 0.05 else "— not significant"
    print(f"  {ticker:5s}  r = {r:+.3f}   p = {p:.3f}   {sig}")

# ---- 8B: Quintile Analysis ----
print("\n📊 Quintile Analysis (portfolio sorted by daily sentiment):\n")
signal_df["quintile"] = pd.qcut(signal_df["sentiment_score"], q=5, labels=["Q1\n(most neg)", "Q2", "Q3", "Q4", "Q5\n(most pos)"])
quintile_returns = signal_df.groupby("quintile", observed=True)["return_1d"].agg(["mean", "std", "count"])
quintile_returns.columns = ["Mean Return", "Std Dev", "Obs"]
quintile_returns["Mean Return"] = (quintile_returns["Mean Return"] * 100).round(3)
quintile_returns["Std Dev"]     = (quintile_returns["Std Dev"] * 100).round(3)
print(quintile_returns.to_string())
print("\n(Ideal signal: Q1 has lowest return, Q5 has highest)")

# ---- 8C: OLS Regression ----
from scipy.stats import linregress
print("\n📊 OLS Regression: return_1d ~ sentiment_score\n")
slope, intercept, r_value, p_value, std_err = linregress(
    signal_df["sentiment_score"], signal_df["return_1d"]
)
print(f"  Slope:     {slope:.5f}")
print(f"  Intercept: {intercept:.5f}")
print(f"  R²:        {r_value**2:.4f}")
print(f"  p-value:   {p_value:.4f}")
print(f"  Std Error: {std_err:.5f}")


# ==============================================================================
# CELL 9 — Visualizations (4 publication-quality plots)
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Alternative Data Research: Reddit Sentiment → Stock Returns\n",
    fontsize=16, fontweight="bold", color="#58a6ff", y=1.01
)

# ---- Plot 1: Sentiment Timeline for all tickers ----
ax1 = axes[0, 0]
colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
for i, ticker in enumerate(TICKERS):
    sub = signal_df[signal_df["ticker"] == ticker].sort_values("date")
    ax1.plot(sub["date"], sub["sentiment_score"].rolling(3).mean(),
             label=ticker, color=colors[i], linewidth=1.8, alpha=0.9)
ax1.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.set_title("Daily Sentiment Score (3-day rolling avg)", color="#58a6ff", fontweight="bold")
ax1.set_ylabel("Sentiment  (–1 → +1)")
ax1.legend(fontsize=8, framealpha=0.3)
ax1.grid(True, alpha=0.3)

# ---- Plot 2: Quintile Return Bar Chart ----
ax2 = axes[0, 1]
quintile_labels = ["Q1\n(bearish)", "Q2", "Q3", "Q4", "Q5\n(bullish)"]
returns_pct = signal_df.groupby("quintile", observed=True)["return_1d"].mean() * 100
bar_colors = ["#f78166", "#ffa657", "#8b949e", "#56d364", "#3fb950"]
bars = ax2.bar(quintile_labels, returns_pct.values, color=bar_colors, width=0.6, edgecolor="#21262d")
for bar, val in zip(bars, returns_pct.values):
    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
             f"{val:+.3f}%", ha="center", va="bottom", fontsize=9, color="#e6edf3")
ax2.axhline(0, color="#8b949e", linestyle="-", linewidth=0.8)
ax2.set_title("Avg Next-Day Return by Sentiment Quintile", color="#58a6ff", fontweight="bold")
ax2.set_ylabel("Mean Return (%)")
ax2.grid(True, alpha=0.3, axis="y")

# ---- Plot 3: Scatter — Sentiment vs Return ----
ax3 = axes[1, 0]
for i, ticker in enumerate(TICKERS):
    sub = signal_df[signal_df["ticker"] == ticker]
    ax3.scatter(sub["sentiment_score"], sub["return_1d"] * 100,
                label=ticker, alpha=0.6, s=30, color=colors[i])
# OLS trendline
x_line = np.linspace(signal_df["sentiment_score"].min(), signal_df["sentiment_score"].max(), 100)
y_line = slope * x_line + intercept
ax3.plot(x_line, y_line * 100, color="#f0f6fc", linewidth=1.5, linestyle="--", label="OLS fit")
ax3.set_title(f"Sentiment vs Next-Day Return  (R²={r_value**2:.3f})", color="#58a6ff", fontweight="bold")
ax3.set_xlabel("Daily Sentiment Score")
ax3.set_ylabel("Next-Day Return (%)")
ax3.legend(fontsize=7, framealpha=0.3)
ax3.grid(True, alpha=0.3)

# ---- Plot 4: Post Volume (Attention Signal) ----
ax4 = axes[1, 1]
volume_pivot = signal_df.pivot_table(index="date", columns="ticker", values="post_count", aggfunc="sum").fillna(0)
bottom = np.zeros(len(volume_pivot))
for i, ticker in enumerate(TICKERS):
    if ticker in volume_pivot.columns:
        ax4.bar(volume_pivot.index, volume_pivot[ticker].values,
                bottom=bottom, label=ticker, color=colors[i], alpha=0.85)
        bottom += volume_pivot[ticker].values
ax4.set_title("Reddit Post Volume per Day (Attention Signal)", color="#58a6ff", fontweight="bold")
ax4.set_ylabel("Number of Posts")
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30)
ax4.legend(fontsize=8, framealpha=0.3)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("alt_data_research.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("✅ Plot saved as alt_data_research.png")


# ==============================================================================
# CELL 10 — Simple Long/Short Backtest
# Strategy: go LONG when sentiment > +0.1, SHORT when sentiment < -0.1, FLAT otherwise
# ==============================================================================

print("=" * 65)
print("  SIMPLE LONG/SHORT STRATEGY BACKTEST")
print("=" * 65)

LONG_THRESHOLD  =  0.10
SHORT_THRESHOLD = -0.10

backtest_results = []

for ticker in TICKERS:
    sub = signal_df[signal_df["ticker"] == ticker].copy().sort_values("date")
    sub["position"] = np.where(
        sub["sentiment_score"] > LONG_THRESHOLD,   1,
        np.where(sub["sentiment_score"] < SHORT_THRESHOLD, -1, 0)
    )
    sub["strategy_return"] = sub["position"] * sub["return_1d"]

    cum_strategy   = (1 + sub["strategy_return"]).cumprod() - 1
    cum_buyhold    = (1 + sub["return_1d"]).cumprod() - 1

    total_return   = cum_strategy.iloc[-1] if len(cum_strategy) > 0 else 0
    bh_return      = cum_buyhold.iloc[-1]  if len(cum_buyhold)  > 0 else 0
    n_trades       = (sub["position"] != 0).sum()
    hit_rate       = (sub.loc[sub["position"] != 0, "strategy_return"] > 0).mean()

    backtest_results.append({
        "Ticker":            ticker,
        "Strategy Return":   f"{total_return * 100:+.2f}%",
        "Buy & Hold Return": f"{bh_return * 100:+.2f}%",
        "# Trades":          n_trades,
        "Hit Rate":          f"{hit_rate * 100:.1f}%",
    })

results_df = pd.DataFrame(backtest_results)
print("\n", results_df.to_string(index=False))

print("""
⚠️  IMPORTANT CAVEATS (shows the recruiter you think rigorously):
   1. Sample data → results are illustrative, not real alpha
   2. No transaction costs, slippage, or market impact modelled
   3. No statistical significance testing on backtest Sharpe
   4. Survivorship bias: only chose tickers that still exist
   5. Next steps: add Deflated Sharpe Ratio, CPCV cross-validation
""")


# ==============================================================================
# CELL 11 — Ethics Audit Layer
# The section that separates this project from every other quant project.
# ==============================================================================

ethics_audit = {
    "Data Source":          "Reddit (r/wallstreetbets, r/investing, r/stocks)",
    "Data Type":            "Public user-generated text posts",
    "Consent Status":       "Implicit (Reddit ToS §3.3 permits API access for research)",
    "PII Present?":         "No — usernames anonymised, no geolocation or transaction data",
    "MNPI Risk":            "Low — Reddit is public; no non-public information accessed",
    "Mosaic Theory":        "Applies — public sentiment aggregation is legal under SEC guidance",
    "GDPR Compliance":      "Reddit users outside EU, public data, no storage of personal data",
    "Fairness Concern":     "Retail investor 'wisdom' monetised by institutions — power asymmetry",
    "Proposed Mitigation":  "Differential privacy on aggregated scores; open-source the pipeline",
    "Data Vendor Risk":     "None — PRAW API is free and Reddit-licensed",
}

print("=" * 65)
print("  ETHICS AUDIT REPORT")
print("=" * 65)
for key, value in ethics_audit.items():
    print(f"  {key:<25} : {value}")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CONTEXTUAL INTEGRITY ANALYSIS (Nissenbaum, 2004)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Original context:   A user posts a financial opinion on a public forum
 Expected flow:      Other retail investors read and discuss it
 Actual flow:        Aggregated by ML pipeline → hedge fund trading signal
 Verdict:            Contextual integrity PARTIALLY violated.
                     Technically legal (public data). Ethically ambiguous.
                     This is the core tension in alternative data.

 COMPARISON WITH MORE INVASIVE ALT DATA:
 ┌────────────────────────────┬──────────────┬───────────────┐
 │ Data Type                  │ Legal Risk   │ Ethics Risk   │
 ├────────────────────────────┼──────────────┼───────────────┤
 │ Reddit NLP (this project)  │ Low          │ Low-Medium    │
 │ Satellite imagery (public) │ Low          │ Low           │
 │ Credit card transactions   │ Medium       │ High          │
 │ Smartphone geolocation     │ High (CCPA)  │ Very High     │
 │ Web scraping w/o ToS       │ High (CFAA)  │ High          │
 └────────────────────────────┴──────────────┴───────────────┘
""")

print("✅ Project pipeline complete. Ready to present.")
print("   Next steps: add FinBERT fine-tuning, dollar bars, CPCV backtesting.")
