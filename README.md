# X Sentiment Price Prediction Agent — Bitcoin (BTC)

Internship assessment for **CrowdWisdomTrading** (via TZURONI): a Python
ML pipeline that predicts an asset's price at horizons **n+0 … n+5** using
**only social sentiment + price**, with rigorous walk-forward
(out-of-sample) validation and a trading evaluation (Sharpe, Sortino,
Max Drawdown). Chosen asset: **Bitcoin (BTC-USD)**.

## What it does

1. **Ingestion**
   - Price: BTC-USD daily OHLCV, last 180 days (Yahoo Finance / yfinance,
     with the Kraken public OHLC API as an automatic real-data fallback —
     the loader records which source served each run).
   - X posts: **Apify** scraper actor (`apidojo/tweet-scraper`, configurable)
     over `$BTC / Bitcoin / BTCUSD` — runs when `APIFY_API_TOKEN` is set.
   - Live sentiment (executed run): **StockTwits** public BTC.X stream —
     real messages with author-labelled Bullish/Bearish tags (no key needed).
   - News snapshot (context only, never a feature): Tavily if a key is
     set, else CoinTelegraph/CoinDesk/Decrypt RSS.
2. **Database join** — daily sentiment aggregates are joined with price
   bars in **SQLite** (`data/dataset.db`); joined CSV also exported.
3. **Features (sentiment + price only)** — lagged returns, rolling
   volatility, sentiment level / 3d & 7d moving averages / momentum,
   bullish ratio, post-volume z-scores, day-of-week.
4. **Model** — one `RandomForestRegressor` per horizon, **expanding-window
   walk-forward validation** (initial train 90 days, 7-day test blocks,
   refit each block; a training row counts only once its full target
   window has closed — no lookahead).
5. **Evaluation** — per-horizon RMSE/MAE + directional accuracy vs
   zero-return and persistence baselines; long/flat strategy from the
   n+0 model scored with **Sharpe, Sortino, Max Drawdown** (365-day
   annualisation) against buy-and-hold over the same OOS days.

**Horizon convention:** origin = close of day *t*; horizon *h* is the
cumulative return from close(*t*) to close(*t+h+1*). So n+0 = next-day
return, n+5 = 6-day-ahead cumulative return.

## Honest data note

The assignment's "historical tweets" document link is dead ("file does
not exist"), and no Apify token existed at build time — so the executed
run's sentiment comes from the live StockTwits stream, which covers the
most recent days densely (coverage is measured and reported, e.g. in
`outputs/EVALUATION_REPORT.md`; unobserved days use neutral sentiment
with an explicit `sent_observed` flag). The Apify X-ingestion path is
fully implemented (`src/apify_ingest.py`) and merges into the identical
dataset/feature code the moment a token is placed in `.env`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # optional: add free-tier APIFY_API_TOKEN / TAVILY_API_KEY
python run_pipeline.py
```

## Outputs

| File | What |
|---|---|
| `data/price_daily.csv` | 180 days of BTC-USD OHLCV |
| `data/stocktwits_messages.csv` | Raw live-stream messages + labels |
| `data/x_posts_apify.csv` | Raw X posts (only when Apify token set) |
| `data/dataset.csv` / `data/dataset.db` | Joined daily dataset (CSV + SQLite) |
| `outputs/walk_forward_metrics.csv` | Per-horizon OOS metrics |
| `outputs/oos_predictions.csv` | Every out-of-sample prediction |
| `outputs/chart_price_vs_sentiment.png` | Price + sentiment overlay |
| `outputs/chart_pred_vs_actual_h0.png` | Predicted vs actual, n+0 |
| `outputs/chart_equity_drawdown.png` | Strategy equity + drawdown |
| `outputs/SENTIMENT_SUMMARY.md` | ~1-page current BTC sentiment outlook |
| `outputs/EVALUATION_REPORT.md` | Full evaluation report with all metrics |

## Tokens & actors (per the assignment's deliverables)

Secrets live only in `.env` (git-ignored) and are supplied by the
candidate from his own free-tier accounts at submission time — never
committed. Configured actor: **`apidojo/tweet-scraper`** (env
`APIFY_ACTOR_ID`), search terms `$BTC`, `Bitcoin`, `BTCUSD`. Tavily
(`TAVILY_API_KEY`) powers news; Exa (`EXA_API_KEY`) is reserved as the
alternative search source named in the assignment.

## Layout

```
run_pipeline.py          # single entry point
src/config.py            # tickers, horizons, actors, token loading
src/price_ingest.py      # yfinance OHLCV
src/apify_ingest.py      # X scraping via Apify (token-gated)
src/stocktwits_ingest.py # public live sentiment stream
src/news_ingest.py       # Tavily / RSS news snapshot
src/features.py          # SQLite join + feature engineering
src/model.py             # walk-forward validation + strategy metrics
src/charts.py            # matplotlib charts
```

*Not financial advice. Crypto returns are near-random at daily horizons;
all metrics are reported exactly as computed, weak or strong.*
