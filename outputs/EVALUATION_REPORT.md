# Evaluation Report — X Sentiment Price Prediction Agent (BTC)

_Generated 2026-11-01 by `run_pipeline.py`. Every number below is
computed from the data files in `data/` during this run — nothing is
hand-entered._

## 1. Objective
Predict Bitcoin's (BTC-USD) price at horizons n+0 … n+5 using **only**
X/social sentiment and price, per the CrowdWisdomTrading intern task
(via TZURONI). Convention used here: the forecast origin is the close of
day t; horizon h targets the cumulative return from close(t) to
close(t+h+1) — so n+0 is the next-day return and n+5 is the 6-day-ahead
cumulative return. A tradable rule is evaluated on out-of-sample
predictions only.

## 2. Data sources (and one transparency note)
- **Price:** BTC-USD daily OHLCV — 180
  rows, 2026-05-06 to 2026-11-01.
  Source actually used in this run: **Yahoo Finance (yfinance), BTC-USD**
  (yfinance is the primary loader; the Kraken public OHLC API is the
  automatic fallback when Yahoo is unreachable — both are real market
  data, and the loader records which one served the run).
- **X sentiment (primary, per assignment):** Apify actor
  `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest` with search terms ['$BTC', 'Bitcoin', 'BTCUSD'].
  X posts via Apify were included in this run (8298 posts).
- **Live sentiment actually executed:** StockTwits public stream for
  BTC.X — 1500 messages, 2026-10-30 20:44:47+00:00 to 2026-11-01 19:50:40+00:00 UTC, 844 author-labelled Bullish/Bearish. StockTwits labels are the authors' own Bullish/Bearish
  tags; unlabelled posts are scored with VADER.
- **Why a fallback was needed:** the assignment document links a
  "historical tweets" Google Doc for training data; that link is dead
  ("Sorry, the file you have requested does not exist"), and no Apify
  token existed at build time. StockTwits is the documented substitute;
  the Apify ingestion is implemented (`src/apify_ingest.py`) and merges
  into the same dataset when a token is provided.
- **News (context only, never a feature):** Tavily when a key is set,
  else CoinTelegraph/CoinDesk/Decrypt RSS — 6
  headlines in `data/news_latest.csv`.
- **Join:** daily sentiment is aggregated per day and joined with price
  bars in **SQLite** (`data/dataset.db`, tables `price_daily`,
  `sentiment_daily`, `dataset_joined`); the joined table is also exported
  to `data/dataset.csv`. Sentiment is observed on 74% of the 180
  days (the live stream covers recent days densely); unobserved days use
  neutral 0 sentiment with an explicit `sent_observed` flag — a stated
  limitation, not hidden imputation.

## 3. Features (sentiment + price only)
From `src/features.py`: lagged returns (1/3/5/7d), rolling volatility
(7/14d), daily mean sentiment, sentiment 3d/7d moving averages, sentiment
momentum (MA3−MA7), bullish ratio (+7d MA), post volume and its 30d
z-score, price-volume 30d z-score, day-of-week, and the coverage flag.
Features for day t use only data dated ≤ t.

## 4. Model & validation design
- One `RandomForestRegressor` (300 trees, depth 6) per horizon.
- **Walk-forward, expanding window:** initial train
  90 days, test blocks of 7
  days, refit every block. A training row is used for horizon h only if
  its full target window closed before the test block — no lookahead.
- Baselines: zero-return forecast (price unchanged) and directional
  persistence (sign of the last 1-day return).

## 5. Out-of-sample results per horizon
| Horizon | OOS n | RMSE | MAE | Naive RMSE | Naive MAE | Dir. acc. | Baseline dir. acc. |
|---|---|---|---|---|---|---|---|
| n+0 | 75 | 0.0234 | 0.0152 | 0.0220 | 0.0143 | 53.3% | 44.0% |
| n+1 | 74 | 0.0363 | 0.0244 | 0.0319 | 0.0207 | 43.2% | 43.2% |
| n+2 | 73 | 0.0470 | 0.0319 | 0.0392 | 0.0252 | 37.0% | 47.9% |
| n+3 | 72 | 0.0509 | 0.0354 | 0.0423 | 0.0284 | 45.8% | 45.8% |
| n+4 | 71 | 0.0564 | 0.0387 | 0.0460 | 0.0304 | 43.7% | 54.9% |
| n+5 | 70 | 0.0635 | 0.0430 | 0.0494 | 0.0330 | 42.9% | 55.7% |

RMSE/MAE are on returns (e.g. 0.03 ≈ 3%). "Dir. acc." is the share of
non-flat days where the predicted return's sign matched the actual.

## 6. Trading evaluation (out-of-sample, h=0 rule)
Rule: long BTC for the next day if the n+0 model predicts a positive
return, otherwise flat. No leverage, no shorting, transaction costs not
modelled. Annualisation uses 365 days/year (BTC trades daily).
Period: 2026-08-18 to 2026-10-31 (75
days, long on 45 of them, win rate on long days
53.3%).

| Metric | Strategy | Buy & hold (same days) |
|---|---|---|
| Sharpe Ratio | 0.51 | 2.32 |
| Sortino Ratio | 0.65 | 4.64 |
| Max Drawdown | -6.8% | -10.9% |
| Total return | 1.8% | 20.0% |

Charts: `outputs/chart_price_vs_sentiment.png`,
`outputs/chart_pred_vs_actual_h0.png`,
`outputs/chart_equity_drawdown.png`.

## 7. Current forecasts (refit on all data)
Origin close 2026-11-01 =
$77,646. Predicted cumulative returns:
n+0: -0.07%, n+1: -0.40%, n+2: -0.84%, n+3: -2.36%, n+4: -1.09%, n+5: +0.20%.

## 8. Limitations (read before judging the numbers)
- Social-sentiment history is shallow in the executed run: the public
  StockTwits stream reaches back only days, so most of the 180 training
  days carry the neutral/flagged sentiment treatment. The Apify 180-day
  X backfill is the intended fix and is wired into the same features.
- One asset, one model family, ~6 months of daily data: results are
  indicative, not production-grade. Crypto returns are close to a random
  walk at these horizons — weak metrics are reported as computed.
- Strategy ignores fees/slippage and uses a single in/out rule.
- News is summarised for the human reader only; it never enters features.

## 9. How to reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your own free-tier tokens
python run_pipeline.py
```
Tokens/actors used by design: Apify token (env `APIFY_API_TOKEN`) with
actor `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`; Tavily key (env `TAVILY_API_KEY`) for
news; Exa key (env `EXA_API_KEY`) reserved as the alternative search
source named in the assignment. Token values are the candidate's own
free-tier credentials, shared at submission time as the assignment
requests, and are never committed to this repository (`.env` is
git-ignored).
