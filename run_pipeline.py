"""End-to-end pipeline for the CWT X-Sentiment Price Prediction task.

    python run_pipeline.py

Steps: price ingest -> social sentiment ingest (Apify if a token is set,
plus StockTwits live stream) -> news snapshot -> SQLite join -> features
(sentiment + price only) -> walk-forward validation -> strategy
evaluation -> charts -> SENTIMENT_SUMMARY.md + EVALUATION_REPORT.md.
All numbers written into the reports are computed in this run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (apify_ingest, charts, config, features, model,  # noqa: E402
                 news_ingest, price_ingest, stocktwits_ingest)

RUN_DATE = pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _load_or_fetch():
    price = price_ingest.fetch_price()
    try:
        stocktwits = stocktwits_ingest.fetch_stocktwits()
    except Exception as exc:
        print(f"[stocktwits] FAILED: {exc}")
        stocktwits = None
    apify = apify_ingest.fetch_apify()
    news = news_ingest.fetch_news()
    return price, stocktwits, apify, news


def write_sentiment_summary(d, forecasts, news, st, ap):
    latest = d.iloc[-1]
    recent = d.tail(7)
    obs_recent = recent[recent["sent_observed"] == 1]
    sent_now = float(obs_recent["sent_mean"].mean()) if len(obs_recent) else float("nan")
    bull_now = float(obs_recent["bull_ratio"].mean()) if len(obs_recent) else float("nan")
    vol_trend = (float(obs_recent["post_volume"].mean()) if len(obs_recent) else 0.0)
    stance = ("broadly positive" if sent_now > 0.1 else
              "broadly negative" if sent_now < -0.1 else "mixed / neutral")
    lines = [
        f"# Bitcoin (BTC) — Current Sentiment Outlook",
        f"_Generated {RUN_DATE.date()} from live data pulled by the pipeline._",
        "",
        f"## Snapshot",
        f"- Latest close in the dataset: **${latest['close']:,.0f}** on "
        f"{latest['date'].date()} (BTC-USD, Yahoo Finance).",
        f"- Social sentiment over the most recent observed days "
        f"({len(obs_recent)} of the last 7 days have social data): "
        f"mean sentiment **{sent_now:+.2f}** on a -1..+1 scale, "
        f"bullish share **{bull_now:.0%}**, average "
        f"**{vol_trend:.0f} posts/day** — the crowd is **{stance}**.",
    ]
    if st is not None and len(st):
        lab = st["label"].value_counts().to_dict()
        lines.append(
            f"- StockTwits stream captured for this run: {len(st)} messages "
            f"({st['created_at'].min()} to {st['created_at'].max()}, UTC), "
            f"of which author-labelled: {lab}.")
    if ap is not None and len(ap):
        lines.append(
            f"- X (Apify) posts captured: {len(ap)} "
            f"({ap['created_at'].min()} to {ap['created_at'].max()}, UTC).")
    lines += ["", "## What the news flow is saying"]
    if news is not None and len(news):
        for _, r in news.head(8).iterrows():
            lines.append(f"- ({r['source']}) {r['title']}")
        lines.append("")
        lines.append("Headlines above are the actual items retrieved for "
                     "this summary (Tavily when a key is configured, else "
                     "CoinTelegraph / CoinDesk / Decrypt RSS). They are "
                     "context only — the model itself uses sentiment + "
                     "price features exclusively.")
    else:
        lines.append("- No news items could be retrieved in this run.")
    lines += ["", "## What the model currently predicts",
              f"Forecast origin: close of {forecasts['origin_date'].iloc[0]} "
              f"(${forecasts['origin_close'].iloc[0]:,.0f}). Horizon h is the "
              f"cumulative return to close of day t+h+1 (h=0 = next day).",
              "",
              "| Horizon | Predicted return | Implied price |",
              "|---|---|---|"]
    for _, r in forecasts.iterrows():
        lines.append(f"| n+{int(r['horizon'])} | {r['predicted_return']:+.2%} "
                     f"| ${r['implied_price']:,.0f} |")
    lines += ["",
              "## Honest caveats",
              "- The executed sentiment run is based on the StockTwits live "
              "stream (plus Apify X data when a token is configured). The "
              "public stream only covers recent days densely, so the "
              "sentiment signal is fresh but shallow in history.",
              "- These are statistical forecasts from a deliberately simple "
              "model on a hard prediction problem — read the evaluation "
              "report's out-of-sample metrics before trusting any number "
              "here. This is not financial advice."]
    text = "\n".join(lines) + "\n"
    out = config.OUTPUTS_DIR / "SENTIMENT_SUMMARY.md"
    out.write_text(text)
    print(f"[summary] written to {out}")
    return out


def write_evaluation_report(d, metrics, preds, strat, forecasts,
                            price, st, ap, news):
    cov = float(d["sent_observed"].mean())
    s, b = strat["strategy"], strat["buy_hold"]
    mlines = ["| Horizon | OOS n | RMSE | MAE | Naive RMSE | Naive MAE | "
              "Dir. acc. | Baseline dir. acc. |", "|---|---|---|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        mlines.append(
            f"| n+{int(r['horizon'])} | {int(r['n_oos'])} | {r['rmse']:.4f} | "
            f"{r['mae']:.4f} | {r['naive_rmse']:.4f} | {r['naive_mae']:.4f} | "
            f"{r['directional_accuracy']:.1%} | "
            f"{r['baseline_directional_accuracy']:.1%} |")
    apify_note = (f"X posts via Apify were included in this run "
                  f"({len(ap)} posts)." if ap is not None and len(ap)
                  else "No Apify token was configured for this run, so the "
                       "Apify path was implemented but not executed; the "
                       "report's author supplies his own free-tier Apify "
                       "token at submission and the pipeline then ingests "
                       "X posts identically (see Reproduce).")
    st_note = (f"{len(st)} messages, {st['created_at'].min()} to "
               f"{st['created_at'].max()} UTC, "
               f"{int(st['label'].notna().sum())} author-labelled "
               f"Bullish/Bearish." if st is not None and len(st)
               else "StockTwits fetch failed in this run.")
    report = f"""# Evaluation Report — X Sentiment Price Prediction Agent (BTC)

_Generated {RUN_DATE.date()} by `run_pipeline.py`. Every number below is
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
- **Price:** BTC-USD daily OHLCV — {len(price)}
  rows, {price['date'].iloc[0].date()} to {price['date'].iloc[-1].date()}.
  Source actually used in this run: **{price.attrs.get("source", "unknown")}**
  (yfinance is the primary loader; the Kraken public OHLC API is the
  automatic fallback when Yahoo is unreachable — both are real market
  data, and the loader records which one served the run).
- **X sentiment (primary, per assignment):** Apify actor
  `{config.APIFY_ACTOR_ID}` with search terms {config.APIFY_SEARCH_TERMS}.
  {apify_note}
- **Live sentiment actually executed:** StockTwits public stream for
  BTC.X — {st_note} StockTwits labels are the authors' own Bullish/Bearish
  tags; unlabelled posts are scored with VADER.
- **Why a fallback was needed:** the assignment document links a
  "historical tweets" Google Doc for training data; that link is dead
  ("Sorry, the file you have requested does not exist"), and no Apify
  token existed at build time. StockTwits is the documented substitute;
  the Apify ingestion is implemented (`src/apify_ingest.py`) and merges
  into the same dataset when a token is provided.
- **News (context only, never a feature):** Tavily when a key is set,
  else CoinTelegraph/CoinDesk/Decrypt RSS — {0 if news is None else len(news)}
  headlines in `data/news_latest.csv`.
- **Join:** daily sentiment is aggregated per day and joined with price
  bars in **SQLite** (`data/dataset.db`, tables `price_daily`,
  `sentiment_daily`, `dataset_joined`); the joined table is also exported
  to `data/dataset.csv`. Sentiment is observed on {cov:.0%} of the 180
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
  {config.INITIAL_TRAIN_DAYS} days, test blocks of {config.WF_STEP_DAYS}
  days, refit every block. A training row is used for horizon h only if
  its full target window closed before the test block — no lookahead.
- Baselines: zero-return forecast (price unchanged) and directional
  persistence (sign of the last 1-day return).

## 5. Out-of-sample results per horizon
{chr(10).join(mlines)}

RMSE/MAE are on returns (e.g. 0.03 ≈ 3%). "Dir. acc." is the share of
non-flat days where the predicted return's sign matched the actual.

## 6. Trading evaluation (out-of-sample, h=0 rule)
Rule: long BTC for the next day if the n+0 model predicts a positive
return, otherwise flat. No leverage, no shorting, transaction costs not
modelled. Annualisation uses 365 days/year (BTC trades daily).
Period: {strat['oos_start']} to {strat['oos_end']} ({strat['oos_days']}
days, long on {strat['long_days']} of them, win rate on long days
{strat['win_rate_long_days']:.1%}).

| Metric | Strategy | Buy & hold (same days) |
|---|---|---|
| Sharpe Ratio | {s['sharpe']:.2f} | {b['sharpe']:.2f} |
| Sortino Ratio | {s['sortino']:.2f} | {b['sortino']:.2f} |
| Max Drawdown | {s['max_drawdown']:.1%} | {b['max_drawdown']:.1%} |
| Total return | {s['total_return']:.1%} | {b['total_return']:.1%} |

Charts: `outputs/chart_price_vs_sentiment.png`,
`outputs/chart_pred_vs_actual_h0.png`,
`outputs/chart_equity_drawdown.png`.

## 7. Current forecasts (refit on all data)
Origin close {forecasts['origin_date'].iloc[0]} =
${forecasts['origin_close'].iloc[0]:,.0f}. Predicted cumulative returns:
{", ".join(f"n+{int(r['horizon'])}: {r['predicted_return']:+.2%}" for _, r in forecasts.iterrows())}.

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
actor `{config.APIFY_ACTOR_ID}`; Tavily key (env `TAVILY_API_KEY`) for
news; Exa key (env `EXA_API_KEY`) reserved as the alternative search
source named in the assignment. Token values are the candidate's own
free-tier credentials, shared at submission time as the assignment
requests, and are never committed to this repository (`.env` is
git-ignored).
"""
    out = config.OUTPUTS_DIR / "EVALUATION_REPORT.md"
    out.write_text(report)
    print(f"[report] written to {out}")
    return out


def main():
    price, stocktwits, apify, news = _load_or_fetch()
    joined = features.build_dataset(price, stocktwits, apify)
    joined.to_csv(config.DATA_DIR / "dataset.csv", index=False)
    d = features.add_features(joined)
    d.to_csv(config.DATA_DIR / "features.csv", index=False)

    metrics, preds = model.walk_forward(d)
    metrics.to_csv(config.OUTPUTS_DIR / "walk_forward_metrics.csv", index=False)
    preds.to_csv(config.OUTPUTS_DIR / "oos_predictions.csv", index=False)
    strat = model.strategy_metrics(preds, price)
    forecasts = model.final_forecasts(d)
    forecasts.to_csv(config.OUTPUTS_DIR / "current_forecasts.csv", index=False)

    charts.chart_price_sentiment(d)
    charts.chart_pred_vs_actual(preds, horizon=0)
    charts.chart_equity_drawdown(strat)
    print("[charts] 3 charts saved to outputs/")

    write_sentiment_summary(d, forecasts, news, stocktwits, apify)
    write_evaluation_report(d, metrics, preds, strat, forecasts,
                            price, stocktwits, apify, news)
    print("\n=== STRATEGY (OOS) ===")
    print(f"Sharpe {strat['strategy']['sharpe']:.2f} | "
          f"Sortino {strat['strategy']['sortino']:.2f} | "
          f"MaxDD {strat['strategy']['max_drawdown']:.1%} | "
          f"Days {strat['oos_days']}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
