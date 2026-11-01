# Bitcoin (BTC) — Current Sentiment Outlook
_Generated 2026-11-01 from live data pulled by the pipeline._

## Snapshot
- Latest close in the dataset: **$77,646** on 2026-11-01 (BTC-USD, Yahoo Finance).
- Social sentiment over the most recent observed days (3 of the last 7 days have social data): mean sentiment **+0.23** on a -1..+1 scale, bullish share **52%**, average **974 posts/day** — the crowd is **broadly positive**.
- StockTwits stream captured for this run: 1500 messages (2026-10-30 20:44:47+00:00 to 2026-11-01 19:50:40+00:00, UTC), of which author-labelled: {'Bullish': 607, 'Bearish': 237}.
- X (Apify) posts captured: 8298 (2026-05-11 13:19:13+00:00 to 2026-11-01 20:05:30+00:00, UTC).

## What the news flow is saying
- (Tavily) Bitcoin Coin Price, Charts & Market Insights
- (Tavily) Bitcoin (BTC) Price, Real-time Quote & News - Google Finance
- (Tavily) BITCOIN: INTO THE WEEKEND ANALYSIS!!! #BTC Price Prediction & Crypto Crash News Today
- (Tavily) Bitcoin (BTC) Price Eyes $84000 as Hidden Bullish ...
- (Tavily) Bitcoin dips below $80,000, liquidating $356 million in crypto longs
- (Tavily) orbix Official - 📉 BTC Falls Below $80,000
Last Price:... | Facebook

Headlines above are the actual items retrieved for this summary (Tavily when a key is configured, else CoinTelegraph / CoinDesk / Decrypt RSS). They are context only — the model itself uses sentiment + price features exclusively.

## What the model currently predicts
Forecast origin: close of 2026-11-01 ($77,646). Horizon h is the cumulative return to close of day t+h+1 (h=0 = next day).

| Horizon | Predicted return | Implied price |
|---|---|---|
| n+0 | -0.07% | $77,589 |
| n+1 | -0.40% | $77,332 |
| n+2 | -0.84% | $76,995 |
| n+3 | -2.36% | $75,814 |
| n+4 | -1.09% | $76,796 |
| n+5 | +0.20% | $77,804 |

## Honest caveats
- The executed sentiment run is based on the StockTwits live stream (plus Apify X data when a token is configured). The public stream only covers recent days densely, so the sentiment signal is fresh but shallow in history.
- These are statistical forecasts from a deliberately simple model on a hard prediction problem — read the evaluation report's out-of-sample metrics before trusting any number here. This is not financial advice.
