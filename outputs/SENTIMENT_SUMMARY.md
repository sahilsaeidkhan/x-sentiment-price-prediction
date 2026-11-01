# Bitcoin (BTC) — Current Sentiment Outlook
_Generated 2026-11-01 from live data pulled by the pipeline._

## Snapshot
- Latest close in the dataset: **$77,991** on 2026-11-01 (BTC-USD, Yahoo Finance).
- Social sentiment over the most recent observed days (3 of the last 7 days have social data): mean sentiment **+0.26** on a -1..+1 scale, bullish share **55%**, average **500 posts/day** — the crowd is **broadly positive**.
- StockTwits stream captured for this run: 1500 messages (2026-10-30 19:23:05+00:00 to 2026-11-01 17:17:02+00:00, UTC), of which author-labelled: {'Bullish': 589, 'Bearish': 228}.

## What the news flow is saying
- (Decrypt) Bitcoin ETF Investors Head for the Exits After Record Binance Hack
- (CoinDesk) Binance opens crypto withdrawals in phases starting with bitcoin following $6.3 billion hack
- (CoinDesk) FBI confirms North Korea's role in $6 billion Binance hack, asks DeFi community to block BTC, ETH, SOL addresses
- (CoinDesk) Bitcoin's white paper turns 18, and small businesses are still paying for the bailouts it answered
- (CoinTelegraph) Bitcoin held up better than expected after Binance hack, Strive CEO says
- (CoinTelegraph) Bitcoin turns 18: Has the original rebel grown up?
- (CoinDesk) As big bitcoin miners chase AI profits, individuals get another shot at the rewards
- (CoinDesk) 18 years in, what has Bitcoin truly done?

Headlines above are the actual items retrieved for this summary (Tavily when a key is configured, else CoinTelegraph / CoinDesk / Decrypt RSS). They are context only — the model itself uses sentiment + price features exclusively.

## What the model currently predicts
Forecast origin: close of 2026-11-01 ($77,991). Horizon h is the cumulative return to close of day t+h+1 (h=0 = next day).

| Horizon | Predicted return | Implied price |
|---|---|---|
| n+0 | -0.13% | $77,886 |
| n+1 | +0.27% | $78,201 |
| n+2 | +0.19% | $78,136 |
| n+3 | -0.98% | $77,230 |
| n+4 | -0.41% | $77,674 |
| n+5 | +0.31% | $78,230 |

## Honest caveats
- The executed sentiment run is based on the StockTwits live stream (plus Apify X data when a token is configured). The public stream only covers recent days densely, so the sentiment signal is fresh but shallow in history.
- These are statistical forecasts from a deliberately simple model on a hard prediction problem — read the evaluation report's out-of-sample metrics before trusting any number here. This is not financial advice.
