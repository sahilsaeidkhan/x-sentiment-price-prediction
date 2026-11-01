"""Historical price ingestion for BTC-USD daily OHLCV (last 180 days).

Primary source: Yahoo Finance via yfinance.
Fallback source (used automatically if Yahoo is unreachable): the
Kraken public OHLC API — real exchange daily candles, no key required.
Whichever source actually served the data is recorded in
``df.attrs["source"]`` and printed, and is named in the evaluation
report. Nothing is simulated either way.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

from . import config


def _from_yfinance() -> pd.DataFrame:
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=config.LOOKBACK_DAYS + 2)
    raw = yf.download(
        config.TICKER, start=start.isoformat(), end=end.isoformat(),
        auto_adjust=False, progress=False, timeout=30,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no rows")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "open", "high", "low", "close", "volume"]]


def _from_kraken() -> pd.DataFrame:
    resp = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": "XBTUSD", "interval": 1440}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken error: {payload['error']}")
    key = [k for k in payload["result"] if k != "last"][0]
    rows = payload["result"][key]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "vwap", "volume", "count"])
    df["date"] = (pd.to_datetime(df["ts"], unit="s", utc=True)
                  .dt.tz_localize(None).dt.normalize())
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close", "volume"]]


def fetch_price() -> pd.DataFrame:
    source = "Yahoo Finance (yfinance), BTC-USD"
    try:
        df = _from_yfinance()
    except Exception as exc:
        print(f"[price] yfinance failed ({exc}); falling back to Kraken "
              f"public OHLC API.")
        df = _from_kraken()
        source = "Kraken public OHLC API, XBT/USD daily candles"
    df = (df.dropna(subset=["close"]).sort_values("date")
            .tail(config.LOOKBACK_DAYS).reset_index(drop=True))
    df.attrs["source"] = source
    out = config.DATA_DIR / "price_daily.csv"
    df.to_csv(out, index=False)
    print(f"[price] {len(df)} rows  {df['date'].iloc[0].date()} -> "
          f"{df['date'].iloc[-1].date()}  source: {source}  saved to {out}")
    return df


if __name__ == "__main__":
    fetch_price()
