"""Dataset build: join daily sentiment with daily price bars in SQLite,
then engineer features from SENTIMENT + PRICE ONLY (per the assignment).

Daily sentiment aggregation (per source, then combined by post count):
  - sent_mean   : mean per-post sentiment score
                  (StockTwits: author Bullish=+1 / Bearish=-1, else VADER;
                   X/Apify: VADER compound)
  - bull_ratio  : share of posts with score > 0.05
  - post_volume : number of posts that day
  - engagement  : total likes

No-lookahead rule: features for day t use only posts/prices dated <= t.
Days with no social posts get sent features = 0 (neutral) and
sent_observed = 0 so coverage is explicit and auditable.
"""
import sqlite3

import numpy as np
import pandas as pd

from . import config


def _daily_sentiment(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "sent_mean", "bull_ratio",
                                     "post_volume", "engagement", "source"])
    d = df.copy()
    d["date"] = d["created_at"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    g = d.groupby("date").agg(
        sent_mean=("sentiment_score", "mean"),
        bull_ratio=("sentiment_score", lambda s: float((s > 0.05).mean())),
        post_volume=("sentiment_score", "size"),
        engagement=("likes", "sum"),
    ).reset_index()
    g["source"] = source
    return g


def build_dataset(price: pd.DataFrame,
                  stocktwits: pd.DataFrame | None,
                  apify: pd.DataFrame | None) -> pd.DataFrame:
    parts = []
    st = _daily_sentiment(stocktwits, "stocktwits")
    if not st.empty:
        parts.append(st)
    ap = _daily_sentiment(apify, "x_apify")
    if not ap.empty:
        parts.append(ap)
    if parts:
        all_sent = pd.concat(parts, ignore_index=True)
        # Combine sources per day, weighted by post volume.
        def _combine(g):
            w = g["post_volume"]
            return pd.Series({
                "sent_mean": float(np.average(g["sent_mean"], weights=w)),
                "bull_ratio": float(np.average(g["bull_ratio"], weights=w)),
                "post_volume": int(w.sum()),
                "engagement": int(g["engagement"].sum()),
            })
        daily = (all_sent.groupby("date").apply(_combine, include_groups=False)
                 .reset_index())
    else:
        daily = pd.DataFrame(columns=["date", "sent_mean", "bull_ratio",
                                      "post_volume", "engagement"])

    df = price.merge(daily, on="date", how="left")
    df["sent_observed"] = df["post_volume"].notna().astype(int)
    for c in ["sent_mean", "bull_ratio"]:
        df[c] = df[c].fillna(0.0)
    df["post_volume"] = df["post_volume"].fillna(0).astype(int)
    df["engagement"] = df["engagement"].fillna(0).astype(int)

    # --- persist the join in SQLite (the assignment's "database") -----------
    db = config.DATA_DIR / "dataset.db"
    with sqlite3.connect(db) as con:
        price.to_sql("price_daily", con, if_exists="replace", index=False)
        daily.to_sql("sentiment_daily", con, if_exists="replace", index=False)
        df.to_sql("dataset_joined", con, if_exists="replace", index=False)
    print(f"[dataset] sqlite join saved to {db} "
          f"(sentiment observed on {int(df['sent_observed'].sum())}/{len(df)} days)")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features from sentiment + price only; targets are future returns.

    Horizon convention (stated in README + report): the forecast origin is
    the close of day t; horizon h in {0..5} targets the cumulative return
    from close(t) to close(t+h+1), so h=0 is the next-day return and h=5
    is the 6-day-ahead cumulative return. Features at t never use data
    after t, and walk-forward fitting only uses rows whose target window
    has fully closed before the test block starts.
    """
    d = df.copy().sort_values("date").reset_index(drop=True)
    close = d["close"]
    ret1 = close.pct_change()
    for k in (1, 3, 5, 7):
        d[f"ret_{k}d"] = close.pct_change(k)
    d["vol_7d"] = ret1.rolling(7).std()
    d["vol_14d"] = ret1.rolling(14).std()
    d["sent_ma3"] = d["sent_mean"].rolling(3).mean()
    d["sent_ma7"] = d["sent_mean"].rolling(7).mean()
    d["sent_momentum"] = d["sent_ma3"] - d["sent_ma7"]
    d["bull_ma7"] = d["bull_ratio"].rolling(7).mean()
    def _z30(s: pd.Series) -> pd.Series:
        # Warm-up rows and zero-variance windows (common for post volume,
        # which is 0 on days with no social data) get a neutral z of 0.
        z = (s - s.rolling(30).mean()) / s.rolling(30).std()
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pv = d["post_volume"].astype(float)
    d["postvol_z30"] = _z30(pv)
    lv = np.log1p(d["volume"].astype(float))
    d["pricevol_z30"] = _z30(lv)
    d["dow"] = d["date"].dt.dayofweek
    for h in config.HORIZONS:
        d[f"target_h{h}"] = close.shift(-(h + 1)) / close - 1.0
    return d


FEATURE_COLUMNS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_7d", "vol_7d", "vol_14d",
    "sent_mean", "sent_ma3", "sent_ma7", "sent_momentum",
    "bull_ratio", "bull_ma7", "post_volume", "postvol_z30",
    "pricevol_z30", "dow", "sent_observed",
]
