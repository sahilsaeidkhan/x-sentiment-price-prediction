"""Live social-sentiment ingestion from the StockTwits public API.

No API key is required. Messages on the BTC.X stream carry the author's own
Bullish/Bearish label (entities.sentiment.basic) plus the message text, so
this gives real, user-labelled social sentiment with timestamps.

Why this source for the executed run: the assignment brief points to Apify
for X scraping (implemented in src/apify_ingest.py, runs when an Apify token
is provided), and its own "historical tweets" document link is broken
("Sorry, the file you have requested does not exist"). StockTwits is used
here as the transparent, working live-sentiment fallback. The public stream
only reaches back a limited window, so daily coverage is dense for the most
recent days only — this limitation is stated in the evaluation report.
"""
import time

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config

API = "https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"


def _label_score(msg, vader):
    """Return (label, score): label from the author tag when present,
    score = +1/-1 for labelled posts, otherwise VADER compound of the text."""
    ent = (msg.get("entities") or {}).get("sentiment") or {}
    basic = ent.get("basic")
    if basic == "Bullish":
        return "Bullish", 1.0
    if basic == "Bearish":
        return "Bearish", -1.0
    return None, float(vader.polarity_scores(msg.get("body", ""))["compound"])


def fetch_stocktwits(max_pages: int = None) -> pd.DataFrame:
    max_pages = max_pages or config.STOCKTWITS_MAX_PAGES
    vader = SentimentIntensityAnalyzer()
    rows, max_id = [], None
    for page in range(max_pages):
        params = {"max": max_id} if max_id else {}
        resp = requests.get(API.format(sym=config.STOCKTWITS_SYMBOL),
                            params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        msgs = payload.get("messages", [])
        if not msgs:
            break
        for m in msgs:
            label, score = _label_score(m, vader)
            user = m.get("user") or {}
            likes = (m.get("likes") or {}).get("total", 0)
            rows.append({
                "id": m.get("id"),
                "created_at": m.get("created_at"),
                "body": m.get("body", ""),
                "label": label,
                "sentiment_score": score,
                "likes": likes or 0,
                "followers": user.get("followers", 0),
            })
        cursor = payload.get("cursor") or {}
        max_id = cursor.get("max")
        if not cursor.get("more") or max_id is None:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    out = config.DATA_DIR / "stocktwits_messages.csv"
    df.to_csv(out, index=False)
    labelled = df["label"].notna().sum()
    print(f"[stocktwits] {len(df)} messages ({labelled} author-labelled)  "
          f"{df['created_at'].iloc[0]} -> {df['created_at'].iloc[-1]}  "
          f"saved to {out}")
    return df


if __name__ == "__main__":
    fetch_stocktwits()
