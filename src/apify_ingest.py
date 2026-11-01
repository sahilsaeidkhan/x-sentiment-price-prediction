"""X (Twitter) ingestion via Apify — the assignment's primary scraping path.

Runs ONLY when APIFY_API_TOKEN is present in .env. The actor configured in
src/config.py (default: "apidojo/tweet-scraper", overridable via
APIFY_ACTOR_ID) is started with the BTC search terms, and the resulting
tweets are normalised to the same schema the feature builder consumes
(timestamp, text, engagement, author followers) plus a VADER sentiment
score (X posts carry no Bullish/Bearish author label).

Output: data/x_posts_apify.csv  (+ a per-run note in the console).
If no token is configured this module exits gracefully and the pipeline
falls back to the StockTwits live-sentiment source.
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config


def fetch_apify() -> pd.DataFrame | None:
    if not config.APIFY_API_TOKEN:
        print("[apify] APIFY_API_TOKEN not set in .env — skipping Apify run "
              "(pipeline will use the StockTwits sentiment source).")
        return None

    from apify_client import ApifyClient

    client = ApifyClient(config.APIFY_API_TOKEN)
    items = []
    # Windowed backfill over the last LOOKBACK_DAYS: one actor run per
    # APIFY_WINDOW_DAYS window, using X advanced-search since:/until:
    # operators, so tweets are sampled across the full 180 days rather
    # than only the latest days a single "Latest" run would return.
    end = pd.Timestamp.now("UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=config.LOOKBACK_DAYS + 1)
    windows = []
    w = start
    while w < end:
        windows.append((w, min(w + pd.Timedelta(days=config.APIFY_WINDOW_DAYS), end)))
        w += pd.Timedelta(days=config.APIFY_WINDOW_DAYS)
    for w_start, w_end in windows:
        terms = [f"{t} since:{w_start.date()} until:{w_end.date()}"
                 for t in config.APIFY_WINDOW_TERMS]
        run_input = {
            "searchTerms": terms,
            "maxItems": config.APIFY_MAX_ITEMS_PER_WINDOW,
            "sort": "Latest",
            "tweetLanguage": "en",
        }
        print(f"[apify] window {w_start.date()} -> {w_end.date()} "
              f"terms={terms} maxItems={config.APIFY_MAX_ITEMS_PER_WINDOW}")
        try:
            run = client.actor(config.APIFY_ACTOR_ID).call(run_input=run_input)
            ds_id = (getattr(run, "default_dataset_id", None)
                     or run["defaultDatasetId"])
            got = list(client.dataset(ds_id).iterate_items())
            got = [g for g in got if not g.get("noResults")]
        except Exception as exc:
            print(f"[apify] window {w_start.date()} FAILED: {exc}")
            continue
        print(f"[apify] window {w_start.date()}: {len(got)} items")
        items.extend(got)
        # Checkpoint after every window so an interrupted run keeps the
        # tweets already paid for; the final block below re-saves the
        # same normalised frame.
        if items:
            pd.DataFrame(items).to_csv(
                config.DATA_DIR / "x_posts_apify_raw_checkpoint.csv",
                index=False)
    if not items:
        print("[apify] actor returned no items.")
        return None

    vader = SentimentIntensityAnalyzer()
    rows = []
    for it in items:
        text = it.get("fullText") or it.get("text") or ""
        author = it.get("author") or {}
        rows.append({
            "id": it.get("id"),
            "created_at": it.get("createdAt"),
            "body": text,
            "label": None,
            "sentiment_score": float(
                vader.polarity_scores(text)["compound"]) if text else 0.0,
            "likes": it.get("likeCount", 0) or 0,
            "retweets": it.get("retweetCount", 0) or 0,
            "views": it.get("viewCount", 0) or 0,
            "followers": author.get("followers", 0) or 0,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    # Actor timestamps arrive as e.g. "Sun Nov 01 19:36:47 +0000 2026"
    # (X's native format), not ISO8601 — parse flexibly.
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True,
                                      format="mixed")
    df = df.sort_values("created_at").reset_index(drop=True)
    out = config.DATA_DIR / "x_posts_apify.csv"
    df.to_csv(out, index=False)
    print(f"[apify] {len(df)} tweets  {df['created_at'].iloc[0]} -> "
          f"{df['created_at'].iloc[-1]}  saved to {out}")
    return df


if __name__ == "__main__":
    fetch_apify()
