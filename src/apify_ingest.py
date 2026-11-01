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
    run_input = {
        "searchTerms": config.APIFY_SEARCH_TERMS,
        "maxItems": config.APIFY_MAX_ITEMS,
        "sort": "Latest",
        "tweetLanguage": "en",
    }
    print(f"[apify] starting actor {config.APIFY_ACTOR_ID} "
          f"terms={config.APIFY_SEARCH_TERMS} maxItems={config.APIFY_MAX_ITEMS}")
    run = client.actor(config.APIFY_ACTOR_ID).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
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
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True,
                                      format="ISO8601")
    df = df.sort_values("created_at").reset_index(drop=True)
    out = config.DATA_DIR / "x_posts_apify.csv"
    df.to_csv(out, index=False)
    print(f"[apify] {len(df)} tweets  {df['created_at'].iloc[0]} -> "
          f"{df['created_at'].iloc[-1]}  saved to {out}")
    return df


if __name__ == "__main__":
    fetch_apify()
