"""Recent-news ingestion for the chosen asset.

Primary path: Tavily search API (when TAVILY_API_KEY is set in .env).
Fallback path (used for the executed run when no key is configured):
public crypto-news RSS feeds (CoinTelegraph, CoinDesk, Decrypt) parsed
with feedparser. The snapshot feeds the "current sentiment outlook"
summary — it is context only and is NOT used as a model feature
(features are restricted to sentiment + price, per the assignment).
"""
import pandas as pd

from . import config

RSS_FEEDS = [
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt", "https://decrypt.co/feed"),
]
KEYWORDS = ("bitcoin", "btc")


def _fetch_tavily() -> pd.DataFrame | None:
    if not config.TAVILY_API_KEY:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        res = client.search(query="Bitcoin BTC price news", topic="news",
                            days=7, max_results=15)
        rows = [{
            "source": "Tavily",
            "published": r.get("published_date"),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
        } for r in res.get("results", [])]
        return pd.DataFrame(rows) if rows else None
    except Exception as exc:  # fall through to RSS
        print(f"[news] Tavily failed ({exc}); falling back to RSS.")
        return None


def _fetch_rss() -> pd.DataFrame:
    import feedparser
    rows = []
    for source, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as exc:
            print(f"[news] RSS {source} failed: {exc}")
            continue
        for e in feed.entries:
            title = getattr(e, "title", "")
            if not any(k in title.lower() for k in KEYWORDS):
                continue
            rows.append({
                "source": source,
                "published": getattr(e, "published", ""),
                "title": title,
                "url": getattr(e, "link", ""),
            })
    return pd.DataFrame(rows)


def fetch_news() -> pd.DataFrame:
    df = _fetch_tavily()
    if df is None or df.empty:
        df = _fetch_rss()
    if df.empty:
        print("[news] WARNING: no news items fetched.")
        df = pd.DataFrame(columns=["source", "published", "title", "url"])
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df = (df.dropna(subset=["title"]).sort_values("published", ascending=False)
            .head(25).reset_index(drop=True))
    out = config.DATA_DIR / "news_latest.csv"
    df.to_csv(out, index=False)
    print(f"[news] {len(df)} Bitcoin headlines saved to {out} "
          f"(sources: {sorted(df['source'].unique()) if len(df) else 'none'})")
    return df


if __name__ == "__main__":
    fetch_news()
