"""Central configuration for the X-sentiment price-prediction pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")

# --- Asset -----------------------------------------------------------------
TICKER = "BTC-USD"          # Yahoo Finance symbol for Bitcoin
ASSET_NAME = "Bitcoin (BTC)"
LOOKBACK_DAYS = 180         # assignment scope: last 180 days

# --- API tokens (from .env; never hard-code, never commit) ------------------
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
EXA_API_KEY = os.getenv("EXA_API_KEY", "").strip()

# --- Apify X (Twitter) scraper ----------------------------------------------
# Actor used to scrape X posts about the asset. Runs only when
# APIFY_API_TOKEN is set. Search terms target the BTC cashtag / keywords.
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "apidojo/tweet-scraper").strip()
APIFY_SEARCH_TERMS = ["$BTC", "Bitcoin", "BTCUSD"]
APIFY_MAX_ITEMS = 5000

# --- StockTwits (public, no key) — live-run sentiment source ----------------
STOCKTWITS_SYMBOL = "BTC.X"
STOCKTWITS_MAX_PAGES = 50   # 30 messages per page

# --- Modelling ---------------------------------------------------------------
HORIZONS = [0, 1, 2, 3, 4, 5]   # n+0 .. n+5 (see README for the convention)
INITIAL_TRAIN_DAYS = 90
WF_STEP_DAYS = 7
TRADING_DAYS_PER_YEAR = 365     # BTC trades every day
