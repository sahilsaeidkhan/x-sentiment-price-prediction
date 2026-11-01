"""Charts saved to outputs/ as PNGs (matplotlib only)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def chart_price_sentiment(df: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["date"], df["close"], color="#1f77b4", lw=1.4,
             label="BTC close (USD)")
    ax1.set_ylabel("Close (USD)")
    ax1.set_xlabel("Date")
    ax2 = ax1.twinx()
    obs = df[df["sent_observed"] == 1]
    ax2.bar(obs["date"], obs["sent_mean"], width=2.0, color="#ff7f0e",
            alpha=0.55, label="Daily mean sentiment (observed days)")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("Mean daily sentiment (-1..+1)")
    ax1.set_title("BTC price vs daily social sentiment (180 days)")
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "chart_price_vs_sentiment.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def chart_pred_vs_actual(preds: pd.DataFrame, horizon: int = 0) -> str:
    sub = (preds[preds["horizon"] == horizon].dropna(subset=["y_true"])
           .sort_values("date"))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub["date"], sub["y_true"] * 100, lw=1.2, label="Actual return %")
    ax.plot(sub["date"], sub["y_pred"] * 100, lw=1.2, alpha=0.8,
            label="Predicted return %")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Next-day return (%)")
    ax.set_xlabel("Date (out-of-sample)")
    ax.set_title(f"Walk-forward predictions vs actual — horizon n+{horizon}")
    ax.legend()
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "chart_pred_vs_actual_h0.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def chart_equity_drawdown(strat: dict) -> str:
    s = strat["_series"]
    dates = pd.to_datetime(s["date"])
    eq_strat = (1 + s["strat_ret"]).cumprod()
    eq_bh = (1 + s["bh_ret"]).cumprod()
    dd = eq_strat / eq_strat.cummax() - 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(dates, eq_strat, lw=1.4, label="Strategy (long/flat, h=0 model)")
    ax1.plot(dates, eq_bh, lw=1.2, alpha=0.8, label="Buy & hold (same period)")
    ax1.set_ylabel("Equity (start = 1.0)")
    ax1.set_title("Out-of-sample strategy equity curve and drawdown")
    ax1.legend()
    ax2.fill_between(dates, dd * 100, 0, color="#d62728", alpha=0.5)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date (out-of-sample)")
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "chart_equity_drawdown.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)
