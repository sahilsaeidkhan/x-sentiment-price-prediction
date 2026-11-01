"""Walk-forward validation + trading evaluation.

Design (rigorous, no lookahead):
  - Features for day t use only data dated <= t.
  - Expanding-window walk-forward: initial train = first
    INITIAL_TRAIN_DAYS feature rows, then test blocks of WF_STEP_DAYS,
    refitting at every step.
  - For horizon h, a training row is used only if its whole target window
    (t .. t+h+1) closed before the test block starts.
  - Model: RandomForestRegressor (scikit-learn), one model per horizon.
  - Baselines: (a) zero-return ("price unchanged") forecast;
               (b) directional persistence — sign of the last 1-day return.
Strategy rule (evaluated on the out-of-sample period only):
  long BTC for the next day when the h=0 model predicts a positive
  next-day return, flat otherwise. No leverage, no shorting, no costs
  modelled (stated as a limitation). Sharpe/Sortino annualised with
  sqrt(365) because BTC trades daily.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config
from .features import FEATURE_COLUMNS


def _make_model():
    return RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        n_jobs=-1, random_state=42)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def walk_forward(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-horizon metrics DataFrame, OOS predictions DataFrame)."""
    feat = d.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    n = len(feat)
    pred_rows = []
    start = config.INITIAL_TRAIN_DAYS
    while start < n:
        block_end = min(start + config.WF_STEP_DAYS, n)
        block_idx = feat.index[start:block_end]
        block_start_date_pos = start
        for h in config.HORIZONS:
            tgt = f"target_h{h}"
            # A row is trainable for horizon h only if its target date
            # (row position + h + 1) is strictly before the block start.
            train_mask = (feat.index + h + 1) < block_start_date_pos
            train = feat.loc[train_mask].dropna(subset=[tgt])
            if len(train) < 30:
                continue
            model = _make_model()
            model.fit(train[FEATURE_COLUMNS], train[tgt])
            block = feat.loc[block_idx]
            preds = model.predict(block[FEATURE_COLUMNS])
            for idx, p in zip(block_idx, preds):
                pred_rows.append({
                    "date": feat.at[idx, "date"], "horizon": h,
                    "y_true": feat.at[idx, tgt], "y_pred": float(p),
                    "naive_pred": 0.0,
                    "persist_sign": np.sign(feat.at[idx, "ret_1d"]),
                })
        start = block_end

    preds = pd.DataFrame(pred_rows)
    metrics = []
    for h in config.HORIZONS:
        sub = preds[preds["horizon"] == h].dropna(subset=["y_true"])
        if sub.empty:
            continue
        y, p = sub["y_true"].values, sub["y_pred"].values
        dir_acc = float((np.sign(p) == np.sign(y))[y != 0].mean())
        base_dir = float((sub.loc[sub["y_true"] != 0, "persist_sign"]
                          == np.sign(sub.loc[sub["y_true"] != 0, "y_true"])
                          ).mean())
        metrics.append({
            "horizon": h, "n_oos": len(sub),
            "rmse": _rmse(y, p), "mae": float(mean_absolute_error(y, p)),
            "naive_rmse": _rmse(y, np.zeros_like(y)),
            "naive_mae": float(mean_absolute_error(y, np.zeros_like(y))),
            "directional_accuracy": dir_acc,
            "baseline_directional_accuracy": base_dir,
        })
    return pd.DataFrame(metrics), preds


def strategy_metrics(preds: pd.DataFrame, price: pd.DataFrame) -> dict:
    """Long/flat strategy from h=0 predictions vs buy-and-hold, same days."""
    h0 = (preds[preds["horizon"] == 0][["date", "y_pred", "y_true"]]
          .dropna().sort_values("date").reset_index(drop=True))
    if h0.empty:
        return {}
    position = (h0["y_pred"] > 0).astype(float)
    strat_ret = position * h0["y_true"]
    bh_ret = h0["y_true"]  # y_true for h=0 IS the next-day return

    def _stats(r: pd.Series) -> dict:
        r = r.astype(float)
        mu, sd = r.mean(), r.std(ddof=1)
        downside = r[r < 0].std(ddof=1)
        sharpe = mu / sd * np.sqrt(config.TRADING_DAYS_PER_YEAR) if sd else 0.0
        sortino = (mu / downside * np.sqrt(config.TRADING_DAYS_PER_YEAR)
                   if downside and not np.isnan(downside) else 0.0)
        equity = (1 + r).cumprod()
        max_dd = float((equity / equity.cummax() - 1).min())
        return {"sharpe": float(sharpe), "sortino": float(sortino),
                "max_drawdown": max_dd,
                "total_return": float(equity.iloc[-1] - 1)}

    s = _stats(strat_ret)
    b = _stats(bh_ret)
    return {
        "oos_days": int(len(h0)),
        "oos_start": str(h0["date"].iloc[0].date()),
        "oos_end": str(h0["date"].iloc[-1].date()),
        "long_days": int(position.sum()),
        "win_rate_long_days": float(
            h0.loc[position == 1, "y_true"].gt(0).mean()
            if (position == 1).any() else 0.0),
        "strategy": s, "buy_hold": b,
        "_series": {"date": h0["date"], "strat_ret": strat_ret,
                    "bh_ret": bh_ret},
    }


def final_forecasts(d: pd.DataFrame) -> pd.DataFrame:
    """Refit per-horizon models on ALL rows with known targets and predict
    from the most recent feature row (for the sentiment summary)."""
    feat = d.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    last = feat.iloc[[-1]]
    rows = []
    for h in config.HORIZONS:
        tgt = f"target_h{h}"
        train = feat.dropna(subset=[tgt])
        model = _make_model()
        model.fit(train[FEATURE_COLUMNS], train[tgt])
        pred_ret = float(model.predict(last[FEATURE_COLUMNS])[0])
        last_close = float(last["close"].iloc[0])
        rows.append({
            "horizon": h,
            "predicted_return": pred_ret,
            "implied_price": last_close * (1 + pred_ret),
            "origin_date": str(last["date"].iloc[0].date()),
            "origin_close": last_close,
        })
    return pd.DataFrame(rows)
