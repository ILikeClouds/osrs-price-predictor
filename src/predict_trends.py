"""
predict_trends.py
─────────────────
Generates the 7-day price forecast using whichever model tier is appropriate:

  Global model    — when the item has < MIN_DAYS_FOR_ITEM_MODEL regime days
  Item-specific   — otherwise

Writes the forecast to docs/predictions/{item_id}.json for GitHub Pages,
and saves a chart to output/price_trend_forecast.png.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib
matplotlib.use('Agg')   # non-interactive backend safe for CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestRegressor

MIN_DAYS_FOR_ITEM_MODEL = 45

RELATIVE_FEATURES = [
    'pct_change_1d', 'pct_change_3d', 'pct_change_7d', 'pct_change_14d',
    'pct_vs_7ma', 'pct_vs_30ma', 'pct_vs_90ma',
    'ema_crossover', 'slope_7d_pct', 'slope_30d_pct',
    'vol_pct_7d', 'spread_pct', 'volume_ratio',
    'rsi_14', 'range_position_7d',
    'lag_return_1d', 'lag_return_2d', 'lag_return_3d',
    'lag_return_7d', 'lag_return_14d',
]

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, 'data')
    models_dir   = os.path.join(project_root, 'models')
    output_dir   = os.path.join(project_root, 'output')
    docs_dir     = os.path.join(project_root, 'docs', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(docs_dir,   exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    config_path = os.path.join(project_root, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.json not found. Run preprocess.py first.")
    with open(config_path) as f:
        config = json.load(f)

    item_id          = config.get('item_id',          12934)
    item_name        = config.get('item_name',        "Zulrah's scales")
    regime_start     = config.get('regime_start',     None)
    regime_days      = config.get('regime_days',      0)
    use_global_model = config.get('use_global_model', True)

    META_KEYS = {'item_id', 'item_name', 'regime_start', 'regime_days', 'use_global_model'}
    model_params = {k: v for k, v in config.items() if k not in META_KEYS}
    model_params.pop('random_state', None)
    if not model_params:
        model_params = {'n_estimators': 200}

    print(f"Item       : {item_name} (ID: {item_id})")
    print(f"Regime     : from {regime_start} ({regime_days} days)")

    # ── Load data ──────────────────────────────────────────────────────────
    train_abs  = pd.read_csv(os.path.join(data_dir, 'preprocessed_data.csv'))
    future_abs = pd.read_csv(os.path.join(data_dir, 'future_inference_data.csv'))

    future_dates  = pd.to_datetime(future_abs['date']) + pd.Timedelta(days=7)
    anchor_prices = future_abs['daily_avg_price_raw'].values

    # ── Tier 1 — Global model ──────────────────────────────────────────────
    # Only attempt if the global model exists AND the relative feature CSV exists
    rel_csv_path     = os.path.join(data_dir, 'future_inference_rel.csv')
    bundle_path      = os.path.join(models_dir, 'global_model.pkl')
    global_available = os.path.exists(bundle_path) and os.path.exists(rel_csv_path)

    if use_global_model and global_available:
        tier_label = "Global"
        print(f"Model tier : {tier_label} (< {MIN_DAYS_FOR_ITEM_MODEL} regime days)")
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        print(f"  Trained on {bundle['items_trained_on']} items, "
              f"{bundle['rows_trained_on']:,} rows")

        future_rel = pd.read_csv(rel_csv_path)
        X_future   = bundle['scaler'].transform(future_rel[RELATIVE_FEATURES])
        pred_pcts  = bundle['model'].predict(X_future)
        # Global model predicts % return; convert back to absolute price
        future_predictions = np.round(anchor_prices * (1 + pred_pcts)).astype(int)

    # ── Tier 2 — Item-specific model ───────────────────────────────────────
    else:
        if use_global_model and not global_available:
            print("⚠  Global model or relative features not available — "
                  "falling back to item-specific model.")
        tier_label = "Item-specific"
        print(f"Model tier : {tier_label}")

        X_train = train_abs.drop(columns=['target_price_7d', 'date', 'daily_avg_price_raw'])
        y_train = train_abs['target_price_7d']
        model   = RandomForestRegressor(**model_params, random_state=42)
        model.fit(X_train, y_train)

        X_future = future_abs.drop(columns=['target_price_7d', 'date', 'daily_avg_price_raw'])
        # Item-specific model predicts absolute price directly — no anchor needed
        future_predictions = np.round(model.predict(X_future)).astype(int)

    # ── Print forecast ─────────────────────────────────────────────────────
    print(f"\n=== 7-DAY FORECAST: {item_name.upper()} [{tier_label} model] ===")
    for d, p in zip(future_dates, future_predictions):
        print(f"  {d.strftime('%Y-%m-%d')}: {p:,} GP")
    print("=" * (40 + len(item_name)) + "\n")

    # ── Fetch raw actuals for chart ────────────────────────────────────────
    headers = {'User-Agent': 'osrs-price-predictor', 'Accept': 'application/json'}
    raw     = pd.DataFrame(
        requests.get(
            f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=24h&id={item_id}",
            headers=headers
        ).json()['data']
    )
    raw = raw.dropna(subset=['avgHighPrice', 'avgLowPrice']).copy()
    raw['date']      = pd.to_datetime(raw['timestamp'], unit='s')
    raw['raw_price'] = np.round((raw['avgHighPrice'] + raw['avgLowPrice']) / 2).astype(int)

    recent = (
        raw[raw['date'] >= pd.to_datetime(regime_start)].copy()
        if regime_start else raw.tail(100).copy()
    )

    last_date  = recent['date'].iloc[-1]
    last_price = int(recent['raw_price'].iloc[-1])
    pred_dates  = [last_date] + future_dates.tolist()
    pred_prices = [last_price] + future_predictions.tolist()

    # ── Write prediction JSON for GitHub Pages ─────────────────────────────
    now_utc     = datetime.now(timezone.utc)
    stale_after = (now_utc + timedelta(hours=29)).strftime('%Y-%m-%dT%H:%M:%SZ')

    prediction_doc = {
        "item_id":       item_id,
        "item_name":     item_name,
        "generated_at":  now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "stale_after":   stale_after,
        "regime_start":  regime_start,
        "regime_days":   regime_days,
        "model_tier":    tier_label,
        "current_price": last_price,
        "forecast": [
            {
                "date":            d.strftime('%Y-%m-%d'),
                "predicted_price": int(p),
                "change_from_now": int(p) - last_price,
            }
            for d, p in zip(future_dates, future_predictions)
        ],
    }

    json_path = os.path.join(docs_dir, f"{item_id}.json")
    with open(json_path, 'w') as f:
        json.dump(prediction_doc, f, indent=2)
    print(f"Prediction JSON saved → {json_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(recent['date'], recent['raw_price'],
            label='Actual Price', color='black', linewidth=2)
    ax.plot(pred_dates, pred_prices,
            label=f'7-Day Forecast ({tier_label})', color='blue',
            linestyle='--', linewidth=2, marker='o')

    if regime_start:
        ax.axvline(pd.to_datetime(regime_start), color='red', linestyle=':',
                   linewidth=1.5, alpha=0.7, label=f'Regime Start ({regime_start})')

    ax.axvspan(last_date, pred_dates[-1], alpha=0.05, color='blue')

    if tier_label == "Global":
        ax.annotate(
            f"⚠ Cold start: only {regime_days} days of item data.\n"
            f"  Global model active until {MIN_DAYS_FOR_ITEM_MODEL} days reached.",
            xy=(0.02, 0.97), xycoords='axes fraction',
            fontsize=8, color='darkorange', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8)
        )

    ax.set_title(f"OSRS Grand Exchange Forecast: {item_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (GP)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    trend_path = os.path.join(output_dir, 'price_trend_forecast.png')
    plt.savefig(trend_path)
    print(f"Chart saved → {trend_path}")
