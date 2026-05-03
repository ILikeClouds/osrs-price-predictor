"""
train_model.py
──────────────
Two-tier model selection:

  Tier 1 — Global model   (< MIN_DAYS_FOR_ITEM_MODEL regime days)
    Uses the pre-trained global_model.pkl which was trained on relative
    features from ALL tracked items.  Provides reasonable cold-start accuracy
    for brand-new items immediately.

  Tier 2 — Item-specific model  (≥ MIN_DAYS_FOR_ITEM_MODEL regime days)
    Trains a Random Forest on the item's own regime data using absolute
    features.  More accurate for established items in a stable regime.

Evaluation uses Walk-Forward Validation in both cases so reported metrics
reflect real-world forecasting performance rather than in-sample fit.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import csv
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Keep in sync with preprocess.py
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


def walk_forward_eval(X, y, anchor_prices, params, warmup_frac=0.6):
    """Simulate day-by-day forecasting; return (actuals_abs, preds_abs)."""
    WARMUP = int(len(X) * warmup_frac)
    actuals, preds = [], []
    for i in range(WARMUP, len(X)):
        m = RandomForestRegressor(**params, random_state=42)
        m.fit(X.iloc[:i], y.iloc[:i])
        pred_delta  = m.predict(X.iloc[[i]])[0]
        pred_abs    = pred_delta + anchor_prices[i]
        actual_abs  = y.iloc[i]  + anchor_prices[i]
        preds.append(pred_abs)
        actuals.append(actual_abs)
    return np.array(actuals), np.array(preds)


def walk_forward_eval_global(bundle, X_rel, anchor_prices, y_pct, warmup_frac=0.6):
    """
    Walk-forward evaluation using the global model.
    The global model is FIXED (not retrained per step) — we simply evaluate
    its predictions across the walk-forward window.
    """
    WARMUP  = int(len(X_rel) * warmup_frac)
    model   = bundle['model']
    scaler  = bundle['scaler']
    actuals, preds = [], []
    for i in range(WARMUP, len(X_rel)):
        X_scaled   = scaler.transform(X_rel.iloc[[i]])
        pred_pct   = model.predict(X_scaled)[0]
        pred_abs   = anchor_prices[i] * (1 + pred_pct)
        actual_abs = anchor_prices[i] * (1 + y_pct.iloc[i])
        preds.append(pred_abs)
        actuals.append(actual_abs)
    return np.array(actuals), np.array(preds)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, 'data')
    models_dir   = os.path.join(project_root, 'models')
    output_dir   = os.path.join(project_root, 'output')
    config_path  = os.path.join(project_root, 'config.json')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    META_KEYS = {'item_id', 'item_name', 'regime_start', 'regime_days',
                 'use_global_model'}
    params = {k: v for k, v in config.items() if k not in META_KEYS}
    params.pop('random_state', None)
    if not params:
        params = {'n_estimators': 200}

    item_name        = config.get('item_name', 'Unknown')
    regime_days      = config.get('regime_days', 0)
    use_global_model = config.get('use_global_model', True)

    print(f"Item         : {item_name}")
    print(f"Regime days  : {regime_days}")
    print(f"Model tier   : {'Global (cold start)' if use_global_model else 'Item-specific'}")
    print(f"Params       : {params}")

    # ── Tier 1 — Global model ──────────────────────────────────────────────
    if use_global_model:
        bundle_path = os.path.join(models_dir, 'global_model.pkl')
        if not os.path.exists(bundle_path):
            print(
                "\n⚠  Global model not found. Run build_global_model.py first,\n"
                "   or track more items to build one. Falling back to item-specific\n"
                "   model with limited data in the meantime.\n"
            )
            use_global_model = False   # fall through to tier 2
        else:
            with open(bundle_path, 'rb') as f:
                bundle = pickle.load(f)
            print(f"\nGlobal model loaded  "
                  f"({bundle['items_trained_on']} items, "
                  f"{bundle['rows_trained_on']:,} rows)")

            df_rel       = pd.read_csv(os.path.join(data_dir, 'preprocessed_data_rel.csv'))
            df_abs       = pd.read_csv(os.path.join(data_dir, 'preprocessed_data.csv'))
            anchor_prices = df_abs['daily_avg_price_raw'].values
            X_rel         = df_rel[RELATIVE_FEATURES]
            y_pct         = df_rel['target_pct_7d']

            print("Running walk-forward evaluation (global model)...")
            actuals, preds = walk_forward_eval_global(
                bundle, X_rel, anchor_prices, y_pct
            )

    # ── Tier 2 — Item-specific model ───────────────────────────────────────
    if not use_global_model:
        df           = pd.read_csv(os.path.join(data_dir, 'preprocessed_data.csv'))
        anchor_prices = df['daily_avg_price_raw'].values
        X = df.drop(columns=['target_price_7d', 'date', 'daily_avg_price_raw'])
        y = df['target_price_7d']

        print(f"\nLoaded tuned parameters: {params}")
        print("Running walk-forward evaluation (item-specific model)...")
        actuals, preds = walk_forward_eval(X, y, anchor_prices, params)

    # ── Metrics ────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)

    tier_label = "Global" if use_global_model else "Item-specific"
    print(f"\n--- Walk-Forward Evaluation ({tier_label} model, {len(actuals)} steps) ---")
    print(f"MAE  : {mae:.2f} GP")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # ── Append to metrics log ──────────────────────────────────────────────
    log_path    = os.path.join(output_dir, 'metrics_log.csv')
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['date', 'item', 'model_tier', 'mae', 'rmse', 'r2',
                        'steps', 'regime_days', 'regime_start'])
        w.writerow([
            date.today(), item_name, tier_label,
            round(mae, 2), round(rmse, 2), round(r2, 4),
            len(actuals), regime_days, config.get('regime_start', ''),
        ])
    print(f"Metrics logged → {log_path}")