"""
build_global_model.py
─────────────────────
Trains a single Random Forest on the combined relative-feature history of
every item that has ever been tracked. Because all features are scale-free
(% returns, RSI, normalised slopes, etc.) data from Dragon bones at 2,500 GP
and Zulrah's scales at 180 GP is directly comparable.

This script is called by the GitHub Actions workflow after all per-item
pipeline runs are complete. It exits cleanly (no error) when fewer than
2 items have accumulated enough history to be useful — the per-item
predict_trends.py handles those cases independently.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

RELATIVE_FEATURES = [
    'pct_change_1d', 'pct_change_3d', 'pct_change_7d', 'pct_change_14d',
    'pct_vs_7ma', 'pct_vs_30ma', 'pct_vs_90ma',
    'ema_crossover', 'slope_7d_pct', 'slope_30d_pct',
    'vol_pct_7d', 'spread_pct', 'volume_ratio',
    'rsi_14', 'range_position_7d',
    'lag_return_1d', 'lag_return_2d', 'lag_return_3d',
    'lag_return_7d', 'lag_return_14d',
]

MIN_ITEMS   = 2      # need at least 2 items before a shared model is worthwhile
MIN_ROWS    = 60     # need at least 60 combined rows to train meaningfully


def load_all_histories(history_dir: str):
    """
    Concatenate every per-item history CSV into one DataFrame.
    Returns None (not an error) if data is missing or insufficient.
    """
    if not os.path.exists(history_dir):
        print(f"History directory not found: {history_dir}")
        print("Skipping global model — run preprocess.py for at least 2 items first.")
        return None

    frames = []
    for fname in sorted(os.listdir(history_dir)):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(history_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['date'])
            # Only use files that have the relative feature columns
            missing = [c for c in RELATIVE_FEATURES + ['target_pct_7d']
                       if c not in df.columns]
            if missing:
                print(f"  Skipping {fname}: missing columns {missing[:3]}...")
                continue
            frames.append(df)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    if len(frames) < MIN_ITEMS:
        print(f"Only {len(frames)} usable item history file(s) found "
              f"(need ≥ {MIN_ITEMS}). Skipping global model build.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=RELATIVE_FEATURES + ['target_pct_7d'])
    # Remove extreme outlier rows (crash days where price moved > 50% in a week)
    combined = combined[combined['target_pct_7d'].abs() < 0.5]

    if len(combined) < MIN_ROWS:
        print(f"Only {len(combined)} usable rows after filtering "
              f"(need ≥ {MIN_ROWS}). Skipping global model build.")
        return None

    print(f"Loaded {len(combined):,} rows from {len(frames)} item(s).")
    return combined


def train_global_model(combined: pd.DataFrame, params: dict):
    """
    Train the global model. Chronological 80/20 split + 5-fold TimeSeriesSplit CV.
    Returns (model, scaler, test_mae).
    """
    combined = combined.sort_values('date').reset_index(drop=True)

    X = combined[RELATIVE_FEATURES]
    y = combined['target_pct_7d']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test_s))
    print(f"Global model test MAE: {mae:.4f} (≈ {mae * 100:.2f}% per prediction)")

    tscv      = TimeSeriesSplit(n_splits=5)
    fold_maes = []
    for tr_idx, te_idx in tscv.split(X_train_s):
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        fold_maes.append(
            mean_absolute_error(y_train.iloc[te_idx], m.predict(X_train_s[te_idx]))
        )
    print(f"CV fold MAEs : {[round(v, 4) for v in fold_maes]}")
    print(f"CV average   : {np.mean(fold_maes):.4f}")

    return model, scaler, mae


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_dir  = os.path.join(project_root, 'data', 'history')
    models_dir   = os.path.join(project_root, 'models')
    config_path  = os.path.join(project_root, 'config.json')
    os.makedirs(models_dir, exist_ok=True)

    # Load hyperparams from config, fall back to sensible defaults
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    META_KEYS = {'item_id', 'item_name', 'regime_start', 'regime_days',
                 'use_global_model'}
    params = {k: v for k, v in config.items() if k not in META_KEYS}
    params.pop('random_state', None)
    if not params:
        params = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2}
    print(f"Model params: {params}")

    print("\nLoading item histories...")
    combined = load_all_histories(history_dir)

    if combined is None:
        # Not enough data yet — exit cleanly so the workflow continues
        print("Global model not built this run. This is expected on day 1.")
        raise SystemExit(0)

    print("\nTraining global model...")
    model, scaler, test_mae = train_global_model(combined, params)

    bundle = {
        'model':            model,
        'scaler':           scaler,
        'test_mae_pct':     test_mae,
        'features':         RELATIVE_FEATURES,
        'items_trained_on': combined['item_id'].nunique()
                            if 'item_id' in combined.columns else len(
                                [f for f in os.listdir(history_dir)
                                 if f.endswith('.csv')]),
        'rows_trained_on':  len(combined),
    }
    bundle_path = os.path.join(models_dir, 'global_model.pkl')
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)

    print(f"\nGlobal model saved → {bundle_path}")
    print(f"  Items: {bundle['items_trained_on']} | "
          f"Rows: {bundle['rows_trained_on']:,} | "
          f"Test MAE: {test_mae:.4f}")