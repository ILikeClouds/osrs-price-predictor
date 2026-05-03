"""
build_global_model.py
─────────────────────
Trains a single Random Forest on the combined relative-feature history of
every item that has ever been tracked.  Because all features are scale-free
(% returns, RSI, normalised slopes, etc.) data from Dragon bones at 2,500 GP
and Zulrah's scales at 180 GP is directly comparable.

Run this manually after adding several new items, or include it in the
pipeline after tune_model.py.  The saved model is used automatically by
train_model.py and predict_trends.py whenever an item has fewer than
MIN_DAYS_FOR_ITEM_MODEL days of regime-specific data.
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

# Keep in sync with preprocess.py
RELATIVE_FEATURES = [
    'pct_change_1d', 'pct_change_3d', 'pct_change_7d', 'pct_change_14d',
    'pct_vs_7ma', 'pct_vs_30ma', 'pct_vs_90ma',
    'ema_crossover', 'slope_7d_pct', 'slope_30d_pct',
    'vol_pct_7d', 'spread_pct', 'volume_ratio',
    'rsi_14', 'range_position_7d',
    'lag_return_1d', 'lag_return_2d', 'lag_return_3d',
    'lag_return_7d', 'lag_return_14d',
]


def load_all_histories(history_dir: str) -> pd.DataFrame:
    """Concatenate every per-item history CSV into one DataFrame."""
    frames = []
    if not os.path.exists(history_dir):
        raise FileNotFoundError(
            f"History directory not found: {history_dir}\n"
            "Run preprocess.py for at least one item first."
        )
    for fname in os.listdir(history_dir):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(history_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['date'])
            frames.append(df)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    if not frames:
        raise ValueError("No valid history files found. Track at least one item first.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=RELATIVE_FEATURES + ['target_pct_7d'])

    # Remove extreme outlier rows that would skew the model
    # (e.g. crash days where price moved > 50% in a week)
    combined = combined[combined['target_pct_7d'].abs() < 0.5]

    print(f"Loaded {len(combined):,} rows from {len(frames)} item(s).")
    return combined


def train_global_model(
    combined: pd.DataFrame,
    params: dict,
) -> tuple[RandomForestRegressor, StandardScaler, float]:
    """
    Train the global model.  Uses a time-aware group split: sort by date and
    use the last 20% as a held-out test set so we don't evaluate on the past.
    Returns (model, scaler, test_mae_in_pct).
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

    y_pred = model.predict(X_test_s)
    mae    = mean_absolute_error(y_test, y_pred)
    print(f"Global model test MAE: {mae:.4f} (as % of price)")
    print(f"  ≈ {mae * 100:.2f}% per-prediction error")

    # Cross-validate for stability check
    tscv = TimeSeriesSplit(n_splits=5)
    fold_maes = []
    for tr_idx, te_idx in tscv.split(X_train_s):
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        fold_maes.append(mean_absolute_error(
            y_train.iloc[te_idx], m.predict(X_train_s[te_idx])
        ))
    print(f"  CV fold MAEs: {[round(v, 4) for v in fold_maes]}")
    print(f"  CV average : {np.mean(fold_maes):.4f}")

    return model, scaler, mae


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_dir  = os.path.join(project_root, 'data', 'history')
    models_dir   = os.path.join(project_root, 'models')
    config_path  = os.path.join(project_root, 'config.json')
    os.makedirs(models_dir, exist_ok=True)

    # Load global hyperparams from config, fall back to sensible defaults
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

    print("\nTraining global model...")
    model, scaler, test_mae = train_global_model(combined, params)

    # Save model + scaler together as a bundle
    bundle = {'model': model, 'scaler': scaler, 'test_mae_pct': test_mae,
              'features': RELATIVE_FEATURES,
              'items_trained_on': combined['item_id'].nunique(),
              'rows_trained_on': len(combined)}
    bundle_path = os.path.join(models_dir, 'global_model.pkl')
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\nGlobal model saved → {bundle_path}")
    print(f"  Items: {bundle['items_trained_on']} | "
          f"Rows: {bundle['rows_trained_on']:,} | "
          f"Test MAE: {test_mae:.4f}")