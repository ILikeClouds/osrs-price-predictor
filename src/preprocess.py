import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum regime days before the item-specific model is trusted over the global one
MIN_DAYS_FOR_ITEM_MODEL = 45


# ── Item Lookup ────────────────────────────────────────────────────────────────

def lookup_item(item_name: str) -> dict:
    headers = {'User-Agent': 'WGU_D683_Task2_Pipeline', 'Accept': 'application/json'}
    mapping = requests.get(
        "https://prices.runescape.wiki/api/v1/osrs/mapping", headers=headers
    ).json()
    query   = item_name.lower().strip()
    exact   = [i for i in mapping if i['name'].lower() == query]
    partial = [i for i in mapping if query in i['name'].lower()]
    matches = exact or partial
    if not matches:
        raise ValueError(
            f"No OSRS item found matching '{item_name}'.\n"
            "Tip: try a more specific name, e.g. \"Zulrah's scales\" or \"Dragon bones\"."
        )
    if len(matches) > 1 and not exact:
        names = [m['name'] for m in matches[:8]]
        extra = f" (+{len(matches)-8} more)" if len(matches) > 8 else ""
        raise ValueError(
            f"'{item_name}' matched {len(matches)} items. Please be more specific.\n"
            f"Candidates: {', '.join(names)}{extra}"
        )
    best = matches[0]
    print(f"Item resolved: '{best['name']}' (ID: {best['id']})")
    return {'item_id': best['id'], 'item_name': best['name']}


# ── Data Fetching ──────────────────────────────────────────────────────────────

def get_osrs_data(item_id: int) -> pd.DataFrame:
    headers = {'User-Agent': 'WGU_D683_Task2_Pipeline', 'Accept': 'application/json'}
    url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=24h&id={item_id}"
    return pd.DataFrame(requests.get(url, headers=headers).json()['data'])


# ── Regime Detection ───────────────────────────────────────────────────────────

def detect_regime_start(
    df: pd.DataFrame, lookback: int = 14, z_threshold: float = 2.5
) -> pd.Timestamp:
    prices       = df['daily_avg_price_raw']
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std  = prices.rolling(lookback).std().replace(0, 1e-10)
    z_scores     = (prices - rolling_mean) / rolling_std
    breakpoints  = df[z_scores.abs() > z_threshold]
    if not breakpoints.empty:
        last_break  = pd.to_datetime(breakpoints['date'].iloc[-1])
        regime_start = last_break + pd.Timedelta(days=1)
        print(f"Regime break detected on {last_break.date()} — "
              f"training from {regime_start.date()} onward.")
    else:
        regime_start = pd.to_datetime(df['date'].max()) - pd.Timedelta(days=90)
        print(f"No structural break detected. Using 90-day fallback "
              f"(from {regime_start.date()}).")
    return regime_start


# ── Feature Engineering ────────────────────────────────────────────────────────

def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    slopes = [np.nan] * len(series)
    x = np.arange(window)
    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1: i + 1].values
        if not np.isnan(y).any():
            slopes[i] = np.polyfit(x, y, 1)[0]
    return pd.Series(slopes, index=series.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ALL features — both absolute (item-specific model) and
    relative/scale-free (global model).  Returns the enriched DataFrame.
    """
    p = df['daily_avg_price_raw']

    # ── Absolute features (item-specific model) ────────────────────────────
    df['daily_avg_price'] = p
    df['total_volume']    = df['highPriceVolume'] + df['lowPriceVolume']
    df['price_spread']    = df['avgHighPrice'] - df['avgLowPrice']

    for w in [7, 14, 30, 90]:
        df[f'{w}_day_moving_avg'] = p.rolling(w).mean()
    df['ema_7']  = p.ewm(span=7,  adjust=False).mean()
    df['ema_14'] = p.ewm(span=14, adjust=False).mean()

    for i in range(1, 8):
        df[f'price_lag_{i}'] = p.shift(i)
    df['price_lag_14'] = p.shift(14)

    df['volatility_7d']  = p.rolling(7).std()
    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-10)
    df['rsi_14']          = 100 - (100 / (1 + gain / loss))
    df['volume_ma_7']     = df['total_volume'].rolling(7).mean()
    df['volume_momentum'] = df['total_volume'] - df['volume_ma_7']
    df['slope_7d']        = rolling_slope(p, 7)
    df['slope_30d']       = rolling_slope(p, 30)

    r_min7 = p.rolling(7).min()
    r_max7 = p.rolling(7).max()
    df['range_position_7d'] = (p - r_min7) / (r_max7 - r_min7).replace(0, 1e-10)

    # ── Relative / scale-free features (global model) ─────────────────────
    # These are price-agnostic — 2,500 GP Dragon bones and 180 GP Zulrah's
    # scales will produce comparable values, allowing the global model to
    # learn universal patterns across all tracked items.

    df['pct_change_1d']  = p.pct_change(1)
    df['pct_change_3d']  = p.pct_change(3)
    df['pct_change_7d']  = p.pct_change(7)
    df['pct_change_14d'] = p.pct_change(14)

    # Price position vs moving averages (as %)
    df['pct_vs_7ma']  = (p - df['7_day_moving_avg'])  / df['7_day_moving_avg'].replace(0, 1e-10)
    df['pct_vs_30ma'] = (p - df['30_day_moving_avg']) / df['30_day_moving_avg'].replace(0, 1e-10)
    df['pct_vs_90ma'] = (p - df['90_day_moving_avg']) / df['90_day_moving_avg'].replace(0, 1e-10)

    # EMA crossover signal
    df['ema_crossover'] = (df['ema_7'] - df['ema_14']) / df['ema_14'].replace(0, 1e-10)

    # Normalised slope (slope per unit of current price)
    df['slope_7d_pct']  = df['slope_7d']  / p.replace(0, 1e-10)
    df['slope_30d_pct'] = df['slope_30d'] / p.replace(0, 1e-10)

    # Normalised volatility (coefficient of variation)
    df['vol_pct_7d'] = df['volatility_7d'] / p.replace(0, 1e-10)

    # Normalised spread
    df['spread_pct'] = df['price_spread'] / p.replace(0, 1e-10)

    # Volume ratio (relative to its own moving average)
    df['volume_ratio'] = (
        df['total_volume'] / df['volume_ma_7'].replace(0, 1e-10)
    )

    # RSI and range_position are already scale-free — reused directly

    # Lag returns (% change from each lag to today)
    for i in [1, 2, 3, 7, 14]:
        df[f'lag_return_{i}d'] = (p - p.shift(i)) / p.shift(i).replace(0, 1e-10)

    return df


# ── Target Variables ───────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    p = df['daily_avg_price_raw']
    # Absolute delta (item-specific model)
    df['target_price_7d']     = p.shift(-7) - p
    # Percentage return (global model — scale-free)
    df['target_pct_7d']       = (p.shift(-7) - p) / p.replace(0, 1e-10)
    return df


# ── Column Lists ───────────────────────────────────────────────────────────────

ABSOLUTE_FEATURES = [
    'avgHighPrice', 'avgLowPrice', 'highPriceVolume', 'lowPriceVolume',
    'daily_avg_price', 'total_volume', 'price_spread',
    '7_day_moving_avg', '14_day_moving_avg', '30_day_moving_avg', '90_day_moving_avg',
    'ema_7', 'ema_14', 'volatility_7d', 'rsi_14',
    'volume_ma_7', 'volume_momentum', 'price_lag_14',
    'slope_7d', 'slope_30d', 'range_position_7d',
] + [f'price_lag_{i}' for i in range(1, 8)]

RELATIVE_FEATURES = [
    'pct_change_1d', 'pct_change_3d', 'pct_change_7d', 'pct_change_14d',
    'pct_vs_7ma', 'pct_vs_30ma', 'pct_vs_90ma',
    'ema_crossover', 'slope_7d_pct', 'slope_30d_pct',
    'vol_pct_7d', 'spread_pct', 'volume_ratio',
    'rsi_14', 'range_position_7d',
    'lag_return_1d', 'lag_return_2d', 'lag_return_3d',
    'lag_return_7d', 'lag_return_14d',
]

DROP_THRESHOLD_COLS = ['rsi_14', 'price_lag_14', '90_day_moving_avg',
                       'slope_30d', 'pct_change_14d', 'lag_return_14d']


# ── Main Preprocessing ─────────────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    item_id: int,
    history_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Returns:
        train_abs   — absolute-feature training set  (item-specific model)
        future_abs  — absolute-feature inference set (item-specific model)
        train_rel   — relative-feature training set  (global model contribution)
        future_rel  — relative-feature inference set (global model)
        regime_start
    """
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.dropna(subset=['avgHighPrice', 'avgLowPrice']).copy()
    df = df.drop(columns=['timestamp']).sort_values('date').reset_index(drop=True)
    df['daily_avg_price_raw'] = np.round(
        (df['avgHighPrice'] + df['avgLowPrice']) / 2
    ).astype(int)

    df = build_features(df)
    df = add_targets(df)
    df = df.dropna(subset=DROP_THRESHOLD_COLS).reset_index(drop=True)

    # ── Regime filter ──────────────────────────────────────────────────────
    regime_start = detect_regime_start(df)
    full_train = df.dropna(subset=['target_price_7d']).copy()
    full_future = df[df['target_price_7d'].isna()].copy()

    train_regime  = full_train[pd.to_datetime(full_train['date'])   >= regime_start].copy().reset_index(drop=True)
    future_regime = full_future[pd.to_datetime(full_future['date']) >= regime_start].copy().reset_index(drop=True)
    print(f"Training on {len(train_regime)} days in current regime.")

    # ── Persist relative-feature history (for global model) ───────────────
    # Saves ALL history (not just regime) so the global model learns from
    # every price environment this item has ever experienced.
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f'{item_id}.csv')
    rel_cols = ['date'] + RELATIVE_FEATURES + ['target_pct_7d']
    history_snapshot = full_train[rel_cols].copy()
    history_snapshot['item_id'] = item_id

    if os.path.exists(history_path):
        existing = pd.read_csv(history_path, parse_dates=['date'])
        combined = pd.concat([existing, history_snapshot], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'item_id']).sort_values('date')
    else:
        combined = history_snapshot

    combined.to_csv(history_path, index=False)
    print(f"History updated → {history_path} ({len(combined)} rows total)")

    # ── Scale absolute features ────────────────────────────────────────────
    train_abs  = train_regime.copy()
    future_abs = future_regime.copy()
    train_abs['target_price_7d']  = train_abs['target_price_7d'].astype(int)

    scaler = StandardScaler()
    train_abs[ABSOLUTE_FEATURES]  = scaler.fit_transform(train_abs[ABSOLUTE_FEATURES])
    future_abs[ABSOLUTE_FEATURES] = scaler.transform(future_abs[ABSOLUTE_FEATURES])

    # ── Scale relative features ────────────────────────────────────────────
    train_rel  = train_regime.copy()
    future_rel = future_regime.copy()
    train_rel['target_pct_7d'] = train_rel['target_pct_7d'].astype(float)

    rel_scaler = StandardScaler()
    train_rel[RELATIVE_FEATURES]  = rel_scaler.fit_transform(train_rel[RELATIVE_FEATURES])
    future_rel[RELATIVE_FEATURES] = rel_scaler.transform(future_rel[RELATIVE_FEATURES])

    return train_abs, future_abs, train_rel, future_rel, regime_start


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSRS Price Predictor — Preprocessing")
    parser.add_argument("item_name", nargs="?", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, 'data')
    history_dir  = os.path.join(project_root, 'data', 'history')
    config_path  = os.path.join(project_root, 'config.json')
    os.makedirs(data_dir, exist_ok=True)

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    if args.item_name:
        item_info = lookup_item(args.item_name)
        config.update(item_info)
    elif 'item_id' in config:
        item_info = {'item_id': config['item_id'], 'item_name': config['item_name']}
        print(f"Re-using saved item: '{item_info['item_name']}' (ID: {item_info['item_id']})")
    else:
        print("No item specified. Defaulting to Zulrah's scales.")
        item_info = lookup_item("Zulrah's scales")
        config.update(item_info)

    print(f"\nFetching price history for: {item_info['item_name']}...")
    raw_df = get_osrs_data(item_info['item_id'])

    train_abs, future_abs, train_rel, future_rel, regime_start = preprocess_data(
        raw_df, item_info['item_id'], history_dir
    )

    config['regime_start']      = regime_start.strftime('%Y-%m-%d')
    config['regime_days']       = len(train_abs)
    config['use_global_model']  = len(train_abs) < MIN_DAYS_FOR_ITEM_MODEL
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config updated → {config_path}")
    if config['use_global_model']:
        print(f"⚠  Only {len(train_abs)} regime days — global model will be used until {MIN_DAYS_FOR_ITEM_MODEL} days are reached.")
    else:
        print(f"✓  {len(train_abs)} regime days — item-specific model will be used.")

    train_abs.to_csv(os.path.join(data_dir, 'preprocessed_data.csv'),      index=False)
    future_abs.to_csv(os.path.join(data_dir, 'future_inference_data.csv'), index=False)
    train_rel.to_csv(os.path.join(data_dir, 'preprocessed_data_rel.csv'),  index=False)
    future_rel.to_csv(os.path.join(data_dir, 'future_inference_rel.csv'),  index=False)
    print("Datasets saved to /data.")