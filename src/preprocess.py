import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
import argparse

MIN_REGIME_ROWS = 14   # need at least this many rows to train meaningfully


# ── Item Lookup ────────────────────────────────────────────────────────────────

def lookup_item(item_name: str) -> dict:
    headers = {'User-Agent': 'osrs-price-predictor', 'Accept': 'application/json'}
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
    headers = {'User-Agent': 'osrs-price-predictor', 'Accept': 'application/json'}
    url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=24h&id={item_id}"
    response = requests.get(url, headers=headers)
    return pd.DataFrame(response.json()['data'])


# ── Regime Detection ───────────────────────────────────────────────────────────

def detect_regime_start(df: pd.DataFrame,
                        lookback: int = 14,
                        z_threshold: float = 2.5) -> pd.Timestamp:
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
        print(f"No structural break detected. Using 90-day fallback window "
              f"(from {regime_start.date()}).")

    return regime_start


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> tuple:
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.dropna(subset=['avgHighPrice', 'avgLowPrice']).copy()
    df = df.drop(columns=['timestamp']).sort_values('date').reset_index(drop=True)

    # Base features
    df['daily_avg_price_raw'] = np.round((df['avgHighPrice'] + df['avgLowPrice']) / 2).astype(int)
    df['daily_avg_price']     = df['daily_avg_price_raw']
    df['total_volume']        = df['highPriceVolume'] + df['lowPriceVolume']
    df['price_spread']        = df['avgHighPrice'] - df['avgLowPrice']

    # Moving averages & EMAs
    for window in [7, 14, 30, 90]:
        df[f'{window}_day_moving_avg'] = df['daily_avg_price_raw'].rolling(window).mean()
    df['ema_7']  = df['daily_avg_price_raw'].ewm(span=7,  adjust=False).mean()
    df['ema_14'] = df['daily_avg_price_raw'].ewm(span=14, adjust=False).mean()

    # Lag features
    for i in range(1, 8):
        df[f'price_lag_{i}'] = df['daily_avg_price_raw'].shift(i)
    df['price_lag_14'] = df['daily_avg_price_raw'].shift(14)

    # Momentum & volatility
    df['volatility_7d'] = df['daily_avg_price_raw'].rolling(7).std()
    delta = df['daily_avg_price_raw'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-10)
    df['rsi_14']          = 100 - (100 / (1 + gain / loss))
    df['volume_ma_7']     = df['total_volume'].rolling(7).mean()
    df['volume_momentum'] = df['total_volume'] - df['volume_ma_7']

    # Target — absolute price 7 days ahead
    df['target_price_7d'] = df['daily_avg_price_raw'].shift(-7)

    # Drop rows where rolling features are not yet populated
    df = df.dropna(subset=['rsi_14', 'price_lag_14', '90_day_moving_avg']).reset_index(drop=True)

    # ── Split into train / future BEFORE any regime filtering ─────────────────
    # Saving these originals is critical: the regime fallback re-filters from
    # here rather than from an already-empty filtered DataFrame.
    all_train  = df.dropna(subset=['target_price_7d']).copy()
    all_future = df[df['target_price_7d'].isna()].copy()
    all_train['target_price_7d'] = all_train['target_price_7d'].astype(int)

    # ── Auto Regime Filter ────────────────────────────────────────────────────
    regime_start = detect_regime_start(df)

    train_df  = all_train[ pd.to_datetime(all_train['date'])  >= regime_start].reset_index(drop=True)
    future_df = all_future[pd.to_datetime(all_future['date']) >= regime_start].reset_index(drop=True)

    if len(train_df) < MIN_REGIME_ROWS:
        # New regime is too fresh — fall back to the last 30 days of ALL data
        fallback_start = pd.to_datetime(all_train['date'].max()) - pd.Timedelta(days=30)
        print(f"Only {len(train_df)} days in new regime — falling back to "
              f"30-day window from {fallback_start.date()}.")
        train_df  = all_train[ pd.to_datetime(all_train['date'])  >= fallback_start].reset_index(drop=True)
        future_df = all_future[pd.to_datetime(all_future['date']) >= fallback_start].reset_index(drop=True)
        regime_start = fallback_start

    if len(train_df) < MIN_REGIME_ROWS:
        # Still not enough — use every available training row
        print(f"Still only {len(train_df)} rows after 30-day fallback — "
              f"using all {len(all_train)} available training rows.")
        train_df  = all_train.copy()
        future_df = all_future.copy()
        regime_start = pd.to_datetime(all_train['date'].min())

    print(f"Training on {len(train_df)} days.")

    # future_df should always include the most recent rows regardless of regime
    if len(future_df) == 0:
        future_df = all_future.copy()

    # ── Scaling ────────────────────────────────────────────────────────────────
    features_to_scale = [
        'avgHighPrice', 'avgLowPrice', 'highPriceVolume', 'lowPriceVolume',
        'daily_avg_price', 'total_volume', 'price_spread',
        '7_day_moving_avg', '14_day_moving_avg', '30_day_moving_avg', '90_day_moving_avg',
        'ema_7', 'ema_14', 'volatility_7d', 'rsi_14',
        'volume_ma_7', 'volume_momentum', 'price_lag_14',
    ] + [f'price_lag_{i}' for i in range(1, 8)]

    scaler = StandardScaler()
    train_df[features_to_scale]  = scaler.fit_transform(train_df[features_to_scale])
    future_df[features_to_scale] = scaler.transform(future_df[features_to_scale])

    return train_df, future_df, regime_start


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSRS Price Predictor — Preprocessing")
    parser.add_argument(
        "--item-name",
        type=str,
        default=None,
        help='Item name to track, e.g. "Dragon bones". '
             'If omitted, re-uses the item stored in config.json.'
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, 'data')
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
        print("No item specified and no config found. Defaulting to Dragon bones.")
        item_info = lookup_item("Dragon bones")
        config.update(item_info)

    print(f"\nFetching price history for: {item_info['item_name']}...")
    raw_df = get_osrs_data(item_info['item_id'])

    train_df, future_df, regime_start = preprocess_data(raw_df)

    config['regime_start'] = regime_start.strftime('%Y-%m-%d')
    config['regime_days']  = len(train_df)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config updated → {config_path}")

    train_df.to_csv(os.path.join(data_dir, 'preprocessed_data.csv'),      index=False)
    future_df.to_csv(os.path.join(data_dir, 'future_inference_data.csv'), index=False)
    print("Datasets saved to /data.")
