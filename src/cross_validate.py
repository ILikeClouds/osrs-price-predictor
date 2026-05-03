import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'preprocessed_data.csv')
    df = pd.read_csv(data_path)
    
    anchor_prices = df['daily_avg_price_raw'].values
    X = df.drop(columns=['target_price_7d', 'date', 'daily_avg_price_raw'])
    y = df['target_price_7d']
    
    config_path = os.path.join(project_root, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f: config = json.load(f)
        META_KEYS = {'item_id', 'item_name', 'regime_start', 'regime_days', 'use_global_model'}
        params = {k: v for k, v in config.items() if k not in META_KEYS}
        if not params: params = {'n_estimators': 100}
    else:
        params = {'n_estimators': 100}
        
    params.pop('random_state', None)
    
    base_model = RandomForestRegressor(**params, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    mae_scores = []
    
    # Custom loop to calculate absolute MAE on delta predictions
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_anchors = anchor_prices[test_index]
        
        base_model.fit(X_train, y_train)
        preds_delta = base_model.predict(X_test)
        
        y_test_abs = y_test + test_anchors
        y_pred_abs = preds_delta + test_anchors
        mae_scores.append(mean_absolute_error(y_test_abs, y_pred_abs))
    
    print("\n--- Cross-Validation Results ---")
    print(f"MAE for each time fold: {np.round(mae_scores, 2)}")
    print(f"Average Cross-Validation MAE: {np.mean(mae_scores):.2f} GP")