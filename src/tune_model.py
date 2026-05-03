import pandas as pd
import os
import json
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path    = os.path.join(project_root, 'data', 'preprocessed_data.csv')
    config_path  = os.path.join(project_root, 'config.json')

    df = pd.read_csv(data_path)
    X  = df.drop(columns=['target_price_7d', 'date', 'daily_avg_price_raw'])
    y  = df['target_price_7d']

    test_ratio = 0.1 if len(X) < 120 else 0.2
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    print("Starting TimeSeries GridSearchCV for Hyperparameter Tuning...")
    param_grid = {
        'n_estimators':      [50, 100, 150, 200],
        'max_depth':         [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"\nHyperparameter Tuning Complete!")
    print(f"Best Parameters: {best_params}")

    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f: config = json.load(f)

    config.update(best_params)
    with open(config_path, 'w') as f: json.dump(config, f, indent=2)