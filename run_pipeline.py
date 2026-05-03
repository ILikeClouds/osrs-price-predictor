"""
run_pipeline.py
───────────────
Daily pipeline runner.  Execution order:

  1. preprocess.py        — fetch data, build features, accumulate history
  2. build_global_model.py — retrain global model from all item histories
  3. train_model.py       — evaluate whichever tier is active, log metrics
  4. cross_validate.py    — cross-validation (item-specific data only)
  5. tune_model.py        — grid-search hyperparameters
  6. predict_trends.py    — generate forecast + chart

build_global_model.py is skipped if only one item has been tracked,
since a single-item "global" model offers no advantage.
"""

import subprocess
import sys
import os
import json


def run_script(script_name: str, *args, required: bool = True) -> bool:
    """
    Run a script in /src/.  Returns True on success, False on failure.
    If required=True, exits the pipeline on failure.
    """
    print(f"\n{'='*52}")
    args_display = f" '{args[0]}'" if args else ""
    print(f"  EXECUTING: {script_name}{args_display}")
    print(f"{'='*52}")

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'src', script_name)
    command     = [sys.executable, script_path] + list(args)

    try:
        subprocess.run(command, check=True)
        print(f"\n  ✓ SUCCESS: {script_name} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ ERROR: {script_name} failed (exit code {e.returncode}).")
        if required:
            sys.exit(1)
        return False


def count_history_items(project_root: str) -> int:
    history_dir = os.path.join(project_root, 'data', 'history')
    if not os.path.exists(history_dir):
        return 0
    return sum(1 for f in os.listdir(history_dir) if f.endswith('.csv'))


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path  = os.path.join(project_root, 'config.json')

    print("=" * 52)
    print("  OSRS Price Predictor — Daily Pipeline")
    print("=" * 52)

    # ── Item selection ─────────────────────────────────────────────────────
    print("\n--- Item Selection ---")
    item_query = input(
        "Enter the OSRS item to track\n"
        "(or press Enter to re-use the saved item): "
    ).strip()

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    if item_query:
        run_script("preprocess.py", item_query)
    else:
        run_script("preprocess.py")

    # ── Step 2: Build / update global model ───────────────────────────────
    n_items = count_history_items(project_root)
    if n_items >= 2:
        print(f"\n  {n_items} items tracked — rebuilding global model...")
        run_script("build_global_model.py", required=False)
    else:
        print(
            f"\n  Only {n_items} item(s) tracked so far.\n"
            f"  Track ≥ 2 items to enable the global model (better cold-start accuracy).\n"
            f"  Skipping build_global_model.py."
        )

    # ── Steps 3-6: Train / validate / tune / forecast ─────────────────────
    for step in ["train_model.py", "cross_validate.py",
                 "tune_model.py", "predict_trends.py"]:
        run_script(step)

    # ── Summary ────────────────────────────────────────────────────────────
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    regime_days = config.get('regime_days', '?')
    tier        = 'Global' if config.get('use_global_model') else 'Item-specific'

    print("\n" + "=" * 52)
    print("  PIPELINE COMPLETE")
    print(f"  Item        : {config.get('item_name', '?')}")
    print(f"  Regime days : {regime_days}")
    print(f"  Model tier  : {tier}")
    print(f"  Items in DB : {n_items}")
    print(f"  Outputs     : /output/")
    print("=" * 52)