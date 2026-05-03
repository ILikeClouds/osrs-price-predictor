"""
commit_output.py
────────────────
Run by the GitHub Actions aggregation job after all per-item predictions
are collected. Scans docs/predictions/ for whatever JSON files exist and
generates docs/index.json — the master manifest that the RuneLite plugin
uses to discover all available items.
"""

import os
import json
from datetime import datetime, timezone


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir     = os.path.join(project_root, 'docs')
    pred_dir     = os.path.join(docs_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    index_entries = []

    pred_files = sorted(
        f for f in os.listdir(pred_dir) if f.endswith('.json')
    )

    if not pred_files:
        print("⚠  No prediction files found in docs/predictions/")
    else:
        for fname in pred_files:
            pred_path = os.path.join(pred_dir, fname)
            try:
                with open(pred_path) as f:
                    data = json.load(f)
                index_entries.append({
                    "id":            data.get('item_id'),
                    "name":          data.get('item_name'),
                    "current_price": data.get('current_price'),
                    "model_tier":    data.get('model_tier'),
                    "generated_at":  data.get('generated_at'),
                    "stale_after":   data.get('stale_after'),
                })
                print(f"  ✓ {data.get('item_name')} (ID: {data.get('item_id')})")
            except Exception as e:
                print(f"  ✗ Failed to read {fname}: {e}")

    index_doc = {
        "generated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "item_count":   len(index_entries),
        "items":        index_entries,
    }

    index_path = os.path.join(docs_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index_doc, f, indent=2)

    print(f"\n✓ index.json written with {len(index_entries)} item(s) → {index_path}")
