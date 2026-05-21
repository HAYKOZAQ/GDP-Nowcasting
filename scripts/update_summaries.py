import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "backtests"

def summarize(predictions):
    summaries = []
    for (model, stage), group in predictions.groupby(["model", "stage"]):
        valid = group[group["prediction"].notna()].copy()
        if valid.empty:
            continue
        # In a real run, intervals are also calculated. 
        # Here we just want to update the main metrics if possible.
        # But for simplicity, we'll just update mape, mae, rmse.
        summaries.append({
            "model": model,
            "stage": stage,
            "n_obs": int(len(valid)),
            "mape": float(valid["abs_pct_error"].mean()),
            "mae": float(valid["abs_error"].mean()),
            "rmse": float(np.sqrt(valid["squared_error"].mean())),
        })
    return pd.DataFrame(summaries)

def update_summaries(results_dir):
    preds_path = results_dir / "backtest_predictions.csv"
    summary_path = results_dir / "backtest_summary.csv"
    
    if not preds_path.exists() or not summary_path.exists():
        return
    
    preds = pd.read_csv(preds_path)
    old_summary = pd.read_csv(summary_path)
    
    new_metrics = summarize(preds)
    
    # Merge new metrics into old summary to keep other columns (like coverage)
    updated_summary = old_summary.copy()
    for _, row in new_metrics.iterrows():
        mask = (updated_summary['model'] == row['model']) & (updated_summary['stage'] == row['stage'])
        if mask.any():
            updated_summary.loc[mask, 'mape'] = row['mape']
            updated_summary.loc[mask, 'mae'] = row['mae']
            updated_summary.loc[mask, 'rmse'] = row['rmse']
            updated_summary.loc[mask, 'n_obs'] = row['n_obs']
    
    updated_summary.to_csv(summary_path, index=False)
    print(f"Updated summary in {results_dir.name}")

if __name__ == "__main__":
    update_summaries(RESULTS_DIR)
