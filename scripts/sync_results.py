import shutil
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "results"
DEST_DIR = BASE_DIR / "nowcasting_results"

def copy_all():
    # Copy backtest CSVs
    backtest_src = SRC_DIR / "backtests"
    for f in backtest_src.glob("*.csv"):
        shutil.copy2(f, DEST_DIR / f.name)
        print(f"Copied {f.name}")
    # Copy forecast files
    forecast_src = SRC_DIR / "forecasts"
    for f in forecast_src.glob("*.*"):
        shutil.copy2(f, DEST_DIR / f.name)
        print(f"Copied {f.name}")

    # Copy figures
    fig_src = SRC_DIR / "figures"
    fig_dest = DEST_DIR / "figures"
    fig_dest.mkdir(parents=True, exist_ok=True)
    for f in fig_src.glob("*.png"):
        shutil.copy2(f, fig_dest / f.name)
        print(f"Copied figure {f.name}")

if __name__ == "__main__":
    copy_all()
