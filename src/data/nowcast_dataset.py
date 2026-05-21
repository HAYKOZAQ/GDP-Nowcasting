from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nowcast_config import BacktestConfig


STAGES = ("Early", "Mid", "Late")
FAST_MONTHLY_COUNTS = {"Early": 1, "Mid": 2, "Late": 3}
OFFICIAL_MONTHLY_COUNTS = {"Early": 0, "Mid": 1, "Late": 3}


def _read_csv_if_exists(path: Path, index_col: int | str = 0) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=index_col)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _merge_monthly_extensions(base: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    merged = base.sort_index()
    extension_files = [
        processed_dir / "cba_nowcast_extension_monthly.csv",
        processed_dir / "armstat_nowcast_extension_monthly.csv",
    ]
    for path in extension_files:
        extension = _read_csv_if_exists(path, index_col="date")
        if extension is None:
            continue
        merged = merged.combine_first(extension.sort_index())
    return merged


def load_source_data(base_dir: Path, config: BacktestConfig) -> dict[str, pd.DataFrame | None]:
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    workbook_path = raw_dir / config.workbook_name

    quarterly = pd.read_excel(workbook_path, sheet_name="Quarterly", index_col="Date")
    monthly = pd.read_excel(workbook_path, sheet_name="Monthly", index_col="Date")
    quarterly.index = pd.to_datetime(quarterly.index)
    monthly.index = pd.to_datetime(monthly.index)
    monthly = _merge_monthly_extensions(monthly, processed_dir)

    return {
        "quarterly": quarterly.sort_index(),
        "monthly": monthly.sort_index(),
        # Annual World Bank series are kept on disk but excluded from the active nowcast pipeline.
        "worldbank_quarterly": None,
        "fintech_alt_quarterly": _read_csv_if_exists(
            processed_dir / "cba_fintech_alternatives_quarterly.csv", index_col="date"
        ),
        "gt_armenia_quarterly": _read_csv_if_exists(
            processed_dir / "google_trends_armenia_quarterly.csv", index_col="date"
        ),
        "gt_armenian_quarterly": _read_csv_if_exists(processed_dir / "google_trends_armenian_quarterly.csv"),
        "gt_shock_quarterly": _read_csv_if_exists(processed_dir / "google_trends_shock_quarterly.csv"),
        "gt_armenia_monthly": _read_csv_if_exists(
            processed_dir / "google_trends_armenia_monthly.csv", index_col="date"
        ),
        "gt_armenian_monthly": _read_csv_if_exists(processed_dir / "google_trends_armenian_monthly.csv"),
        "gt_shock_monthly": _read_csv_if_exists(processed_dir / "google_trends_shock_monthly.csv"),
        "wiki_quarterly": None,
        "wiki_monthly": None,
    }
