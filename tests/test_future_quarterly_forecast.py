from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.recursive_quarterly import run_future_quarterly_forecast


def test_future_quarterly_forecast_writes_expected_artifacts(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    quarterly_index = pd.date_range("2010-01-01", periods=64, freq="QS")
    quarterly = pd.DataFrame(
        {
            "Real_GDP_Armenia_YoY": 100 + np.sin(np.arange(64) / 3) * 3 + np.linspace(0, 5, 64),
            "Real_GDP_Russia_YoY": 99 + np.cos(np.arange(64) / 4),
            "CPI_YoY": 102 + np.linspace(0, 1, 64),
            "Exchange_Rate_AMD_USD_YoY": 100 + np.sin(np.arange(64) / 6),
            "REER_YoY": 101 + np.cos(np.arange(64) / 5),
            "Brent_Oil_Price_USD_bbl": 70 + np.sin(np.arange(64) / 7) * 4,
            "Copper_Price_USD_mt": 7000 + np.cos(np.arange(64) / 8) * 120,
            "Employment_YoY": 101 + np.linspace(0, 1.5, 64),
            "Unemployment_Rate_Pct": 17 - np.linspace(0, 1.0, 64),
            "Primary_Income_Labor_Mln_USD": 10 + np.linspace(0, 3, 64),
            "Secondary_Income_Transfers_Mln_USD": 8 + np.linspace(0, 2, 64),
            "Exchange_Rate_AMD_RUB_Abs": 6.5 + np.linspace(0, 0.8, 64),
        },
        index=quarterly_index,
    )

    monthly_index = pd.date_range("2010-01-01", periods=64 * 3, freq="MS")
    monthly = pd.DataFrame(
        {
            "Exchange_Rate_AMD_USD": 400 + np.linspace(0, 30, len(monthly_index)),
            "Economic_Activity_Index_Discrete_YoY": 103 + np.sin(np.arange(len(monthly_index)) / 6),
        },
        index=monthly_index,
    )

    workbook_path = raw_dir / "Translated_Cleaned_Nowcasting_Data.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        quarterly.to_excel(writer, sheet_name="Quarterly", index_label="Date")
        monthly.to_excel(writer, sheet_name="Monthly", index_label="Date")

    forecasts, scores = run_future_quarterly_forecast(tmp_path)

    assert len(forecasts) == 3
    assert forecasts["target_quarter"].tolist() == ["2026-Q1", "2026-Q2", "2026-Q3"]
    assert forecasts["selected_model"].notna().all()
    assert forecasts["forecast"].notna().all()
    assert not scores.empty
    assert (tmp_path / "results" / "forecasts" / "future_gdp_forecast.csv").exists()
    assert (tmp_path / "results" / "forecasts" / "future_gdp_model_scores.csv").exists()
    assert (tmp_path / "results" / "forecasts" / "future_gdp_forecast.md").exists()
