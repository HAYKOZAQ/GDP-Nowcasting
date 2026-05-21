from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_acquisition.fetch_armstat_official_monthly import (
    _load_exported_csv,
    _reshape_indicator_year_month_matrix,
    _reshape_measure_columns,
    _reshape_table,
    _reshape_year_month_matrix,
)


def test_armstat_csv_parser_and_reshape(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        '\n'.join(
            [
                '"Sample title"',
                "",
                '"types of economic activity according to the two-digit classification","2025 january","2025 february"',
                '"TOTAL INDUSTRY","1000","1100"',
                '"B. MINING AND QUARRYING","200","220"',
                '"07. Mining of metal ores","150","160"',
                '"C. MANUFACTURING","500","550"',
                '"D. ELECTRICITY, GAS, STEAM AND AIR CONDITIONING SUPPLY","100","101"',
                '"E. WATER SUPPLY;SEWERAGE, WASTE MANAGEMENT AND REMEDIATION ACTIVITIES","50","51"',
            ]
        ),
        encoding="utf-8-sig",
    )

    raw = _load_exported_csv(csv_path)
    reshaped = _reshape_table(raw, "Current_Mln_AMD", 0.001)

    assert list(reshaped.columns) == [
        "ArmStat_Industry_Total_Current_Mln_AMD",
        "ArmStat_Industry_Mining_Current_Mln_AMD",
        "ArmStat_Industry_MetalOres_Current_Mln_AMD",
        "ArmStat_Industry_Manufacturing_Current_Mln_AMD",
        "ArmStat_Industry_ElectricityGas_Current_Mln_AMD",
        "ArmStat_Industry_WaterWaste_Current_Mln_AMD",
    ]
    assert reshaped.loc[pd.Timestamp("2025-01-01"), "ArmStat_Industry_Total_Current_Mln_AMD"] == 1.0
    assert reshaped.loc[pd.Timestamp("2025-02-01"), "ArmStat_Industry_MetalOres_Current_Mln_AMD"] == 0.16


def test_measure_column_parser():
    df = pd.DataFrame(
        {
            "years": [2025, 2025],
            "months": ["January", "February"],
            "Chain-link indexes (2023=100)": [101.0, 102.0],
            "Chain-link indexes with seasonal adjustment, % (2023=100)": [99.0, 100.0],
        }
    )
    reshaped = _reshape_measure_columns(
        df,
        {
            "Chain-link indexes (2023=100)": "EAI",
            "Chain-link indexes with seasonal adjustment, % (2023=100)": "EAI_SA",
        },
    )
    assert reshaped.loc[pd.Timestamp("2025-01-01"), "EAI"] == 101.0
    assert reshaped.loc[pd.Timestamp("2025-02-01"), "EAI_SA"] == 100.0


def test_year_month_matrix_parser():
    df = pd.DataFrame({"years": [2024, 2025], "January": [99.0, 101.0], "February": [98.0, 102.0]})
    reshaped = _reshape_year_month_matrix(df, "TradeIndex")
    assert reshaped.loc[pd.Timestamp("2024-01-01"), "TradeIndex"] == 99.0
    assert reshaped.loc[pd.Timestamp("2025-02-01"), "TradeIndex"] == 102.0


def test_indicator_year_month_matrix_parser():
    df = pd.DataFrame(
        {
            "2025": [2024, 2024],
            "indicators": ["Freight tariffs index, total", "by road"],
            "January": [100.0, 105.0],
            "February": [101.0, 106.0],
        }
    )
    reshaped = _reshape_indicator_year_month_matrix(
        df,
        {
            "FREIGHT TARIFFS INDEX, TOTAL": "Total",
            "BY ROAD": "Road",
        },
    )
    assert reshaped.loc[pd.Timestamp("2024-01-01"), "Total"] == 100.0
    assert reshaped.loc[pd.Timestamp("2024-02-01"), "Road"] == 106.0
