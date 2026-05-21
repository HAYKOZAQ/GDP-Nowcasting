from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.mixed_frequency_panel import build_dfm_panel, get_stage_panel, stage_cutoff_month
from src.nowcast_config import BacktestConfig


def _source_data_for_dfm() -> dict[str, pd.DataFrame | None]:
    monthly_index = pd.date_range("2018-01-01", periods=36, freq="MS")
    quarterly_index = pd.date_range("2018-01-01", periods=12, freq="QS")

    monthly = pd.DataFrame(
        {
            "Economic_Activity_Index_Discrete_YoY": np.linspace(100, 110, 36),
            "Industry_Real_Growth_YoY": np.linspace(99, 109, 36),
            "Construction_Real_Growth_YoY": np.linspace(98, 108, 36),
            "Services_Real_Growth_YoY": np.linspace(97, 107, 36),
            "Remittance_Inflow_Mln_AMD": np.linspace(1000, 1500, 36),
            "Remittance_Outflow_Mln_AMD": np.linspace(600, 900, 36),
            "Remittance_Net_Mln_AMD": np.linspace(400, 600, 36),
            "Loans_Industry_Mln_AMD": np.linspace(800, 1000, 36),
            "Loans_Construction_Mln_AMD": np.linspace(500, 850, 36),
            "Loans_Services_Mln_AMD": np.linspace(900, 1300, 36),
            "CPI_YoY": np.linspace(101, 104, 36),
            "Exchange_Rate_AMD_USD": np.linspace(480, 500, 36),
            "Exchange_Rate_AMD_RUB": np.linspace(6.5, 7.2, 36),
            "Brent_Oil_Price_USD_bbl": np.linspace(55, 75, 36),
            "Copper_Price_USD_mt": np.linspace(6000, 7200, 36),
            "ArmStat_EAI_ChainLink_2023_Index": np.linspace(90, 110, 36),
            "ArmStat_EAI_SA_2023_Index": np.linspace(91, 111, 36),
            "ArmStat_Industry_Total_Index": np.linspace(95, 115, 36),
            "ArmStat_Industry_Manufacturing_Index": np.linspace(96, 116, 36),
            "ArmStat_Industry_Mining_Index": np.linspace(97, 117, 36),
            "ArmStat_Export_UnitValue_MoM_Index": np.linspace(100, 104, 36),
            "ArmStat_Import_UnitValue_MoM_Index": np.linspace(99, 103, 36),
            "ArmStat_FreightTariff_Total_MoM_Index": np.linspace(101, 105, 36),
            "ArmStat_FreightTariff_Road_MoM_Index": np.linspace(100, 106, 36),
            "ArmStat_FreightTariff_Rail_MoM_Index": np.linspace(102, 107, 36),
        },
        index=monthly_index,
    )
    monthly["CPI_YoY_constant"] = 102.0

    quarterly = pd.DataFrame(
        {
            "Real_GDP_Armenia_YoY": np.linspace(100, 112, 12),
            "Primary_Income_Labor_Mln_USD": np.linspace(10, 20, 12),
            "Secondary_Income_Transfers_Mln_USD": np.linspace(8, 18, 12),
            "Exchange_Rate_AMD_USD_YoY": np.linspace(99, 101, 12),
            "Exchange_Rate_AMD_RUB_Abs": np.linspace(6.5, 7.2, 12),
            "REER_YoY": np.linspace(100, 103, 12),
        },
        index=quarterly_index,
    )
    gt = pd.DataFrame({"relocation armenia": np.linspace(5, 60, 36)}, index=monthly_index)
    return {
        "quarterly": quarterly,
        "monthly": monthly,
        "worldbank_quarterly": None,
        "gt_armenia_quarterly": None,
        "gt_armenian_quarterly": None,
        "gt_shock_quarterly": None,
        "gt_armenia_monthly": gt,
        "gt_armenian_monthly": None,
        "gt_shock_monthly": None,
        "wiki_quarterly": None,
        "wiki_monthly": None,
    }


def test_build_dfm_panel_aligns_indices():
    panel = build_dfm_panel(_source_data_for_dfm(), BacktestConfig())

    assert panel.monthly_panel.index[0] == pd.Timestamp("2018-01-31")
    assert panel.quarterly_target.index[0] == pd.Timestamp("2018-03-31")
    assert panel.quarterly_target.index.freqstr == "QE-DEC"
    assert "FAST_SHOCK_RELOCATION" in panel.monthly_panel.columns


def test_get_stage_panel_applies_ragged_edge_masks_and_hides_target_quarter():
    config = BacktestConfig()
    panel = build_dfm_panel(_source_data_for_dfm(), config)
    window = get_stage_panel(panel, pd.Timestamp("2020-04-01"), "Early", pd.Timestamp("2020-01-01"), config)

    assert stage_cutoff_month(pd.Timestamp("2020-04-01"), "Early") == pd.Timestamp("2020-04-30")
    assert window.quarterly_endog.index.max() == pd.Timestamp("2020-03-31")
    assert pd.isna(window.monthly_endog.loc[pd.Timestamp("2020-04-30"), "Economic_Activity_Index_Discrete_YoY"])
    assert pd.isna(window.monthly_endog.loc[pd.Timestamp("2020-05-31"), "Exchange_Rate_AMD_USD"])
    assert pd.notna(window.monthly_endog.loc[pd.Timestamp("2020-04-30"), "Exchange_Rate_AMD_USD"])


def test_get_stage_panel_filters_low_coverage_and_constant_series():
    config = BacktestConfig()
    source = _source_data_for_dfm()
    source["monthly"].loc[source["monthly"].index[10:], "Construction_Real_Growth_YoY"] = np.nan
    source["monthly"]["CPI_YoY"] = 102.0

    panel = build_dfm_panel(source, config)
    window = get_stage_panel(panel, pd.Timestamp("2020-04-01"), "Late", pd.Timestamp("2020-01-01"), config)

    assert "Construction_Real_Growth_YoY" not in window.monthly_endog.columns
    assert "CPI_YoY" not in window.monthly_endog.columns


def test_get_stage_panel_uses_bounded_monthly_history():
    config = BacktestConfig(dfm_history_months=12)
    panel = build_dfm_panel(_source_data_for_dfm(), config)

    window = get_stage_panel(panel, pd.Timestamp("2020-04-01"), "Late", pd.Timestamp("2020-01-01"), config)

    assert len(window.monthly_endog) == 12
    assert window.monthly_endog.index.min() == pd.Timestamp("2019-07-31")
