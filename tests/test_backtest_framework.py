from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.nowcast_features import build_stage_datasets
from src.evaluation.walkforward import (
    _add_empirical_intervals,
    _add_error_metrics,
    _base_monthly_series_name,
    _build_diebold_mariano_tests,
    _build_focus_quarter_information_sets,
    _build_model_family_summary,
    _diebold_mariano_test,
    _google_trends_ablation_specs,
    _dfm_shock_adjusted_prediction,
    _early_shock_adjusted_prediction,
    _early_shock_features,
    _shadow_features,
    _shock_regime_probability,
    _shock_sign_agreement,
    _stage_ar_features,
    _stage_quarterly_columns,
    _summarize_predictions,
    _summarize_predictions_detailed,
    _write_supporting_backtest_artifacts,
)


def _mock_source_data() -> dict[str, pd.DataFrame | None]:
    q_index = pd.date_range("2010-01-01", periods=8, freq="QS")
    m_index = pd.date_range("2010-01-01", periods=24, freq="MS")

    quarterly = pd.DataFrame(
        {
            "Real_GDP_Armenia_YoY": [100, 101, 102, 103, 104, 105, 106, 107],
            "Real_GDP_Russia_YoY": [99, 99, 100, 100, 101, 101, 102, 102],
            "CPI_YoY": [103] * 8,
            "Exchange_Rate_AMD_USD_YoY": [100] * 8,
            "REER_YoY": [101] * 8,
            "Employment_YoY": [102] * 8,
            "Unemployment_Rate_Pct": [18] * 8,
            "Primary_Income_Labor_Mln_USD": [10, 11, 12, 13, 14, 15, 16, 17],
            "Secondary_Income_Transfers_Mln_USD": [5, 6, 7, 8, 9, 10, 11, 12],
            "Exchange_Rate_AMD_RUB_Abs": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            "Brent_Oil_Price_USD_bbl": [80] * 8,
            "Copper_Price_USD_mt": [8000] * 8,
        },
        index=q_index,
    )

    monthly = pd.DataFrame(
        {
            "Exchange_Rate_AMD_USD": range(24),
            "Exchange_Rate_AMD_RUB": range(24),
            "Brent_Oil_Price_USD_bbl": range(24),
            "Copper_Price_USD_mt": range(24),
            "CPI_YoY": [103] * 24,
            "Economic_Activity_Index_Discrete_YoY": [104] * 24,
            "Industry_Real_Growth_YoY": [101] * 24,
            "Construction_Real_Growth_YoY": [102] * 24,
            "Services_Real_Growth_YoY": [103] * 24,
            "Cash_in_Circulation_Mln_AMD": range(24),
            "Money_Supply_M2_Mln_AMD": range(24),
            "Money_Supply_M2X_Mln_AMD": range(24),
            "Commercial_Bank_Loans_Mln_AMD": range(24),
            "Household_Loans_Mln_AMD": range(24),
            "Total_Loans_Mln_AMD": range(24),
            "Loans_Industry_Mln_AMD": range(24),
            "Loans_Construction_Mln_AMD": range(24),
            "Loans_Services_Mln_AMD": range(24),
        },
        index=m_index,
    )

    monthly_alt = pd.DataFrame({"term": range(24)}, index=m_index)
    return {
        "quarterly": quarterly,
        "monthly": monthly,
        "worldbank_quarterly": None,
        "fintech_alt_quarterly": pd.DataFrame(
            {
                "ALTQ_CardTxn_Noncash_Volume_Mln_AMD": np.linspace(1000, 1700, len(q_index)),
                "ALTQ_EMoney_ActiveAccounts": np.linspace(50, 120, len(q_index)),
            },
            index=q_index,
        ),
        "gt_armenia_quarterly": None,
        "gt_armenian_quarterly": None,
        "gt_shock_quarterly": None,
        "gt_armenia_monthly": monthly_alt,
        "gt_armenian_monthly": None,
        "gt_shock_monthly": None,
        "wiki_quarterly": None,
        "wiki_monthly": None,
    }


def test_stage_datasets_respect_release_windows():
    datasets = build_stage_datasets(_mock_source_data(), "Real_GDP_Armenia_YoY")
    sample_date = pd.Timestamp("2010-01-01")

    early = datasets["Early"].frame.loc[sample_date]
    mid = datasets["Mid"].frame.loc[sample_date]
    late = datasets["Late"].frame.loc[sample_date]

    assert pd.isna(early["OFF_CPI_YoY_LAST"])
    assert mid["OFF_CPI_YoY_LAST"] == 103
    assert late["OFF_CPI_YoY_LAST"] == 103

    assert early["FAST_Exchange_Rate_AMD_USD_LAST"] == 0
    assert mid["FAST_Exchange_Rate_AMD_USD_LAST"] == 1
    assert late["FAST_Exchange_Rate_AMD_USD_LAST"] == 2


def test_summary_contains_interval_metrics():
    df = pd.DataFrame(
        {
            "prediction_date": pd.date_range("2015-01-01", periods=12, freq="QS"),
            "target_quarter": [f"2015-Q{i}" for i in [1, 2, 3, 4] * 3],
            "stage": ["Late"] * 12,
            "model": ["AR"] * 12,
            "actual": [100] * 12,
            "prediction": [99, 101, 100, 98, 102, 100, 99, 101, 100, 99, 101, 100],
            "train_end": pd.date_range("2014-01-01", periods=12, freq="QS"),
            "feature_count": [3] * 12,
        }
    )
    df = _add_error_metrics(df)
    df = _add_empirical_intervals(df)
    summary = _summarize_predictions(df)

    assert {"coverage_50", "coverage_90", "avg_width_50", "avg_width_90"}.issubset(summary.columns)
    assert len(summary) == 1


def test_diebold_mariano_test_detects_better_model():
    better = np.array([1.0, 1.5, 1.2, 0.8, 1.1, 1.0, 0.9, 1.3])
    worse = np.array([2.5, 2.8, 2.3, 1.9, 2.2, 2.0, 1.8, 2.4])

    result = _diebold_mariano_test(better, worse)

    assert result["n_obs"] == 8
    assert result["dm_stat"] > 0
    assert result["p_value"] < 0.05


def test_build_diebold_mariano_tests_generates_stage_pairs():
    df = pd.DataFrame(
        {
            "prediction_date": pd.date_range("2015-01-01", periods=8, freq="QS").repeat(2),
            "target_quarter": [f"{d.year}-Q{((d.month - 1) // 3) + 1}" for d in pd.date_range("2015-01-01", periods=8, freq="QS") for _ in range(2)],
            "stage": ["Late"] * 16,
            "model": ["ModelA", "ModelB"] * 8,
            "actual": [100.0] * 16,
            "prediction": [100.2, 102.5, 99.9, 102.0, 100.1, 101.8, 99.8, 102.2, 100.0, 101.9, 100.3, 102.4, 99.7, 102.1, 100.2, 102.6],
            "train_end": pd.date_range("2014-01-01", periods=16, freq="QS"),
            "feature_count": [3] * 16,
            "shock_flag": [False] * 16,
        }
    )
    df = _add_error_metrics(df)

    dm = _build_diebold_mariano_tests(df)

    assert len(dm) == 1
    row = dm.iloc[0]
    assert row["stage"] == "Late"
    assert {row["model_a"], row["model_b"]} == {"ModelA", "ModelB"}
    assert row["better_model"] == "ModelA"


def test_google_trends_ablation_specs_split_market_and_google_blocks():
    frame = pd.DataFrame(
        {
            "target": [100.0, 101.0],
            "target_quarter": ["2020-Q1", "2020-Q2"],
            "stage": ["Early", "Early"],
            "AR_LAG1": [99.0, 100.0],
            "Q_CPI_YoY": [103.0, 104.0],
            "OFF_CPI_YoY_LAST": [np.nan, np.nan],
            "FAST_Exchange_Rate_AMD_USD_LAST": [400.0, 405.0],
            "FAST_FIN_STRESS_PROXY_LAST": [2.0, 2.5],
            "FAST_GTG_Yerevan apartment_LAST": [30.0, 35.0],
            "FAST_SHOCK_COMPOSITE_LAST": [55.0, 56.0],
            "FAST_GOOGLE_ALL_LAST": [0.4, 0.5],
        }
    )

    specs = _google_trends_ablation_specs(frame)

    assert "FAST_Exchange_Rate_AMD_USD_LAST" in specs["Base+Market"]
    assert "FAST_FIN_STRESS_PROXY_LAST" in specs["Base+Market"]
    assert "FAST_GTG_Yerevan apartment_LAST" not in specs["Base+Market"]
    assert "FAST_SHOCK_COMPOSITE_LAST" not in specs["Base+Market"]
    assert "FAST_GOOGLE_ALL_LAST" not in specs["Base+Market"]
    assert "FAST_GTG_Yerevan apartment_LAST" not in specs["Base+Google"]
    assert "FAST_SHOCK_COMPOSITE_LAST" in specs["Base+Google"]
    assert "FAST_GOOGLE_ALL_LAST" in specs["Base+Google"]


def test_stage_datasets_add_shock_dummies_and_clean_fast_shock_names():
    q_index = pd.date_range("2020-01-01", periods=12, freq="QS")
    m_index = pd.date_range("2020-01-01", periods=36, freq="MS")

    quarterly = pd.DataFrame(
        {
            "Real_GDP_Armenia_YoY": range(100, 112),
            "Real_GDP_Russia_YoY": [99] * 12,
            "CPI_YoY": [103] * 12,
            "Exchange_Rate_AMD_USD_YoY": [100] * 12,
            "REER_YoY": [101] * 12,
            "Employment_YoY": [102] * 12,
            "Unemployment_Rate_Pct": [18] * 12,
            "Primary_Income_Labor_Mln_USD": range(10, 22),
            "Secondary_Income_Transfers_Mln_USD": range(5, 17),
            "Exchange_Rate_AMD_RUB_Abs": np.linspace(1.0, 2.1, 12),
            "Brent_Oil_Price_USD_bbl": [80] * 12,
            "Copper_Price_USD_mt": [8000] * 12,
        },
        index=q_index,
    )

    monthly = pd.DataFrame(
        {
            "Exchange_Rate_AMD_USD": range(36),
            "Exchange_Rate_AMD_RUB": range(36),
            "Brent_Oil_Price_USD_bbl": range(36),
            "Copper_Price_USD_mt": range(36),
            "CPI_YoY": [103] * 36,
            "Economic_Activity_Index_Discrete_YoY": [-7.0] + [104.0] * 35,
            "Industry_Real_Growth_YoY": [101] * 36,
            "Construction_Real_Growth_YoY": [102] * 36,
            "Services_Real_Growth_YoY": [103] * 36,
            "Remittance_Net_Mln_AMD": np.linspace(100, 300, 36),
        },
        index=m_index,
    )
    monthly_alt = pd.DataFrame({"apartment rent yerevan": [70.0] * 36}, index=m_index)

    datasets = build_stage_datasets(
        {
            "quarterly": quarterly,
            "monthly": monthly,
            "worldbank_quarterly": None,
            "fintech_alt_quarterly": None,
            "gt_armenia_quarterly": None,
            "gt_armenian_quarterly": None,
            "gt_shock_quarterly": None,
            "gt_armenia_monthly": monthly_alt,
            "gt_armenian_monthly": None,
            "gt_shock_monthly": None,
            "wiki_quarterly": None,
            "wiki_monthly": None,
        },
        "Real_GDP_Armenia_YoY",
    )

    mid = datasets["Mid"].frame

    assert "FAST_SHOCK_COMPOSITE_LAST" in mid.columns
    assert "FAST_FAST_SHOCK_COMPOSITE_LAST" not in mid.columns
    assert mid.loc[pd.Timestamp("2020-04-01"), "CURR_Dummy_COVID_LOCKDOWN"] == 1
    assert mid.loc[pd.Timestamp("2022-04-01"), "CURR_Dummy_RELOCATION_BOOM"] == 1
    assert mid.loc[pd.Timestamp("2020-01-01"), "CURR_Dummy_ACTIVITY_CRASH"] == 1
    assert mid.loc[pd.Timestamp("2021-01-01"), "CURR_Dummy_REMITTANCE_SURGE"] == 1


def test_late_stage_keeps_armstat_monthly_block():
    datasets = build_stage_datasets(_mock_source_data(), "Real_GDP_Armenia_YoY")
    late_cols = datasets["Late"].frame.columns

    # Synthetic mock data has no ArmStat series, so build a focused fixture.
    q_index = pd.date_range("2024-01-01", periods=4, freq="QS")
    m_index = pd.date_range("2024-01-01", periods=12, freq="MS")
    quarterly = pd.DataFrame(
        {
            "Real_GDP_Armenia_YoY": [100.0, 101.0, 102.0, 103.0],
            "Real_GDP_Russia_YoY": [99.0] * 4,
            "CPI_YoY": [103.0] * 4,
            "Exchange_Rate_AMD_USD_YoY": [100.0] * 4,
            "REER_YoY": [101.0] * 4,
            "Employment_YoY": [102.0] * 4,
            "Unemployment_Rate_Pct": [18.0] * 4,
            "Primary_Income_Labor_Mln_USD": [10.0, 11.0, 12.0, 13.0],
            "Secondary_Income_Transfers_Mln_USD": [5.0, 6.0, 7.0, 8.0],
            "Exchange_Rate_AMD_RUB_Abs": [1.0, 1.1, 1.2, 1.3],
            "Brent_Oil_Price_USD_bbl": [80.0] * 4,
            "Copper_Price_USD_mt": [8000.0] * 4,
        },
        index=q_index,
    )
    monthly = pd.DataFrame(
        {
            "Exchange_Rate_AMD_USD": np.linspace(380, 390, 12),
            "Exchange_Rate_AMD_RUB": np.linspace(4.5, 5.0, 12),
            "Brent_Oil_Price_USD_bbl": np.linspace(75, 80, 12),
            "Copper_Price_USD_mt": np.linspace(7800, 8000, 12),
            "CPI_YoY": [103.0] * 12,
            "Economic_Activity_Index_Discrete_YoY": [104.0] * 12,
            "ArmStat_EAI_ChainLink_2023_Index": np.linspace(99, 105, 12),
        },
        index=m_index,
    )
    focused = build_stage_datasets(
        {
            "quarterly": quarterly,
            "monthly": monthly,
            "worldbank_quarterly": None,
            "fintech_alt_quarterly": None,
            "gt_armenia_quarterly": None,
            "gt_armenian_quarterly": None,
            "gt_shock_quarterly": None,
            "gt_armenia_monthly": None,
            "gt_armenian_monthly": None,
            "gt_shock_monthly": None,
            "wiki_quarterly": None,
            "wiki_monthly": None,
        },
        "Real_GDP_Armenia_YoY",
    )

    assert any(col.startswith("OFF_ArmStat_EAI_ChainLink_2023_Index_") for col in focused["Late"].frame.columns)


def test_stage_datasets_include_fintech_quarterly_alternatives():
    datasets = build_stage_datasets(_mock_source_data(), "Real_GDP_Armenia_YoY")
    early_cols = datasets["Early"].frame.columns

    assert "Q_ALT_ALTQ_CardTxn_Noncash_Volume_Mln_AMD" in early_cols
    assert "Q_ALT_ALTQ_EMoney_ActiveAccounts" in early_cols


def test_mid_stage_prunes_redundant_regime_dummies():
    datasets = build_stage_datasets(_mock_source_data(), "Real_GDP_Armenia_YoY")
    mid_frame = datasets["Mid"].frame

    quarterly_cols = _stage_quarterly_columns(mid_frame, "Mid")
    ar_cols = _stage_ar_features(mid_frame, "Mid")

    for col in (
        "CURR_Dummy_COVID",
        "CURR_Dummy_RU_WAR",
        "CURR_Dummy_RELOCATION_NORMALIZE",
        "CURR_Dummy_ACTIVITY_CRASH",
        "CURR_Dummy_REMITTANCE_SURGE",
        "CURR_Dummy_GOOGLE_SHOCK",
    ):
        assert col not in quarterly_cols
        assert col not in ar_cols


def test_dfm_shock_adjustment_anchors_to_current_activity_signal():
    test_df = pd.DataFrame(
        {
            "OFF_Economic_Activity_Index_Discrete_YoY_LAST": [83.0],
            "OFF_Industry_Real_Growth_YoY_LAST": [92.0],
            "OFF_Construction_Real_Growth_YoY_LAST": [49.0],
            "OFF_Services_Real_Growth_YoY_LAST": [80.0],
            "OFF_Remittance_Net_Mln_AMD_YoY_LAST": [-24.0],
            "OFF_Remittance_Inflow_Mln_AMD_YoY_LAST": [-36.0],
            "OFF_Remittance_Outflow_Mln_AMD_YoY_LAST": [-40.0],
        },
        index=[pd.Timestamp("2020-04-01")],
    )
    predictions = {"DFM": 110.0, "ElasticNet": 100.0}

    adjusted = _dfm_shock_adjusted_prediction(test_df, "Late", True, predictions)

    assert adjusted is not None
    assert adjusted < predictions["ElasticNet"]
    assert adjusted > 70.0


def test_dfm_shock_adjustment_leaves_non_shock_rows_unchanged():
    test_df = pd.DataFrame(
        {"OFF_Economic_Activity_Index_Discrete_YoY_LAST": [105.0]},
        index=[pd.Timestamp("2021-10-01")],
    )

    adjusted = _dfm_shock_adjusted_prediction(test_df, "Late", False, {"DFM": 107.0, "ElasticNet": 104.0})

    assert adjusted == 107.0


def test_early_shock_features_restrict_to_fast_block():
    df = pd.DataFrame(
        {
            "AR_L1": [1.0],
            "FAST_SHOCK_COMPOSITE_LAST": [10.0],
            "FAST_SHOCK_BANKING_LAST": [20.0],
            "FAST_SHOCK_HOUSING_LAST": [11.0],
            "FAST_Exchange_Rate_AMD_RUB_LAST": [7.2],
            "FAST_Exchange_Rate_AMD_USD_LAST": [480.0],
            "FAST_Brent_Oil_Price_USD_bbl_LAST": [80.0],
            "FAST_Copper_Price_USD_mt_LAST": [8000.0],
            "CURR_Dummy_COVID_LOCKDOWN": [1],
            "CURR_Dummy_WAR_ONSET": [0],
            "OFF_Economic_Activity_Index_Discrete_YoY_LAST": [88.0],
        }
    )

    cols = _early_shock_features(df)

    assert "AR_L1" not in cols
    assert "OFF_Economic_Activity_Index_Discrete_YoY_LAST" not in cols
    assert "FAST_SHOCK_COMPOSITE_LAST" in cols
    assert "CURR_Dummy_COVID_LOCKDOWN" in cols


def test_shadow_features_stay_compact_and_stable():
    df = pd.DataFrame(
        {
            "AR_LAG1": [1.0],
            "AR_LAG4": [1.0],
            "Q_Real_GDP_Russia_YoY": [99.0],
            "Q_CPI_YoY": [103.0],
            "FAST_Exchange_Rate_AMD_USD_LAST": [480.0],
            "FAST_Exchange_Rate_AMD_RUB_LAST": [6.8],
            "OFF_Economic_Activity_Index_Discrete_YoY_LAST": [95.0],
            "FAST_SHOCK_COMPOSITE_LAST": [10.0],
        }
    )

    cols = _shadow_features(df)

    assert len(cols) <= 10
    assert "AR_LAG1" in cols
    assert "FAST_SHOCK_COMPOSITE_LAST" not in cols


def test_early_shock_adjustment_activates_for_extreme_shocks():
    train_df = pd.DataFrame(
        {
            "FAST_SHOCK_COMPOSITE_LAST": [5.0, 6.0, 7.0, 8.0, 9.0],
            "FAST_SHOCK_BANKING_LAST": [20.0, 21.0, 22.0, 23.0, 24.0],
            "FAST_SHOCK_HOUSING_LAST": [9.0, 9.5, 10.0, 10.5, 11.0],
            "FAST_Exchange_Rate_AMD_USD_LAST": [480.0, 481.0, 482.0, 483.0, 484.0],
            "FAST_Exchange_Rate_AMD_RUB_LAST": [6.5, 6.6, 6.7, 6.8, 6.9],
            "FAST_Brent_Oil_Price_USD_bbl_LAST": [80.0, 81.0, 82.0, 83.0, 84.0],
            "FAST_Copper_Price_USD_mt_LAST": [8000.0, 8050.0, 8100.0, 8150.0, 8200.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "stage": ["Early"],
            "FAST_SHOCK_COMPOSITE_LAST": [20.0],
            "FAST_SHOCK_BANKING_LAST": [30.0],
            "FAST_SHOCK_HOUSING_LAST": [13.0],
            "FAST_Exchange_Rate_AMD_USD_LAST": [495.0],
            "FAST_Exchange_Rate_AMD_RUB_LAST": [7.5],
            "FAST_Brent_Oil_Price_USD_bbl_LAST": [70.0],
            "FAST_Copper_Price_USD_mt_LAST": [7800.0],
            "CURR_Dummy_COVID_LOCKDOWN": [1],
        },
        index=[pd.Timestamp("2020-04-01")],
    )

    adjusted = _early_shock_adjusted_prediction(
        train_df,
        test_df,
        {"ElasticNet": 103.0, "EarlyShockBridge": 101.0, "Shadow": 99.0},
        0.85,
    )

    assert adjusted is not None
    assert adjusted < 101.0


def test_sign_agreement_requires_multiple_unrelated_signals():
    train_df = pd.DataFrame(
        {
            "FAST_SHOCK_COMPOSITE_LAST": [5.0, 6.0, 7.0, 8.0],
            "FAST_SHOCK_BANKING_LAST": [5.0, 6.0, 7.0, 8.0],
            "FAST_SHOCK_HOUSING_LAST": [5.0, 6.0, 7.0, 8.0],
            "FAST_Exchange_Rate_AMD_USD_LAST": [480.0, 481.0, 482.0, 483.0],
            "FAST_Brent_Oil_Price_USD_bbl_LAST": [80.0, 81.0, 82.0, 83.0],
            "FAST_Copper_Price_USD_mt_LAST": [8000.0, 8050.0, 8100.0, 8150.0],
        }
    )
    row = pd.Series(
        {
            "FAST_SHOCK_COMPOSITE_LAST": 20.0,
            "FAST_SHOCK_BANKING_LAST": 19.0,
            "FAST_SHOCK_HOUSING_LAST": 18.0,
            "FAST_Exchange_Rate_AMD_USD_LAST": 490.0,
            "FAST_Brent_Oil_Price_USD_bbl_LAST": 70.0,
            "FAST_Copper_Price_USD_mt_LAST": 7800.0,
        }
    )

    assert _shock_sign_agreement(train_df, row) >= 3


def test_shock_probability_classifier_returns_probability():
    train_df = pd.DataFrame(
        {
            "FAST_SHOCK_COMPOSITE_LAST": [1.0, 2.0, 10.0, 11.0, 2.0, 12.0],
            "FAST_SHOCK_BANKING_LAST": [1.0, 2.0, 10.0, 11.0, 2.0, 12.0],
            "FAST_SHOCK_HOUSING_LAST": [1.0, 2.0, 9.0, 10.0, 2.0, 11.0],
            "FAST_Exchange_Rate_AMD_RUB_LAST": [6.0, 6.1, 7.1, 7.2, 6.1, 7.3],
            "FAST_Exchange_Rate_AMD_USD_LAST": [480.0, 481.0, 510.0, 511.0, 482.0, 512.0],
            "FAST_Brent_Oil_Price_USD_bbl_LAST": [80.0, 81.0, 65.0, 64.0, 82.0, 63.0],
            "FAST_Copper_Price_USD_mt_LAST": [8000.0, 8050.0, 7600.0, 7550.0, 8100.0, 7500.0],
            "FAST_FIN_STRESS_PROXY_LAST": [0.2, 0.3, 3.0, 3.2, 0.4, 3.4],
            "FAST_RUS_LINK_OIL_RUB_LAST": [480.0, 494.1, 461.5, 460.8, 500.2, 459.9],
            "FAST_RUS_LINK_RUB_STRESS_LAST": [0.1, 0.2, 2.0, 2.1, 0.2, 2.2],
            "CURR_Dummy_COVID_LOCKDOWN": [0, 0, 1, 1, 0, 1],
            "CURR_Dummy_WAR_ONSET": [0, 0, 0, 0, 0, 0],
            "CURR_Dummy_COVID": [0, 0, 1, 1, 0, 1],
            "CURR_Dummy_RU_WAR": [0, 0, 0, 0, 0, 0],
        }
    )
    test_df = train_df.iloc[[2]].copy()

    prob = _shock_regime_probability(train_df, test_df)

    assert 0.0 <= prob <= 1.0
    assert prob > 0.5


def test_information_set_audit_and_family_summary_build():
    datasets = build_stage_datasets(_mock_source_data(), "Real_GDP_Armenia_YoY")
    info_sets = _build_focus_quarter_information_sets(datasets)

    assert "available_official_count" in info_sets.columns
    assert "mean_shock_composite" in info_sets.columns

    assert _base_monthly_series_name("OFF_CPI_YoY_LAST") == "OFF_CPI_YoY"
    assert _base_monthly_series_name("FAST_SHOCK_COMPOSITE_ALMON") == "FAST_SHOCK_COMPOSITE"

    predictions = pd.DataFrame(
        {
            "prediction_date": pd.date_range("2020-01-01", periods=6, freq="QS"),
            "target_quarter": ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", "2021-Q1", "2021-Q2"],
            "stage": ["Early", "Early", "Late", "Late", "Mid", "Mid"],
            "model": ["DFM", "DFM", "AdaptiveEnsemble", "AdaptiveEnsemble", "ElasticNet", "ElasticNet"],
            "actual": [100, 90, 95, 105, 100, 110],
            "prediction": [101, 99, 94, 102, 98, 109],
            "train_end": pd.date_range("2019-01-01", periods=6, freq="QS"),
            "feature_count": [5, 5, 7, 7, 8, 8],
            "shock_flag": [False, True, True, False, False, True],
        }
    )
    predictions = _add_error_metrics(predictions)
    detailed = _summarize_predictions_detailed(predictions)
    family_summary = _build_model_family_summary(detailed)

    assert "shock_mape" in family_summary.columns
    assert set(family_summary["family"]) <= {"Structural", "ML", "Combination"}


def test_supporting_backtest_artifacts_are_written(tmp_path):
    datasets = build_stage_datasets(_mock_source_data(), "Real_GDP_Armenia_YoY")
    predictions = pd.DataFrame(
        {
            "prediction_date": pd.date_range("2020-01-01", periods=6, freq="QS"),
            "target_quarter": ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", "2021-Q1", "2021-Q2"],
            "stage": ["Early", "Early", "Late", "Late", "Mid", "Mid"],
            "model": ["DFM", "DFMShockAdjusted", "AdaptiveEnsemble", "ShockSwitch", "ElasticNet", "EarlyShockAdjusted"],
            "actual": [100, 90, 95, 105, 100, 110],
            "prediction": [101, 95, 94, 102, 98, 101],
            "train_end": pd.date_range("2019-01-01", periods=6, freq="QS"),
            "feature_count": [5, 5, 7, 7, 8, 8],
            "shock_flag": [False, True, True, False, False, True],
        }
    )
    predictions = _add_error_metrics(predictions)
    predictions = _add_empirical_intervals(predictions)
    summary = _summarize_predictions(predictions)
    detailed = _summarize_predictions_detailed(predictions)

    _write_supporting_backtest_artifacts(tmp_path, predictions, summary, detailed, datasets)

    for name in (
        "model_selection_report.md",
        "model_family_summary.csv",
        "model_failure_report.md",
        "backtest_summary_detailed.csv",
        "residual_bias_summary.csv",
        "focus_quarter_information_sets.csv",
    ):
        assert (tmp_path / name).exists()
