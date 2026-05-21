from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import MonthEnd, QuarterEnd

from src.data.nowcast_dataset import FAST_MONTHLY_COUNTS, OFFICIAL_MONTHLY_COUNTS
from src.features.nowcast_features import _prepare_monthly_groups
from src.nowcast_config import BacktestConfig


REAL_ACTIVITY_PRIORITY = [
    "Economic_Activity_Index_Discrete_YoY",
    "Industry_Real_Growth_YoY",
    "Construction_Real_Growth_YoY",
    "Services_Real_Growth_YoY",
]

ARMSTAT_PRIORITY = [
    "ArmStat_EAI_ChainLink_2023_Index",
    "ArmStat_EAI_SA_2023_Index",
    "ArmStat_Industry_Total_Index",
    "ArmStat_Industry_Manufacturing_Index",
    "ArmStat_Industry_Mining_Index",
    "ArmStat_Export_UnitValue_MoM_Index",
    "ArmStat_Import_UnitValue_MoM_Index",
    "ArmStat_FreightTariff_Total_MoM_Index",
    "ArmStat_FreightTariff_Road_MoM_Index",
    "ArmStat_FreightTariff_Rail_MoM_Index",
]

FINANCIAL_PRIORITY = [
    "Remittance_Inflow_Mln_AMD_YoY",
    "Remittance_Outflow_Mln_AMD_YoY",
    "Remittance_Net_Mln_AMD_YoY",
    "Loans_Industry_Mln_AMD_YoY",
    "Loans_Construction_Mln_AMD_YoY",
    "Loans_Services_Mln_AMD_YoY",
    "CPI_YoY",
]

FAST_PRIORITY = [
    "Exchange_Rate_AMD_USD",
    "Exchange_Rate_AMD_RUB",
    "Brent_Oil_Price_USD_bbl",
    "Copper_Price_USD_mt",
]

SHOCK_PRIORITY = [
    "FAST_SHOCK_COMPOSITE",
    "FAST_SHOCK_RELOCATION",
    "FAST_SHOCK_BANKING",
]

DFM_PRIORITY_ORDER = REAL_ACTIVITY_PRIORITY + ARMSTAT_PRIORITY + FINANCIAL_PRIORITY + FAST_PRIORITY + SHOCK_PRIORITY
STAGE_MONTH_END_COUNTS = {"Early": 1, "Mid": 2, "Late": 3}


@dataclass(frozen=True)
class MixedFrequencyPanel:
    monthly_panel: pd.DataFrame
    quarterly_target: pd.DataFrame
    target_column: str
    official_columns: tuple[str, ...]
    fast_columns: tuple[str, ...]
    priority_columns: tuple[str, ...]


@dataclass(frozen=True)
class DFMWindowData:
    monthly_endog: pd.DataFrame
    quarterly_endog: pd.DataFrame
    target_quarter: pd.Timestamp
    target_month: pd.Timestamp
    cutoff_month: pd.Timestamp
    selected_columns: tuple[str, ...]


def build_dfm_panel(source_data: dict[str, pd.DataFrame | None], config: BacktestConfig) -> MixedFrequencyPanel:
    monthly = source_data["monthly"].copy()
    gt_monthly = [_prefix_if_present(source_data[key], prefix) for key, prefix in _gt_prefixes()]
    official_groups = _prepare_monthly_groups(monthly, gt_monthly, None)
    official = official_groups["official"]
    fast = official_groups["fast"]

    official_selected = [col for col in REAL_ACTIVITY_PRIORITY + ARMSTAT_PRIORITY + FINANCIAL_PRIORITY if col in official.columns]
    fast_selected = [col for col in FAST_PRIORITY + SHOCK_PRIORITY if col in fast.columns]

    monthly_panel = pd.concat([official[official_selected], fast[fast_selected]], axis=1)
    monthly_panel = _to_month_end(monthly_panel)
    monthly_panel = monthly_panel.loc[:, ~monthly_panel.columns.duplicated()].sort_index()
    monthly_panel = monthly_panel.reindex(pd.date_range(monthly_panel.index.min(), monthly_panel.index.max(), freq="ME"))

    quarterly_target = source_data["quarterly"][[config.target_column]].dropna().copy()
    quarterly_target.index = pd.to_datetime(quarterly_target.index) + QuarterEnd(0)
    quarterly_target = quarterly_target.groupby(level=0).last().sort_index()
    quarterly_target = quarterly_target.reindex(
        pd.date_range(quarterly_target.index.min(), quarterly_target.index.max(), freq="QE-DEC")
    )

    priority_columns = [col for col in DFM_PRIORITY_ORDER if col in monthly_panel.columns]
    return MixedFrequencyPanel(
        monthly_panel=monthly_panel,
        quarterly_target=quarterly_target,
        target_column=config.target_column,
        official_columns=tuple(official_selected),
        fast_columns=tuple(fast_selected),
        priority_columns=tuple(priority_columns),
    )


def get_stage_panel(
    panel: MixedFrequencyPanel,
    target_quarter: pd.Timestamp,
    stage: str,
    train_end: pd.Timestamp,
    config: BacktestConfig,
) -> DFMWindowData:
    target_quarter = pd.Timestamp(target_quarter)
    target_month = quarter_target_month(target_quarter)
    cutoff_month = stage_cutoff_month(target_quarter, stage)
    train_quarter_end = pd.Timestamp(train_end) + QuarterEnd(0)
    history_start = max(
        panel.monthly_panel.index.min(),
        target_month - pd.DateOffset(months=config.dfm_history_months - 1),
    )

    monthly = panel.monthly_panel.loc[history_start:target_month].copy()
    month1 = target_quarter + MonthEnd(0)
    quarter_months = pd.date_range(month1, periods=3, freq="ME")
    monthly = _apply_stage_masks(monthly, quarter_months, stage, panel)
    monthly = _filter_monthly_panel(monthly, panel.priority_columns, config)

    quarterly_endog = panel.quarterly_target.loc[:train_quarter_end].copy()
    quarterly_endog = quarterly_endog.asfreq("QE-DEC")

    return DFMWindowData(
        monthly_endog=monthly,
        quarterly_endog=quarterly_endog,
        target_quarter=target_quarter,
        target_month=target_month,
        cutoff_month=cutoff_month,
        selected_columns=tuple(monthly.columns),
    )


def quarter_target_month(target_quarter: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(target_quarter) + pd.DateOffset(months=2) + MonthEnd(0)


def stage_cutoff_month(target_quarter: pd.Timestamp, stage: str) -> pd.Timestamp:
    month_offset = STAGE_MONTH_END_COUNTS[stage] - 1
    return pd.Timestamp(target_quarter) + pd.DateOffset(months=month_offset) + MonthEnd(0)


def _apply_stage_masks(
    monthly: pd.DataFrame,
    quarter_months: pd.DatetimeIndex,
    stage: str,
    panel: MixedFrequencyPanel,
) -> pd.DataFrame:
    monthly = monthly.copy()
    fast_available = set(quarter_months[: FAST_MONTHLY_COUNTS[stage]])
    official_available = set(quarter_months[: OFFICIAL_MONTHLY_COUNTS[stage]])

    fast_blocked = [month for month in quarter_months if month not in fast_available]
    official_blocked = [month for month in quarter_months if month not in official_available]

    if fast_blocked and panel.fast_columns:
        monthly.loc[fast_blocked, list(panel.fast_columns)] = pd.NA
    if official_blocked and panel.official_columns:
        monthly.loc[official_blocked, list(panel.official_columns)] = pd.NA
    return monthly


def _filter_monthly_panel(monthly: pd.DataFrame, priority_columns: tuple[str, ...], config: BacktestConfig) -> pd.DataFrame:
    if monthly.empty:
        return monthly

    non_missing = monthly.notna().sum()
    coverage = non_missing / len(monthly)
    varying = monthly.nunique(dropna=True) > 1
    eligible = [
        col
        for col in priority_columns
        if col in monthly.columns
        and non_missing.get(col, 0) >= config.dfm_min_monthly_observations
        and coverage.get(col, 0.0) >= config.dfm_min_monthly_coverage
        and varying.get(col, False)
    ]
    selected = eligible[: config.dfm_max_monthly_series]
    return monthly[selected].copy()


def _to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index) + MonthEnd(0)
    out = out.groupby(level=0).last()
    return out.sort_index()


def _gt_prefixes() -> list[tuple[str, str]]:
    return [
        ("gt_armenia_monthly", "GTG_"),
        ("gt_armenian_monthly", "GTL_"),
        ("gt_shock_monthly", "GTS_"),
    ]


def _prefix_if_present(df: pd.DataFrame | None, prefix: str) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    out.columns = [f"{prefix}{col}" for col in out.columns]
    return out
