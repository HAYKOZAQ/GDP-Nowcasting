from __future__ import annotations

from dataclasses import dataclass
import unicodedata

import numpy as np
import pandas as pd

from src.data.nowcast_dataset import FAST_MONTHLY_COUNTS, OFFICIAL_MONTHLY_COUNTS, STAGES


ALMON_WEIGHTS = np.array([0.21194156, 0.57611688, 0.21194156])


@dataclass
class StageDataset:
    name: str
    frame: pd.DataFrame
    feature_groups: dict[str, list[str]]


def _prefix_columns(df: pd.DataFrame | None, prefix: str) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    out.columns = [f"{prefix}{col}" for col in out.columns]
    return out


def build_stage_datasets(source_data: dict[str, pd.DataFrame | None], target_column: str) -> dict[str, StageDataset]:
    quarterly = source_data["quarterly"].copy()
    monthly = source_data["monthly"].copy()
    worldbank = _prefix_columns(source_data["worldbank_quarterly"], "WB_")
    fintech_alt = _prefix_columns(source_data.get("fintech_alt_quarterly"), "ALT_")

    gt_quarterly = [
        _prefix_columns(source_data["gt_armenia_quarterly"], "GTG_"),
        _prefix_columns(source_data["gt_armenian_quarterly"], "GTL_"),
        _prefix_columns(source_data["gt_shock_quarterly"], "GTS_"),
    ]
    gt_monthly = [
        _prefix_columns(source_data["gt_armenia_monthly"], "GTG_"),
        _prefix_columns(source_data["gt_armenian_monthly"], "GTL_"),
        _prefix_columns(source_data["gt_shock_monthly"], "GTS_"),
    ]
    wiki_quarterly = _prefix_columns(source_data["wiki_quarterly"], "WIKI_")
    wiki_monthly = _prefix_columns(source_data["wiki_monthly"], "WIKI_")

    quarterly = _engineer_quarterly_features(quarterly, worldbank, fintech_alt, gt_quarterly, wiki_quarterly)
    monthly_groups = _prepare_monthly_groups(monthly, gt_monthly, wiki_monthly)

    return {
        stage: StageDataset(
            name=stage,
            frame=_build_stage_frame(quarterly, monthly_groups, stage, target_column),
            feature_groups={},
        )
        for stage in STAGES
    }


def _engineer_quarterly_features(
    quarterly: pd.DataFrame,
    worldbank: pd.DataFrame | None,
    fintech_alt: pd.DataFrame | None,
    gt_quarterly: list[pd.DataFrame | None],
    wiki_quarterly: pd.DataFrame | None,
) -> pd.DataFrame:
    quarterly = quarterly.copy()
    quarterly["Primary_Income_YoY"] = quarterly["Primary_Income_Labor_Mln_USD"].pct_change(4, fill_method=None) * 100
    quarterly["Secondary_Income_YoY"] = quarterly["Secondary_Income_Transfers_Mln_USD"].pct_change(
        4, fill_method=None
    ) * 100
    quarterly["AMD_USD_StrongSignal"] = 100 - quarterly["Exchange_Rate_AMD_USD_YoY"]
    quarterly["AMD_RUB_QoQ"] = quarterly["Exchange_Rate_AMD_RUB_Abs"].pct_change(fill_method=None) * 100
    quarterly["Migration_Inflow_Signal"] = (
        quarterly["AMD_USD_StrongSignal"].clip(lower=0) * quarterly["Primary_Income_YoY"].clip(lower=0)
    ) / 100
    quarterly["REER_Surge"] = (quarterly["REER_YoY"] - 100).clip(lower=0)
    quarterly["RU_ARM_Growth_Gap"] = quarterly["Real_GDP_Armenia_YoY"] - quarterly["Real_GDP_Russia_YoY"]
    quarterly["RU_Oil_Link"] = quarterly["Real_GDP_Russia_YoY"] * quarterly["Brent_Oil_Price_USD_bbl"]
    quarterly["RU_AMD_RUB_Link"] = quarterly["Real_GDP_Russia_YoY"] * quarterly["Exchange_Rate_AMD_RUB_Abs"]
    quarterly["External_Stress_Signal"] = (
        quarterly["Exchange_Rate_AMD_USD_YoY"].abs()
        + quarterly["Exchange_Rate_AMD_RUB_Abs"].pct_change(fill_method=None).mul(100).abs()
        + quarterly["Brent_Oil_Price_USD_bbl"].pct_change(fill_method=None).mul(100).abs()
    ) / 3
    _add_quarterly_momentum_features(
        quarterly,
        [
            "Real_GDP_Armenia_YoY",
            "Real_GDP_Russia_YoY",
            "Real_Private_Consumption_YoY",
            "Real_Aggregate_Investments_YoY",
            "Employment_YoY",
            "Primary_Income_YoY",
        ],
    )

    if worldbank is not None:
        quarterly = quarterly.merge(worldbank, left_index=True, right_index=True, how="left")
    if fintech_alt is not None:
        quarterly = quarterly.merge(fintech_alt, left_index=True, right_index=True, how="left")
    for item in gt_quarterly:
        if item is not None:
            quarterly = quarterly.merge(item, left_index=True, right_index=True, how="left")
    if wiki_quarterly is not None:
        quarterly = quarterly.merge(wiki_quarterly, left_index=True, right_index=True, how="left")

    return quarterly.sort_index()


def _add_quarterly_momentum_features(quarterly: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in quarterly.columns:
            continue
        quarterly[f"{col}_DIFF1"] = quarterly[col].diff(1)
        quarterly[f"{col}_DIFF2"] = quarterly[col].diff(2)
        quarterly[f"{col}_ACCEL"] = quarterly[col].diff(1) - quarterly[col].diff(2)


def _prepare_monthly_groups(
    monthly: pd.DataFrame,
    gt_monthly: list[pd.DataFrame | None],
    wiki_monthly: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    monthly = monthly.copy()

    official_monthly_cols = [
        col
        for col in [
            "CPI_YoY",
            "Economic_Activity_Index_Discrete_YoY",
            "Industry_Real_Growth_YoY",
            "Construction_Real_Growth_YoY",
            "Services_Real_Growth_YoY",
            "Private_Construction_Real_Growth_YoY",
            "Hired_Workers",
            "Short_Term_Nominal_Interest_Rate_Loans_AMD",
            "Short_Term_Nominal_Interest_Rate_Loans_USD",
            "Short_Term_Nominal_Interest_Rate_Deposits_AMD",
            "Short_Term_Nominal_Interest_Rate_Deposits_USD",
            "Long_Term_Nominal_Interest_Rate_Loans_AMD",
            "Long_Term_Nominal_Interest_Rate_Loans_USD",
            "Long_Term_Nominal_Interest_Rate_Deposits_AMD",
            "Long_Term_Nominal_Interest_Rate_Deposits_USD",
            "Cash_in_Circulation_Mln_AMD",
            "Money_Supply_M2_Mln_AMD",
            "Money_Supply_M2X_Mln_AMD",
            "Demand_Deposits_Dram_Mln_AMD",
            "Time_Deposits_Dram_Mln_AMD",
            "Deposits_FX_Mln_AMD",
            "Commercial_Bank_Loans_Mln_AMD",
            "Enterprise_Loans_Mln_AMD",
            "Private_Enterprise_Loans_Mln_AMD",
            "Household_Loans_Mln_AMD",
            "Total_Loans_Mln_AMD",
            "Loans_Industry_Mln_AMD",
            "Loans_Agriculture_Mln_AMD",
            "Loans_Construction_Mln_AMD",
            "Loans_Transport_Communication_Mln_AMD",
            "Loans_Trade_Mln_AMD",
            "Loans_Services_Mln_AMD",
            "Consumer_Loans_Mln_AMD",
            "Mortgage_Loans_Mln_AMD",
            "Other_Loans_Mln_AMD",
            "Loans_Residents_Banks_Mln_AMD",
            "Loans_Residents_Credit_Orgs_Mln_AMD",
            "Remittance_Inflow_Mln_AMD",
            "Remittance_Outflow_Mln_AMD",
            "Remittance_Net_Mln_AMD",
            "Industry_Output_Mln_AMD",
            "Construction_Output_Mln_AMD",
            "Services_Output_Mln_AMD",
        ]
        if col in monthly.columns
    ]
    official_monthly_cols.extend(sorted(col for col in monthly.columns if col.startswith("ArmStat_")))
    fast_monthly_cols = [
        col
        for col in ["Exchange_Rate_AMD_USD", "Exchange_Rate_AMD_RUB", "Brent_Oil_Price_USD_bbl", "Copper_Price_USD_mt"]
        if col in monthly.columns
    ]

    official = monthly[official_monthly_cols].copy()
    for col in [c for c in official.columns if c.endswith("_Mln_AMD")]:
        official[f"{col}_YoY"] = official[col].pct_change(12, fill_method=None) * 100
        official[f"{col}_QoQ"] = official[col].pct_change(3, fill_method=None) * 100

    fast = monthly[fast_monthly_cols].copy()
    for item in gt_monthly:
        if item is not None:
            fast = fast.join(item, how="outer")
    if wiki_monthly is not None:
        fast = fast.join(wiki_monthly, how="outer")
    fast = _add_external_leading_features(fast)
    fast = _add_google_composites(fast)

    return {"official": official.sort_index(), "fast": fast.sort_index()}


def _build_stage_frame(
    quarterly: pd.DataFrame,
    monthly_groups: dict[str, pd.DataFrame],
    stage: str,
    target_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    quarterly_feature_cols = _quarterly_feature_columns(quarterly, target_column)
    official_monthly = monthly_groups["official"]

    for idx, row in quarterly.iterrows():
        row_data: dict[str, float | str | pd.Timestamp] = {
            "prediction_date": idx,
            "target_quarter": f"{idx.year}-Q{idx.quarter}",
            "stage": stage,
            "target": row[target_column],
        }
        for lag in (1, 2, 4):
            prev_idx = idx - pd.DateOffset(months=3 * lag)
            row_data[f"AR_LAG{lag}"] = quarterly[target_column].get(prev_idx, np.nan)

        prev_quarter = idx - pd.DateOffset(months=3)
        for col in quarterly_feature_cols:
            row_data[f"Q_{col}"] = quarterly[col].get(prev_quarter, np.nan)

        quarter_months = pd.date_range(start=idx, periods=3, freq="MS")
        row_data.update(_aggregate_stage_monthly(monthly_groups["fast"], quarter_months, FAST_MONTHLY_COUNTS[stage], "FAST"))
        row_data.update(_aggregate_stage_monthly(official_monthly, quarter_months, OFFICIAL_MONTHLY_COUNTS[stage], "OFF"))
        row_data.update(_current_quarter_dummies(idx, row_data))
        rows.append(row_data)

    return pd.DataFrame(rows).set_index("prediction_date").sort_index()


def _quarterly_feature_columns(quarterly: pd.DataFrame, target_column: str) -> list[str]:
    excluded = {
        target_column,
        "Real_GDP_Armenia_Abs",
        "Nominal_GDP_Mln_AMD",
        "Real_Private_Consumption_Abs",
        "Real_Private_Investments_Abs",
        "Real_Construction_Abs",
        "Primary_Income_Mln_AMD",
        "Secondary_Income_Mln_AMD",
        "Disposable_Income_Mln_AMD",
    }
    return [col for col in quarterly.columns if col not in excluded]


def _aggregate_stage_monthly(
    monthly_df: pd.DataFrame,
    quarter_months: pd.DatetimeIndex,
    available_count: int,
    prefix: str,
) -> dict[str, float]:
    row: dict[str, float] = {}
    if available_count <= 0 or monthly_df.empty:
        for col in monthly_df.columns:
            row[f"{prefix}_{col}_MEAN"] = np.nan
            row[f"{prefix}_{col}_LAST"] = np.nan
            row[f"{prefix}_{col}_ALMON"] = np.nan
        return row

    available_months = quarter_months[:available_count]
    subset = monthly_df.reindex(available_months)
    for col in monthly_df.columns:
        feature_base = col if col.startswith(f"{prefix}_") else f"{prefix}_{col}"
        vals = subset[col].to_numpy(dtype=float)
        row[f"{feature_base}_MEAN"] = np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan
        row[f"{feature_base}_LAST"] = subset[col].iloc[-1] if available_count > 0 else np.nan
        if available_count == 3:
            row[f"{feature_base}_ALMON"] = np.nansum(ALMON_WEIGHTS * vals) if not np.all(np.isnan(vals)) else np.nan
        elif available_count == 2:
            row[f"{feature_base}_ALMON"] = np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan
        else:
            row[f"{feature_base}_ALMON"] = vals[0] if len(vals) else np.nan
    return row


def _current_quarter_dummies(idx: pd.Timestamp, row_data: dict[str, float | str | pd.Timestamp]) -> dict[str, int]:
    def _value(name: str) -> float:
        value = row_data.get(name, np.nan)
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    activity_last = _value("OFF_Economic_Activity_Index_Discrete_YoY_LAST")
    remittance_yoy_last = _value("OFF_Remittance_Net_Mln_AMD_YoY_LAST")
    if np.isnan(remittance_yoy_last):
        remittance_yoy_last = _value("OFF_Remittance_Inflow_Mln_AMD_YoY_LAST")
    shock_composite_last = _value("FAST_SHOCK_COMPOSITE_LAST")

    return {
        "CURR_Dummy_Q1": int(idx.quarter == 1),
        "CURR_Dummy_Q2": int(idx.quarter == 2),
        "CURR_Dummy_Q3": int(idx.quarter == 3),
        "CURR_Dummy_GFC": int(pd.Timestamp("2008-10-01") <= idx <= pd.Timestamp("2010-06-30")),
        "CURR_Dummy_COVID": int(pd.Timestamp("2020-01-01") <= idx <= pd.Timestamp("2021-06-30")),
        "CURR_Dummy_COVID_LOCKDOWN": int(idx == pd.Timestamp("2020-04-01")),
        "CURR_Dummy_COVID_REOPEN": int(pd.Timestamp("2020-07-01") <= idx <= pd.Timestamp("2021-06-30")),
        "CURR_Dummy_RU_WAR": int(pd.Timestamp("2022-01-01") <= idx <= pd.Timestamp("2023-06-30")),
        "CURR_Dummy_WAR_ONSET": int(pd.Timestamp("2022-01-01") <= idx <= pd.Timestamp("2022-06-30")),
        "CURR_Dummy_RELOCATION_BOOM": int(pd.Timestamp("2022-04-01") <= idx <= pd.Timestamp("2023-03-31")),
        "CURR_Dummy_RELOCATION_NORMALIZE": int(pd.Timestamp("2023-04-01") <= idx <= pd.Timestamp("2024-03-31")),
        "CURR_Dummy_ACTIVITY_CRASH": int(not np.isnan(activity_last) and activity_last <= -5.0),
        "CURR_Dummy_REMITTANCE_SURGE": int(not np.isnan(remittance_yoy_last) and remittance_yoy_last >= 20.0),
        "CURR_Dummy_GOOGLE_SHOCK": int(not np.isnan(shock_composite_last) and shock_composite_last >= 55.0),
    }


def _add_google_composites(fast: pd.DataFrame) -> pd.DataFrame:
    if fast.empty:
        return fast

    out = fast.copy()
    normalized_lookup = {_normalize_google_label(str(col)): col for col in out.columns}
    groups = {
        "FAST_SHOCK_HOUSING": [
            "բնակարան",
            "հիփոթեք",
            "անշարժ գույք",
            "շինարարություն",
            "apartment rent yerevan",
            "yerevan apartment",
            "квартира ереван",
            "снять квартиру ереван",
            "купить квартиру ереван",
            "аренда ереван",
            "ипотека армения",
            "estate.am",
        ],
        "FAST_SHOCK_RELOCATION": [
            "armenia relocation",
            "relocation armenia",
            "move to armenia",
            "live in armenia",
            "visa armenia",
            "переезд армения",
            "переехать армения",
            "виза армения",
            "внж армения",
            "жить армения",
            "гражданство армения",
            "регистрация ереван",
            "овир армения",
            "загранпаспорт армения",
        ],
        "FAST_SHOCK_BANKING": [
            "վարկ",
            "բանկային վարկ",
            "փոխանցում",
            "փոխարժեք",
            "money transfer armenia",
            "armenian dram",
            "usd amd",
            "банк армения",
            "открыть счет армения",
            "moneygram",
            "rate.am",
            "idram",
            "telcell",
            "dolari kurs",
        ],
        "FAST_SHOCK_JOBS_IT": [
            "աշխատանք",
            "работа армения",
            "работа ереван",
            "работа в ереване",
            "работа в армении",
            "staff.am",
            "tapvur ashkhatatagekh",
            "ит армения",
        ],
        "FAST_GOOGLE_MOBILITY": [
            "զբոսաշրջություն",
            "ереван",
            "armenia tourism",
            "turizm",
            "zvartnots",
            "aviasales",
            "booking.com",
        ],
        "FAST_GOOGLE_CONSUMPTION": [
            "wildberries",
            "iphone",
            "list.am",
            "auto.am",
            "mekena",
        ],
    }

    for new_col, labels in groups.items():
        cols = [normalized_lookup[label] for label in labels if label in normalized_lookup]
        if not cols:
            continue
        normalized_cols = [_normalize_google_signal(pd.to_numeric(out[col], errors="coerce")) for col in cols]
        out[new_col] = pd.concat(normalized_cols, axis=1).mean(axis=1, skipna=True)

    shock_cols = [
        col
        for col in ("FAST_SHOCK_HOUSING", "FAST_SHOCK_RELOCATION", "FAST_SHOCK_BANKING", "FAST_SHOCK_JOBS_IT")
        if col in out.columns
    ]
    broad_cols = [col for col in ("FAST_GOOGLE_MOBILITY", "FAST_GOOGLE_CONSUMPTION") if col in out.columns]
    if shock_cols:
        out["FAST_SHOCK_COMPOSITE"] = out[shock_cols].mean(axis=1, skipna=True)
    if broad_cols:
        out["FAST_GOOGLE_COMPOSITE"] = out[broad_cols].mean(axis=1, skipna=True)
    if shock_cols or broad_cols:
        out["FAST_GOOGLE_ALL"] = out[shock_cols + broad_cols].mean(axis=1, skipna=True)
    return out


def _normalize_google_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = " ".join(normalized.split())
    for prefix in ("gtg_", "gtl_", "gts_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


def _normalize_google_signal(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").replace(0, np.nan)
    history_mean = series.expanding(min_periods=6).mean().shift(1)
    history_std = series.expanding(min_periods=6).std(ddof=0).shift(1)
    z_score = (series - history_mean) / history_std.replace(0, np.nan)

    history_median = series.expanding(min_periods=6).median().shift(1)
    centered = series - history_median
    history_mad = series.expanding(min_periods=6).apply(
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))),
        raw=True,
    ).shift(1)
    robust_score = centered / history_mad.replace(0, np.nan)

    combined = z_score.where(z_score.notna(), robust_score)
    return combined.clip(lower=-4.0, upper=4.0)


def _add_external_leading_features(fast: pd.DataFrame) -> pd.DataFrame:
    out = fast.copy()
    if "Exchange_Rate_AMD_RUB" in out.columns and "Brent_Oil_Price_USD_bbl" in out.columns:
        out["RUS_LINK_OIL_RUB"] = out["Exchange_Rate_AMD_RUB"] * out["Brent_Oil_Price_USD_bbl"]
    if "Exchange_Rate_AMD_RUB" in out.columns and "Copper_Price_USD_mt" in out.columns:
        out["RUS_LINK_COPPER_RUB"] = out["Exchange_Rate_AMD_RUB"] * out["Copper_Price_USD_mt"]
    if "Exchange_Rate_AMD_RUB" in out.columns:
        out["RUS_LINK_RUB_STRESS"] = out["Exchange_Rate_AMD_RUB"].pct_change(fill_method=None).mul(100).abs()
    if "Exchange_Rate_AMD_USD" in out.columns and "Exchange_Rate_AMD_RUB" in out.columns:
        out["FIN_STRESS_PROXY"] = (
            out["Exchange_Rate_AMD_USD"].pct_change(fill_method=None).mul(100).abs()
            + out["Exchange_Rate_AMD_RUB"].pct_change(fill_method=None).mul(100).abs()
        ) / 2
    return out
