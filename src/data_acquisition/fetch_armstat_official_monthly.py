from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pandas as pd
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "processed"
TMP_DIR = BASE_DIR / "results" / "armstat_downloads"

TABLE_SPECS = [
    {
        "code": "IC-ind-m-01",
        "url": "https://statbank.armstat.am/pxweb/en/ArmStatBank/ArmStatBank__3%20Industry,%20Construction,%20trade%20and%20services__32%20Industry__321%20Industry,%20RA__3212%20Monthly%20indicators/IC-ind-m-01.px/",
        "export_radio_suffix": "SaveAsRadioButtonList_5",
        "parser": "industry_matrix",
        "measure": "Current_Mln_AMD",
        "scale": 0.001,  # source is thousand drams
    },
    {
        "code": "IC-ind-m-02",
        "url": "https://statbank.armstat.am/pxweb/en/ArmStatBank/ArmStatBank__3%20Industry,%20Construction,%20trade%20and%20services__32%20Industry__321%20Industry,%20RA__3212%20Monthly%20indicators/IC-ind-m-02.px/",
        "export_radio_suffix": "SaveAsRadioButtonList_5",
        "parser": "industry_matrix",
        "measure": "Index",
        "scale": 1.0,
    },
    {
        "code": "EF-NA-01 D",
        "url": "https://statbank.armstat.am/pxweb/en/ArmStatBank/ArmStatBank__1%20Econnomy%20and%20finance__15%20National%20Accounts/EF-NA-01%20D.px/",
        "export_radio_suffix": "SaveAsRadioButtonList_5",
        "parser": "measure_columns",
        "measure_columns": {
            "Chain-link indexes (2023=100)": "ArmStat_EAI_ChainLink_2023_Index",
            "Chain-link indexes with seasonal adjustment, % (2023=100)": "ArmStat_EAI_SA_2023_Index",
        },
    },
    {
        "code": "EF-PP-FTPI-EXP-1",
        "url": "https://statbank.armstat.am/pxweb/en/ArmStatBank/ArmStatBank__1%20Econnomy%20and%20finance__16%20Producer%20Prices__162%20FTPI__1621%20FTPI-EXP/EF-PP-FTPI-EXP-1.px/",
        "export_radio_suffix": "SaveAsRadioButtonList_5",
        "parser": "year_month_matrix",
        "value_name": "ArmStat_Export_UnitValue_MoM_Index",
    },
    {
        "code": "EF-PP-FTPI-IMP-1",
        "url": "https://statbank.armstat.am/pxweb/en/ArmStatBank/ArmStatBank__1%20Econnomy%20and%20finance__16%20Producer%20Prices__162%20FTPI__1622%20FTPI-IMP/EF-PP-FTPI-IMP-1.px/",
        "export_radio_suffix": "SaveAsRadioButtonList_5",
        "parser": "year_month_matrix",
        "value_name": "ArmStat_Import_UnitValue_MoM_Index",
    },
    {
        "code": "EF-PP-FTI-1",
        "url": "https://statbank.armstat.am/pxweb/en/ArmStatBank/ArmStatBank__1%20Econnomy%20and%20finance__16%20Producer%20Prices__163%20FTI/EF-PP-FTI-1.px/",
        "export_radio_suffix": "SaveAsRadioButtonList_5",
        "parser": "indicator_year_month_matrix",
        "indicator_aliases": {
            "FREIGHT TARIFFS INDEX, TOTAL": "ArmStat_FreightTariff_Total_MoM_Index",
            "BY ROAD": "ArmStat_FreightTariff_Road_MoM_Index",
            "BY RAILWAY": "ArmStat_FreightTariff_Rail_MoM_Index",
        },
    },
]

ROW_ALIASES = {
    "TOTAL INDUSTRY": "Total",
    "B. MINING AND QUARRYING": "Mining",
    "07. MINING OF METAL ORES": "MetalOres",
    "C. MANUFACTURING": "Manufacturing",
    "D. ELECTRICITY, GAS, STEAM AND AIR CONDITIONING SUPPLY": "ElectricityGas",
    "E. WATER SUPPLY;SEWERAGE, WASTE MANAGEMENT AND REMEDIATION ACTIVITIES": "WaterWaste",
}

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _normalize_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label).strip().upper()
    label = label.replace("; ", ";")
    return label


def _parse_period(column_name: str) -> pd.Timestamp:
    cleaned = re.sub(r"\s+", " ", str(column_name)).strip().lower()
    year_str, month_name = cleaned.split(" ", 1)
    month = MONTH_MAP[month_name.strip()]
    return pd.Timestamp(year=int(year_str), month=month, day=1)


def _load_exported_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=2)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "row_label"})
    df["row_label"] = df["row_label"].astype(str).map(_normalize_label)
    for col in df.columns[1:]:
        cleaned = df[col].replace({"..": pd.NA, "...": pd.NA, "-": pd.NA})
        non_null = cleaned.dropna().astype(str).str.strip()
        converted = pd.to_numeric(cleaned, errors="coerce")
        if non_null.empty or converted.notna().sum() >= len(non_null) * 0.8:
            df[col] = converted
        else:
            df[col] = cleaned
    return df


def _reshape_table(df: pd.DataFrame, measure: str, scale: float) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    label_lookup = {label: alias for label, alias in ROW_ALIASES.items()}
    selected = df[df["row_label"].isin(label_lookup)].copy()

    for _, row in selected.iterrows():
        alias = label_lookup[row["row_label"]]
        series = row.iloc[1:].copy()
        series.index = pd.to_datetime([_parse_period(col) for col in series.index])
        series = series.sort_index().astype(float) * scale
        data[f"ArmStat_Industry_{alias}_{measure}"] = series

    return pd.DataFrame(data).sort_index()


def _reshape_measure_columns(df: pd.DataFrame, measure_columns: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    year_col = out.columns[0]
    month_col = out.columns[1]
    out["date"] = pd.to_datetime(
        out[year_col].astype(str).str.strip() + "-" + out[month_col].astype(str).str.strip(),
        format="%Y-%B",
        errors="coerce",
    )
    out = out.dropna(subset=["date"]).set_index("date").sort_index()
    data = {}
    for source, target in measure_columns.items():
        if source in out.columns:
            data[target] = pd.to_numeric(out[source], errors="coerce")
    return pd.DataFrame(data).sort_index()


def _reshape_year_month_matrix(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    year_col = df.columns[0]
    melted = df.melt(id_vars=[year_col], var_name="month_name", value_name=value_name)
    melted["date"] = pd.to_datetime(
        melted[year_col].astype(str).str.strip() + "-" + melted["month_name"].astype(str).str.strip(),
        format="%Y-%B",
        errors="coerce",
    )
    melted = melted.dropna(subset=["date"]).set_index("date").sort_index()
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    return melted[[value_name]]


def _reshape_indicator_year_month_matrix(df: pd.DataFrame, indicator_aliases: dict[str, str]) -> pd.DataFrame:
    year_col = df.columns[0]
    indicator_col = df.columns[1]
    out: dict[str, pd.Series] = {}
    df = df.copy()
    df[indicator_col] = df[indicator_col].astype(str).map(_normalize_label)
    for label, target in indicator_aliases.items():
        subset = df[df[indicator_col] == label]
        if subset.empty:
            continue
        series = _reshape_year_month_matrix(subset.drop(columns=[indicator_col]), target)[target]
        out[target] = series
    return pd.DataFrame(out).sort_index()


async def _download_table_csv(page, url: str, radio_suffix: str, output_path: Path) -> None:
    for attempt in range(5):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=120000)
            await page.wait_for_timeout(1500)
            selectors = await page.locator('input[id$="VariableValueSelect_VariableValueSelect_SelectAllButton"]').all()
            for button in selectors:
                if await button.is_visible():
                    await button.click()
                    await page.wait_for_timeout(350)

            await page.click('input[id$="ButtonViewTable"]')
            await page.wait_for_load_state("domcontentloaded", timeout=120000)
            await page.wait_for_timeout(1500)
            await page.click("#SaveAsHeaderButton")
            await page.wait_for_timeout(500)
            await page.check(f'input[id$="{radio_suffix}"]')
            await page.wait_for_timeout(300)
            async with page.expect_download(timeout=120000) as download_info:
                await page.click('input[id$="CommandBar1_CommandBar1_SaveAsBtn"]')
            download = await download_info.value
            await download.save_as(str(output_path))
            return
        except PlaywrightTimeoutError:
            if attempt == 4:
                raise
            await page.wait_for_timeout(2000 * (attempt + 1))


async def _run_downloads() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True, viewport={"width": 1600, "height": 2200})
        page = await context.new_page()
        try:
            for spec in TABLE_SPECS:
                output_path = TMP_DIR / f"{spec['code']}.csv"
                print(f"Downloading ArmStat export for {spec['code']}...")
                await _download_table_csv(page, spec["url"], spec["export_radio_suffix"], output_path)
        finally:
            await browser.close()


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run_downloads())

    frames: list[pd.DataFrame] = []
    for spec in TABLE_SPECS:
        raw_path = TMP_DIR / f"{spec['code']}.csv"
        raw_df = _load_exported_csv(raw_path)
        if spec["parser"] == "industry_matrix":
            frame = _reshape_table(raw_df, spec["measure"], spec["scale"])
        elif spec["parser"] == "measure_columns":
            frame = _reshape_measure_columns(raw_df, spec["measure_columns"])
        elif spec["parser"] == "year_month_matrix":
            frame = _reshape_year_month_matrix(raw_df, spec["value_name"])
        elif spec["parser"] == "indicator_year_month_matrix":
            frame = _reshape_indicator_year_month_matrix(raw_df, spec["indicator_aliases"])
        else:
            raise ValueError(f"Unknown parser: {spec['parser']}")
        frames.append(frame)
        print(f"Parsed {spec['code']} -> {frame.shape[1]} curated series")

    monthly = pd.concat(frames, axis=1).sort_index()
    quarterly = monthly.resample("QS").mean()
    monthly.to_csv(PROC_DIR / "armstat_nowcast_extension_monthly.csv", index_label="date")
    quarterly.to_csv(PROC_DIR / "armstat_nowcast_extension_quarterly.csv", index_label="date")

    print(f"Saved {monthly.shape[1]} ArmStat monthly series to {PROC_DIR / 'armstat_nowcast_extension_monthly.csv'}")
    print(f"Coverage: {monthly.index.min().date()} -> {monthly.index.max().date()}")


if __name__ == "__main__":
    main()
