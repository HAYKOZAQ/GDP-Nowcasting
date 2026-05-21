from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
import re


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_XLSX = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.xlsx"
PROC_DIR = BASE_DIR / "data" / "processed"


def _fetch_monthly_market_q1_2026() -> pd.DataFrame:
    daily = {}
    tickers = {
        "Exchange_Rate_AMD_USD": "USDAMD=X",
        "USDRUB": "RUB=X",
        "Brent_Oil_Price_USD_bbl": "BZ=F",
        "Copper_raw": "HG=F",
    }
    for name, ticker in tickers.items():
        hist = yf.Ticker(ticker).history(start="2026-01-01", end="2026-04-01", interval="1d")
        if hist.empty:
            daily[name] = pd.Series(dtype=float)
            continue
        series = hist["Close"].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        daily[name] = series

    monthly = pd.DataFrame(index=pd.date_range("2026-01-01", "2026-03-01", freq="MS"))
    if not daily["Exchange_Rate_AMD_USD"].empty:
        monthly["Exchange_Rate_AMD_USD"] = daily["Exchange_Rate_AMD_USD"].resample("MS").mean().reindex(monthly.index)
    if not daily["USDRUB"].empty and "Exchange_Rate_AMD_USD" in monthly:
        usdrub = daily["USDRUB"].resample("MS").mean().reindex(monthly.index)
        monthly["Exchange_Rate_AMD_RUB"] = monthly["Exchange_Rate_AMD_USD"] / usdrub
    if not daily["Brent_Oil_Price_USD_bbl"].empty:
        monthly["Brent_Oil_Price_USD_bbl"] = (
            daily["Brent_Oil_Price_USD_bbl"].resample("MS").mean().reindex(monthly.index)
        )
    if not daily["Copper_raw"].empty:
        monthly["Copper_Price_USD_mt"] = daily["Copper_raw"].resample("MS").mean().reindex(monthly.index) * 2204.62262185
    monthly.index.name = "Date"
    monthly = _fill_fx_from_cba_archive(monthly)
    return monthly


def _fill_fx_from_cba_archive(monthly: pd.DataFrame) -> pd.DataFrame:
    daily_records = []
    for day in pd.date_range("2026-01-01", "2026-03-31", freq="B"):
        url = f"https://old.cba.am/en/sitepages/exchangearchive.aspx?FilterDate={day.strftime('%Y-%m-%d')}"
        try:
            html = requests.get(url, timeout=20).text
        except Exception:
            continue
        usd_match = re.search(r"<span>USD</span><em>(\d+)</em></td><td>([0-9.]+)</td>", html)
        rub_match = re.search(r"<span>RUB</span><em>(\d+)</em></td><td>([0-9.]+)</td>", html)
        if usd_match:
            daily_records.append(
                {
                    "Date": day,
                    "Exchange_Rate_AMD_USD": float(usd_match.group(2)) / float(usd_match.group(1)),
                    "Exchange_Rate_AMD_RUB": (
                        float(rub_match.group(2)) / float(rub_match.group(1)) if rub_match else pd.NA
                    ),
                }
            )
    if not daily_records:
        return monthly
    daily = pd.DataFrame(daily_records).set_index("Date")
    monthly_fx = daily.resample("MS").mean().reindex(monthly.index)
    for col in ["Exchange_Rate_AMD_USD", "Exchange_Rate_AMD_RUB"]:
        monthly[col] = monthly.get(col, pd.Series(index=monthly.index, dtype=float)).combine_first(monthly_fx[col])
    return monthly


def _merge_monthly_extensions(df_m: pd.DataFrame) -> pd.DataFrame:
    df_m = df_m.copy().set_index("Date").sort_index()
    for filename in ("armstat_nowcast_extension_monthly.csv", "cba_nowcast_extension_monthly.csv"):
        path = PROC_DIR / filename
        if not path.exists():
            continue
        ext = pd.read_csv(path, parse_dates=["date"]).rename(columns={"date": "Date"}).set_index("Date").sort_index()
        df_m = df_m.combine_first(ext)

    market = _fetch_monthly_market_q1_2026()
    for col in market.columns:
        if col not in df_m.columns:
            df_m[col] = pd.NA
        df_m.loc[market.index, col] = df_m.loc[market.index, col].combine_first(market[col])

    return df_m.sort_index().reset_index()


def _upsert_quarterly_q1_2026(df_q: pd.DataFrame, df_m: pd.DataFrame) -> pd.DataFrame:
    df_q = df_q.copy().set_index("Date").sort_index()
    q_date = pd.Timestamp("2026-01-01")
    if q_date not in df_q.index:
        df_q.loc[q_date] = pd.Series(dtype=float)

    monthly = df_m.copy().set_index("Date").sort_index()
    q_months = pd.date_range(q_date, periods=3, freq="MS")
    if set(q_months).issubset(set(monthly.index)):
        for src_col, q_col in (
            ("Exchange_Rate_AMD_USD", "Exchange_Rate_AMD_USD_Abs"),
            ("Exchange_Rate_AMD_RUB", "Exchange_Rate_AMD_RUB_Abs"),
            ("Brent_Oil_Price_USD_bbl", "Brent_Oil_Price_USD_bbl"),
            ("Copper_Price_USD_mt", "Copper_Price_USD_mt"),
        ):
            if src_col in monthly.columns and q_col in df_q.columns:
                value = monthly.loc[q_months, src_col].mean()
                if pd.notna(value):
                    df_q.at[q_date, q_col] = value

        if "Exchange_Rate_AMD_USD_YoY" in df_q.columns and "Exchange_Rate_AMD_USD" in monthly.columns:
            prev_q = pd.date_range("2025-01-01", periods=3, freq="MS")
            curr = monthly.loc[q_months, "Exchange_Rate_AMD_USD"].mean()
            prev = monthly.loc[prev_q, "Exchange_Rate_AMD_USD"].mean() if set(prev_q).issubset(set(monthly.index)) else pd.NA
            if pd.notna(curr) and pd.notna(prev) and prev != 0:
                df_q.at[q_date, "Exchange_Rate_AMD_USD_YoY"] = curr / prev * 100

    return df_q.sort_index().reset_index()


def main() -> None:
    xls = pd.ExcelFile(RAW_XLSX)
    df_q = pd.read_excel(xls, sheet_name="Quarterly")
    df_m = pd.read_excel(xls, sheet_name="Monthly")
    df_q["Date"] = pd.to_datetime(df_q["Date"])
    df_m["Date"] = pd.to_datetime(df_m["Date"])

    df_m = _merge_monthly_extensions(df_m)
    df_q = _upsert_quarterly_q1_2026(df_q, df_m)

    with pd.ExcelWriter(RAW_XLSX, engine="openpyxl") as writer:
        df_q.to_excel(writer, sheet_name="Quarterly", index=False)
        df_m.to_excel(writer, sheet_name="Monthly", index=False)

    q1_monthly = df_m[(df_m["Date"] >= "2026-01-01") & (df_m["Date"] <= "2026-03-01")]
    q1_quarterly = df_q[df_q["Date"] == pd.Timestamp("2026-01-01")]
    print("Workbook updated for available 2026 Q1 data.")
    print("Monthly rows:")
    print(q1_monthly[["Date", "Exchange_Rate_AMD_USD", "Exchange_Rate_AMD_RUB", "Brent_Oil_Price_USD_bbl", "Copper_Price_USD_mt"]].to_string(index=False))
    print("\nQuarterly row:")
    cols = [c for c in ["Date", "Real_GDP_Armenia_YoY", "Real_GDP_Russia_YoY", "Exchange_Rate_AMD_USD_YoY", "Exchange_Rate_AMD_USD_Abs", "Exchange_Rate_AMD_RUB_Abs", "Brent_Oil_Price_USD_bbl", "Copper_Price_USD_mt"] if c in q1_quarterly.columns]
    print(q1_quarterly[cols].to_string(index=False))


if __name__ == "__main__":
    main()
