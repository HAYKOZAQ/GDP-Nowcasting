from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import requests
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_XLSX = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.xlsx"
PROC_DIR = BASE_DIR / "data" / "processed"


def fetch_cba_monthly_fx(start: str = "2025-07-01", end: str = "2026-03-31") -> pd.DataFrame:
    records: list[dict[str, float | pd.Timestamp]] = []
    usd_pattern = re.compile(r"<span>USD</span><em>(\d+)</em></td><td>([0-9.]+)</td>")
    rub_pattern = re.compile(r"<span>RUB</span><em>(\d+)</em></td><td>([0-9.]+)</td>")

    for day in pd.date_range(start, end, freq="B"):
        url = f"https://old.cba.am/en/sitepages/exchangearchive.aspx?FilterDate={day.strftime('%Y-%m-%d')}"
        try:
            html = requests.get(url, timeout=20).text
        except Exception:
            continue
        usd = usd_pattern.search(html)
        rub = rub_pattern.search(html)
        if not usd:
            continue
        records.append(
            {
                "Date": day,
                "Exchange_Rate_AMD_USD": float(usd.group(2)) / float(usd.group(1)),
                "Exchange_Rate_AMD_RUB": float(rub.group(2)) / float(rub.group(1)) if rub else pd.NA,
            }
        )

    if not records:
        return pd.DataFrame(columns=["Exchange_Rate_AMD_USD", "Exchange_Rate_AMD_RUB"])

    daily = pd.DataFrame(records).set_index("Date").sort_index()
    monthly = daily.resample("MS").mean()
    monthly.index.name = "Date"
    return monthly


def fetch_market_monthly(start: str = "2025-07-01", end: str = "2026-03-31") -> pd.DataFrame:
    ticker_map = {
        "Brent_Oil_Price_USD_bbl": "BZ=F",
        "Copper_Price_USD_mt_raw": "HG=F",
    }
    out = pd.DataFrame(index=pd.date_range(start=start, end=end, freq="MS"))
    out.index.name = "Date"
    for col, ticker in ticker_map.items():
        hist = yf.Ticker(ticker).history(start=start, end=(pd.Timestamp(end) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"), interval="1d")
        if hist.empty:
            continue
        series = hist["Close"].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        out[col] = series.resample("MS").mean().reindex(out.index)

    if "Copper_Price_USD_mt_raw" in out.columns:
        out["Copper_Price_USD_mt"] = out["Copper_Price_USD_mt_raw"] * 2204.62262185
        out = out.drop(columns=["Copper_Price_USD_mt_raw"])
    return out


def load_extension(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"]).rename(columns={"date": "Date"})
    return df.set_index("Date").sort_index()


def recompute_quarterly_from_monthly(df_q: pd.DataFrame, df_m: pd.DataFrame) -> pd.DataFrame:
    quarterly = df_q.copy().set_index("Date").sort_index()
    monthly = df_m.copy().set_index("Date").sort_index()

    quarterly_cols = [
        ("Exchange_Rate_AMD_USD", "Exchange_Rate_AMD_USD_Abs"),
        ("Exchange_Rate_AMD_RUB", "Exchange_Rate_AMD_RUB_Abs"),
        ("Brent_Oil_Price_USD_bbl", "Brent_Oil_Price_USD_bbl"),
        ("Copper_Price_USD_mt", "Copper_Price_USD_mt"),
    ]

    for q_date in quarterly.index:
        q_months = pd.date_range(q_date, periods=3, freq="MS")
        if not set(q_months).issubset(monthly.index):
            continue
        for monthly_col, quarterly_col in quarterly_cols:
            if monthly_col not in monthly.columns or quarterly_col not in quarterly.columns:
                continue
            vals = monthly.loc[q_months, monthly_col]
            if vals.notna().all():
                quarterly.at[q_date, quarterly_col] = float(vals.mean())

        if (
            "Exchange_Rate_AMD_USD_YoY" in quarterly.columns
            and "Exchange_Rate_AMD_USD" in monthly.columns
        ):
            prev_months = pd.date_range(q_date - pd.DateOffset(years=1), periods=3, freq="MS")
            if set(prev_months).issubset(monthly.index):
                curr = monthly.loc[q_months, "Exchange_Rate_AMD_USD"]
                prev = monthly.loc[prev_months, "Exchange_Rate_AMD_USD"]
                if curr.notna().all() and prev.notna().all() and float(prev.mean()) != 0.0:
                    quarterly.at[q_date, "Exchange_Rate_AMD_USD_YoY"] = float(curr.mean() / prev.mean() * 100)

    return quarterly.reset_index()


def trim_trailing_empty_months(df_m: pd.DataFrame) -> pd.DataFrame:
    work = df_m.copy().sort_values("Date")
    non_empty = work.drop(columns=["Date"]).notna().any(axis=1)
    if not non_empty.any():
        return work
    last_idx = non_empty[non_empty].index.max()
    return work.loc[:last_idx].reset_index(drop=True)


def main() -> None:
    xls = pd.ExcelFile(RAW_XLSX)
    df_q = pd.read_excel(xls, sheet_name="Quarterly")
    df_m = pd.read_excel(xls, sheet_name="Monthly")
    df_q["Date"] = pd.to_datetime(df_q["Date"])
    df_m["Date"] = pd.to_datetime(df_m["Date"])

    before_q = df_q.copy()
    before_m = df_m.copy()

    df_m = df_m.set_index("Date").sort_index()

    armstat = load_extension(PROC_DIR / "armstat_nowcast_extension_monthly.csv")
    cba = load_extension(PROC_DIR / "cba_nowcast_extension_monthly.csv")
    fx = fetch_cba_monthly_fx()
    market = fetch_market_monthly()

    for ext in [armstat, cba]:
        if not ext.empty:
            for col in ext.columns:
                if col not in df_m.columns:
                    df_m[col] = pd.NA
            df_m = df_m.combine_first(ext)

    for ext in [fx, market]:
        if not ext.empty:
            for col in ext.columns:
                if col not in df_m.columns:
                    df_m[col] = pd.NA
                df_m.loc[ext.index, col] = ext[col]

    df_m = df_m.reset_index()
    df_m = trim_trailing_empty_months(df_m)
    df_q = recompute_quarterly_from_monthly(df_q, df_m)

    with pd.ExcelWriter(RAW_XLSX, engine="openpyxl") as writer:
        df_q.sort_values("Date").to_excel(writer, sheet_name="Quarterly", index=False)
        df_m.sort_values("Date").to_excel(writer, sheet_name="Monthly", index=False)

    # Audit summary
    q_new = df_q.set_index("Date").sort_index()
    m_new = df_m.set_index("Date").sort_index()
    q_old = before_q.set_index("Date").sort_index()
    m_old = before_m.set_index("Date").sort_index()

    print("Repair complete.")
    print("Monthly date range:", m_new.index.min().date(), "->", m_new.index.max().date())
    print("Quarterly date range:", q_new.index.min().date(), "->", q_new.index.max().date())

    print("\nChanged monthly key values:")
    for date in pd.date_range("2025-07-01", "2026-03-01", freq="MS"):
        if date not in m_new.index:
            continue
        for col in ["Exchange_Rate_AMD_USD", "Exchange_Rate_AMD_RUB", "Brent_Oil_Price_USD_bbl", "Copper_Price_USD_mt"]:
            old = q = None
            old = m_old[col].get(date) if col in m_old.columns else pd.NA
            new = m_new[col].get(date) if col in m_new.columns else pd.NA
            if (pd.isna(old) and pd.notna(new)) or (pd.notna(old) and pd.notna(new) and abs(float(old) - float(new)) > 1e-9):
                print(f"{date.date()} {col}: {old} -> {new}")

    print("\nQuarterly recent key values:")
    cols = [
        c
        for c in [
            "Real_GDP_Armenia_YoY",
            "Real_GDP_Russia_YoY",
            "Exchange_Rate_AMD_USD_YoY",
            "Exchange_Rate_AMD_USD_Abs",
            "Exchange_Rate_AMD_RUB_Abs",
            "Brent_Oil_Price_USD_bbl",
            "Copper_Price_USD_mt",
        ]
        if c in q_new.columns
    ]
    print(q_new.loc[pd.Timestamp("2025-01-01"):pd.Timestamp("2026-01-01"), cols].to_string())


if __name__ == "__main__":
    main()
