from __future__ import annotations

import io
import re
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "processed"

CBA_EXPORTS = {
    "remittances": "https://www.cba.am/en/statistics/monthly-transfers-received-from-abroad-in-the-name-of-individuals-through-ra-banks-and-made-abroad-by-individuals-from-ra/38/export-all/",
    "monetary_aggregates": "https://www.cba.am/en/statistics/monetary-aggregates/50/export-all/",
}

ARMENIAN_CHAR_MAP = str.maketrans(
    {
        "օ": "o",
        "Օ": "O",
        "ո": "o",
        "Ո": "O",
        "ւ": "u",
        "Ւ": "U",
        "ե": "e",
        "Ե": "E",
        "մ": "m",
        "Մ": "M",
        "լ": "l",
        "Լ": "L",
        "ն": "n",
        "Ն": "N",
        "թ": "t",
        "Թ": "T",
        "ա": "a",
        "Ա": "A",
        "ր": "r",
        "Ր": "R",
        "ի": "i",
        "Ի": "I",
        "ֆ": "f",
        "Ֆ": "F",
        "վ": "v",
        "Վ": "V",
        "հ": "h",
        "Հ": "H",
        "ս": "s",
        "Ս": "S",
        "պ": "p",
        "Պ": "P",
        "ջ": "j",
        "Ջ": "J",
    }
)


def _download_excel(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.translate(ARMENIAN_CHAR_MAP)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("thossand", "thousand")
    text = text.replace("thousaod", "thousand")
    text = text.replace("mlo, AMD", "mln, AMD")
    text = text.replace("ml o, AMD", "mln, AMD")
    return text


def _load_cba_sheet(content: bytes, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(content), sheet_name=sheet_name, header=None)
    header_top = raw.iloc[0].apply(_normalize_text)
    header_bottom = raw.iloc[1].apply(_normalize_text)
    headers: list[str] = []
    for idx, (top, bottom) in enumerate(zip(header_top, header_bottom, strict=True)):
        label = " ".join(part for part in [top, bottom] if part)
        label = re.sub(r"\s+", " ", label).strip()
        headers.append(label or f"column_{idx}")

    df = raw.iloc[2:].copy()
    df.columns = headers
    df = df.dropna(how="all")
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def _map_nowcast_columns(monetary: pd.DataFrame, remittances: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Currency in circulation mln, AMD": "Cash_in_Circulation_Mln_AMD",
        "Demand deposits in drams mln, AMD": "Demand_Deposits_Dram_Mln_AMD",
        "M1 mln, AMD": "Money_Supply_M1_Mln_AMD",
        "Time deposits in drams mln, AMD": "Time_Deposits_Dram_Mln_AMD",
        "M2 mln, AMD": "Money_Supply_M2_Mln_AMD",
        "Deposits in foreign currency mln, AMD": "Deposits_FX_Mln_AMD",
        "M2X mln, AMD": "Money_Supply_M2X_Mln_AMD",
        "Inflow mln, AMD": "Remittance_Inflow_Mln_AMD",
        "Inflow thousand, USD": "Remittance_Inflow_Thousand_USD",
        "Inflow of which non-commercial mln, AMD": "Remittance_Inflow_NonCommercial_Mln_AMD",
        "Inflow of which non-commercial thousand, USD": "Remittance_Inflow_NonCommercial_Thousand_USD",
        "Outflow mln, AMD": "Remittance_Outflow_Mln_AMD",
        "Outflow thousand, USD": "Remittance_Outflow_Thousand_USD",
        "Outflow of which non-commercial mln, AMD": "Remittance_Outflow_NonCommercial_Mln_AMD",
        "Outflow of which non-commercial thousand, USD": "Remittance_Outflow_NonCommercial_Thousand_USD",
    }

    merged = monetary.join(remittances, how="outer")
    available = {src: dest for src, dest in mapping.items() if src in merged.columns}
    extension = merged[list(available)].rename(columns=available)

    if {"Remittance_Inflow_Mln_AMD", "Remittance_Outflow_Mln_AMD"}.issubset(extension.columns):
        extension["Remittance_Net_Mln_AMD"] = (
            extension["Remittance_Inflow_Mln_AMD"] - extension["Remittance_Outflow_Mln_AMD"]
        )
    if {"Remittance_Inflow_Thousand_USD", "Remittance_Outflow_Thousand_USD"}.issubset(extension.columns):
        extension["Remittance_Net_Thousand_USD"] = (
            extension["Remittance_Inflow_Thousand_USD"] - extension["Remittance_Outflow_Thousand_USD"]
        )

    return extension.sort_index()


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CBA official exports...")
    remittance_xlsx = _download_excel(CBA_EXPORTS["remittances"])
    monetary_xlsx = _download_excel(CBA_EXPORTS["monetary_aggregates"])

    remittances = _load_cba_sheet(remittance_xlsx, "value, month")
    monetary = _load_cba_sheet(monetary_xlsx, "value, month")
    extension = _map_nowcast_columns(monetary, remittances)
    quarterly = extension.resample("QS").mean()

    remittances.to_csv(PROC_DIR / "cba_remittances_monthly.csv", index_label="date")
    monetary.to_csv(PROC_DIR / "cba_monetary_aggregates_monthly.csv", index_label="date")
    extension.to_csv(PROC_DIR / "cba_nowcast_extension_monthly.csv", index_label="date")
    quarterly.to_csv(PROC_DIR / "cba_nowcast_extension_quarterly.csv", index_label="date")

    print(f"Saved {len(remittances.columns)} remittance series to {PROC_DIR / 'cba_remittances_monthly.csv'}")
    print(f"Saved {len(monetary.columns)} monetary series to {PROC_DIR / 'cba_monetary_aggregates_monthly.csv'}")
    print(f"Saved {len(extension.columns)} mapped nowcast series to {PROC_DIR / 'cba_nowcast_extension_monthly.csv'}")
    print(
        "Coverage:",
        f"{extension.index.min().date()} -> {extension.index.max().date()}",
    )


if __name__ == "__main__":
    main()
