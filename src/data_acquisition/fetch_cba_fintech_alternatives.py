from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / "data" / "processed"

SOURCES = {
    "cards": "https://old.cba.am/stat/stat_data_eng/Number_of_payment_cards_eng.xls",
    "transactions": "https://old.cba.am/stat/stat_data_eng/Payment_card_transactions_eng.xls",
    "emoney": "https://old.cba.am/stat/stat_data_eng/E-Money_eng.xls",
}


def _download_excel(url: str) -> BytesIO:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return BytesIO(response.content)


def _parse_quarter_label(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"([IVX]+)\s*-\s*(\d{4})", text)
    if not match:
        return None
    roman, year_text = match.groups()
    quarter_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    quarter = quarter_map.get(roman)
    if quarter is None:
        return None
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=int(year_text), month=month, day=1)


def _safe_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(pd.to_numeric(value, errors="coerce"))


def _build_cards_quarterly() -> pd.DataFrame:
    raw = pd.read_excel(_download_excel(SOURCES["cards"]), sheet_name="Number of payment cards", header=None)
    quarter_positions: list[tuple[int, pd.Timestamp]] = []
    for col_idx, value in enumerate(raw.iloc[3].tolist()):
        quarter = _parse_quarter_label(value)
        if quarter is not None:
            quarter_positions.append((col_idx, quarter))

    total_row = raw[raw.iloc[:, 0].astype(str).str.strip() == "Total"]
    if total_row.empty:
        raise ValueError("Could not find Total row in payment cards file.")
    total_row = total_row.iloc[0]

    records: list[dict[str, float | pd.Timestamp]] = []
    for start_col, quarter in quarter_positions:
        settlement = _safe_float(total_row.iloc[start_col]) + _safe_float(total_row.iloc[start_col + 1])
        debit = _safe_float(total_row.iloc[start_col + 2]) + _safe_float(total_row.iloc[start_col + 3])
        credit = _safe_float(total_row.iloc[start_col + 4]) + _safe_float(total_row.iloc[start_col + 5])
        total = _safe_float(total_row.iloc[start_col + 6])
        records.append(
            {
                "date": quarter,
                "ALTQ_PaymentCards_Total": total,
                "ALTQ_PaymentCards_Settlement": settlement,
                "ALTQ_PaymentCards_Debit": debit,
                "ALTQ_PaymentCards_Credit": credit,
                "ALTQ_PaymentCards_Credit_Share": credit / total if pd.notna(total) and total != 0 else float("nan"),
            }
        )

    return pd.DataFrame(records).set_index("date").sort_index()


def _build_transactions_quarterly() -> pd.DataFrame:
    workbook = pd.ExcelFile(_download_excel(SOURCES["transactions"]))
    records: list[dict[str, float | pd.Timestamp]] = []
    for sheet_name in workbook.sheet_names:
        quarter = _parse_quarter_label(sheet_name)
        if quarter is None:
            continue
        raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
        volume_row = raw[raw.iloc[:, 1].astype(str).str.strip() == "Volume*"]
        number_row = raw[raw.iloc[:, 1].astype(str).str.strip() == "Number"]
        if volume_row.empty or number_row.empty:
            continue
        volume_row = volume_row.iloc[0]
        number_row = number_row.iloc[0]

        cash_volume = _safe_float(volume_row.iloc[2]) + _safe_float(volume_row.iloc[3])
        purchase_volume = _safe_float(volume_row.iloc[5]) + _safe_float(volume_row.iloc[6])
        card_to_card_volume = _safe_float(volume_row.iloc[7]) + _safe_float(volume_row.iloc[8])
        utility_volume = (
            _safe_float(volume_row.iloc[9]) + _safe_float(volume_row.iloc[10]) + _safe_float(volume_row.iloc[11])
        )
        budget_volume = _safe_float(volume_row.iloc[12]) + _safe_float(volume_row.iloc[13])
        insurance_volume = _safe_float(volume_row.iloc[14]) + _safe_float(volume_row.iloc[15])
        other_noncash_volume = (
            utility_volume
            + budget_volume
            + insurance_volume
            + _safe_float(volume_row.iloc[16])
            + _safe_float(volume_row.iloc[17])
            + _safe_float(volume_row.iloc[18])
            + _safe_float(volume_row.iloc[19])
            + _safe_float(volume_row.iloc[20])
        )
        total_noncash_volume = purchase_volume + card_to_card_volume + other_noncash_volume

        cash_number = _safe_float(number_row.iloc[2]) + _safe_float(number_row.iloc[3])
        purchase_number = _safe_float(number_row.iloc[5]) + _safe_float(number_row.iloc[6])
        card_to_card_number = _safe_float(number_row.iloc[7]) + _safe_float(number_row.iloc[8])
        utility_number = (
            _safe_float(number_row.iloc[9]) + _safe_float(number_row.iloc[10]) + _safe_float(number_row.iloc[11])
        )
        budget_number = _safe_float(number_row.iloc[12]) + _safe_float(number_row.iloc[13])
        insurance_number = _safe_float(number_row.iloc[14]) + _safe_float(number_row.iloc[15])
        other_noncash_number = (
            utility_number
            + budget_number
            + insurance_number
            + _safe_float(number_row.iloc[16])
            + _safe_float(number_row.iloc[17])
            + _safe_float(number_row.iloc[18])
            + _safe_float(number_row.iloc[19])
            + _safe_float(number_row.iloc[20])
        )
        total_noncash_number = purchase_number + card_to_card_number + other_noncash_number

        records.append(
            {
                "date": quarter,
                "ALTQ_CardTxn_CashWithdrawal_Volume_Mln_AMD": cash_volume,
                "ALTQ_CardTxn_Purchases_Volume_Mln_AMD": purchase_volume,
                "ALTQ_CardTxn_CardToCard_Volume_Mln_AMD": card_to_card_volume,
                "ALTQ_CardTxn_Noncash_Volume_Mln_AMD": total_noncash_volume,
                "ALTQ_CardTxn_Noncash_Share": total_noncash_volume / (total_noncash_volume + cash_volume)
                if pd.notna(total_noncash_volume) and pd.notna(cash_volume) and (total_noncash_volume + cash_volume) != 0
                else float("nan"),
                "ALTQ_CardTxn_CashWithdrawal_Number": cash_number,
                "ALTQ_CardTxn_Purchases_Number": purchase_number,
                "ALTQ_CardTxn_CardToCard_Number": card_to_card_number,
                "ALTQ_CardTxn_Noncash_Number": total_noncash_number,
            }
        )

    return pd.DataFrame(records).set_index("date").sort_index()


def _build_emoney_quarterly() -> pd.DataFrame:
    raw = pd.read_excel(_download_excel(SOURCES["emoney"]), sheet_name="E-Money", header=None)
    quarter_positions: list[tuple[int, pd.Timestamp]] = []
    for col_idx, value in enumerate(raw.iloc[3].tolist()):
        quarter = _parse_quarter_label(value)
        if quarter is not None:
            quarter_positions.append((col_idx, quarter))

    def row_for(label: str) -> pd.Series:
        match = raw[raw.iloc[:, 1].astype(str).str.strip() == label]
        if match.empty:
            raise ValueError(f"Could not find '{label}' row in e-money file.")
        return match.iloc[0]

    active_accounts = row_for("Active E-Money accounts")
    deposits = row_for("E-Money account deposits, of which")
    transactions = row_for("E-Money transactions, of which")
    goods_services = row_for("for goods and services")

    records: list[dict[str, float | pd.Timestamp]] = []
    for start_col, quarter in quarter_positions:
        records.append(
            {
                "date": quarter,
                "ALTQ_EMoney_ActiveAccounts": _safe_float(active_accounts.iloc[start_col]),
                "ALTQ_EMoney_Deposits_Volume_Mln_AMD": _safe_float(deposits.iloc[start_col]),
                "ALTQ_EMoney_Deposits_Number": _safe_float(deposits.iloc[start_col + 1]),
                "ALTQ_EMoney_Transactions_Volume_Mln_AMD": _safe_float(transactions.iloc[start_col]),
                "ALTQ_EMoney_Transactions_Number": _safe_float(transactions.iloc[start_col + 1]),
                "ALTQ_EMoney_GoodsServices_Volume_Mln_AMD": _safe_float(goods_services.iloc[start_col]),
                "ALTQ_EMoney_GoodsServices_Number": _safe_float(goods_services.iloc[start_col + 1]),
            }
        )

    return pd.DataFrame(records).set_index("date").sort_index()


def _with_growth_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_index().copy()
    for col in list(out.columns):
        series = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_YoY"] = series.pct_change(4, fill_method=None) * 100
        out[f"{col}_QoQ"] = series.pct_change(1, fill_method=None) * 100
    return out


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    cards = _build_cards_quarterly()
    transactions = _build_transactions_quarterly()
    emoney = _build_emoney_quarterly()
    quarterly = cards.join(transactions, how="outer").join(emoney, how="outer").sort_index()
    quarterly = _with_growth_features(quarterly)

    output_path = PROC_DIR / "cba_fintech_alternatives_quarterly.csv"
    quarterly.to_csv(output_path, index_label="date")

    print(f"Saved {quarterly.shape[1]} curated fintech alternative series to {output_path}")
    print(f"Coverage: {quarterly.index.min().date()} -> {quarterly.index.max().date()}")


if __name__ == "__main__":
    main()
