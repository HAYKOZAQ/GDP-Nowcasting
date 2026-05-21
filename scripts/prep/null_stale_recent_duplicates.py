from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_XLSX = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.xlsx"


MONTHLY_STALE_COLS = [
    "CPI_YoY",
    "Commercial_Bank_Loans_Mln_AMD",
    "Construction_Output_Mln_AMD",
    "Construction_Real_Growth_YoY",
    "Consumer_Loans_Mln_AMD",
    "Economic_Activity_Index_Discrete_YoY",
    "Enterprise_Loans_Mln_AMD",
    "Hired_Workers",
    "Household_Loans_Mln_AMD",
    "Industry_Output_Mln_AMD",
    "Industry_Real_Growth_YoY",
    "Loans_Agriculture_Mln_AMD",
    "Loans_Construction_Mln_AMD",
    "Loans_Industry_Mln_AMD",
    "Loans_Residents_Banks_Mln_AMD",
    "Loans_Residents_Credit_Orgs_Mln_AMD",
    "Loans_Services_Mln_AMD",
    "Loans_Trade_Mln_AMD",
    "Loans_Transport_Communication_Mln_AMD",
    "Long_Term_Nominal_Interest_Rate_Deposits_AMD",
    "Long_Term_Nominal_Interest_Rate_Deposits_USD",
    "Long_Term_Nominal_Interest_Rate_Loans_AMD",
    "Long_Term_Nominal_Interest_Rate_Loans_USD",
    "Mortgage_Loans_Mln_AMD",
    "Other_Loans_Mln_AMD",
    "Private_Construction_Real_Growth_YoY",
    "Private_Enterprise_Loans_Mln_AMD",
    "Services_Output_Mln_AMD",
    "Services_Real_Growth_YoY",
    "Short_Term_Nominal_Interest_Rate_Deposits_AMD",
    "Short_Term_Nominal_Interest_Rate_Deposits_USD",
    "Short_Term_Nominal_Interest_Rate_Loans_AMD",
    "Short_Term_Nominal_Interest_Rate_Loans_USD",
    "Total_Loans_Mln_AMD",
]

QUARTERLY_STALE_Q3_Q4_COLS = [
    "Real_Private_Consumption_YoY",
    "Real_Government_Consumption_YoY",
    "Real_Aggregate_Investments_YoY",
    "Real_Fixed_Capital_Investments_YoY",
    "Real_Private_Investments_YoY",
    "Real_Government_Investments_YoY",
    "Real_Exports_YoY",
    "Real_Imports_YoY",
    "Unemployment_Growth_YoY",
    "Employment_YoY",
    "Hired_Workers_YoY",
    "Average_Nominal_Salary_YoY",
    "CPI_YoY",
    "REER_YoY",
    "Unemployment_Rate_Pct",
    "Real_Disposable_Income_Abs",
    "Real_GDP_Armenia_Abs",
    "Real_Private_Consumption_Abs",
    "Real_Private_Investments_Abs",
    "Real_Construction_Abs",
    "Economic_Activity_Indicator_YoY",
    "Nominal_GDP_Mln_AMD",
    "Primary_Income_Labor_Mln_USD",
    "Secondary_Income_Transfers_Mln_USD",
    "Net_Non_Commercial_Inflow_Nominal_USD_YoY",
    "Primary_Income_Mln_AMD",
    "Secondary_Income_Mln_AMD",
    "Disposable_Income_Mln_AMD",
    "Disposable_Income_YoY",
    "Real_Disposable_Income_YoY_2",
]

QUARTERLY_STALE_Q2_FROM_Q1_COLS = [
    "Real_Private_Consumption_YoY",
    "Real_Government_Consumption_YoY",
    "Real_Aggregate_Investments_YoY",
    "Real_Fixed_Capital_Investments_YoY",
    "Real_Private_Investments_YoY",
    "Real_Government_Investments_YoY",
    "Real_Exports_YoY",
    "Real_Imports_YoY",
    "Unemployment_Growth_YoY",
    "Employment_YoY",
    "REER_YoY",
    "Unemployment_Rate_Pct",
    "Real_Private_Consumption_Abs",
    "Real_Private_Investments_Abs",
]


def main() -> None:
    xls = pd.ExcelFile(RAW_XLSX)
    q = pd.read_excel(xls, sheet_name="Quarterly")
    m = pd.read_excel(xls, sheet_name="Monthly")
    q["Date"] = pd.to_datetime(q["Date"])
    m["Date"] = pd.to_datetime(m["Date"])

    # Null stale repeated monthly values in 2025 H2 only.
    monthly_mask = (m["Date"] >= pd.Timestamp("2025-07-01")) & (m["Date"] <= pd.Timestamp("2025-12-01"))
    changed_monthly: list[tuple[str, str, object]] = []
    for col in MONTHLY_STALE_COLS:
        if col not in m.columns:
            continue
        vals = m.loc[monthly_mask, col]
        uniq = vals.dropna().unique()
        if len(uniq) == 1 and len(vals.dropna()) == 6:
            stale_value = uniq[0]
            m.loc[monthly_mask, col] = pd.NA
            changed_monthly.append((col, "2025-07..2025-12", stale_value))

    # Null quarterly stale duplicated Q3/Q4 values when they match Q2.
    q2 = pd.Timestamp("2025-04-01")
    q3 = pd.Timestamp("2025-07-01")
    q4 = pd.Timestamp("2025-10-01")
    q1 = pd.Timestamp("2025-01-01")
    changed_quarterly: list[tuple[str, str, object]] = []
    for col in QUARTERLY_STALE_Q3_Q4_COLS:
        if col not in q.columns:
            continue
        q2v = q.loc[q["Date"] == q2, col]
        q3v = q.loc[q["Date"] == q3, col]
        q4v = q.loc[q["Date"] == q4, col]
        if q2v.empty or q3v.empty or q4v.empty:
            continue
        q2v, q3v, q4v = q2v.iloc[0], q3v.iloc[0], q4v.iloc[0]
        if pd.notna(q2v) and pd.notna(q3v) and pd.notna(q4v) and q2v == q3v == q4v:
            q.loc[q["Date"].isin([q3, q4]), col] = pd.NA
            changed_quarterly.append((col, "2025-Q3/Q4", q2v))

    # Null quarterly Q2 values that are exact carry-overs from Q1 for suspicious columns.
    for col in QUARTERLY_STALE_Q2_FROM_Q1_COLS:
        if col not in q.columns:
            continue
        q1v = q.loc[q["Date"] == q1, col]
        q2v = q.loc[q["Date"] == q2, col]
        if q1v.empty or q2v.empty:
            continue
        q1v, q2v = q1v.iloc[0], q2v.iloc[0]
        if pd.notna(q1v) and pd.notna(q2v) and q1v == q2v:
            q.loc[q["Date"] == q2, col] = pd.NA
            changed_quarterly.append((col, "2025-Q2", q2v))

    with pd.ExcelWriter(RAW_XLSX, engine="openpyxl") as writer:
        q.to_excel(writer, sheet_name="Quarterly", index=False)
        m.to_excel(writer, sheet_name="Monthly", index=False)

    print(f"Nullified {len(changed_monthly)} stale monthly columns and {len(changed_quarterly)} stale quarterly columns.")
    if changed_monthly:
        print("\\nMonthly columns nulled:")
        for col, period, value in changed_monthly:
            print(f"{period} | {col}: removed repeated value {value}")
    if changed_quarterly:
        print("\\nQuarterly columns nulled:")
        for col, period, value in changed_quarterly:
            print(f"{period} | {col}: removed repeated value {value}")


if __name__ == "__main__":
    main()
