from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
ACTIVE_PATH = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.xlsx"
PARTIAL_PATH = BASE_DIR / "Nowcasting_Data_filled_partial_real_values.xlsx"


QUARTERLY_FIELDS = {
    pd.Timestamp("2025-04-01"): ["Real_GDP_Armenia_YoY", "Nominal_GDP_Mln_AMD"],
    pd.Timestamp("2025-07-01"): ["Real_GDP_Armenia_YoY", "Nominal_GDP_Mln_AMD"],
    pd.Timestamp("2025-10-01"): ["Real_GDP_Armenia_YoY", "Nominal_GDP_Mln_AMD"],
}

MONTHLY_FIELDS = {
    pd.Timestamp("2026-01-01"): [
        "Economic_Activity_Index_Discrete_YoY",
        "CPI_YoY",
        "Industry_Output_Mln_AMD",
        "Industry_Real_Growth_YoY",
        "Construction_Output_Mln_AMD",
        "Construction_Real_Growth_YoY",
    ],
    pd.Timestamp("2026-02-01"): [
        "Economic_Activity_Index_Discrete_YoY",
        "CPI_YoY",
        "Industry_Output_Mln_AMD",
        "Industry_Real_Growth_YoY",
        "Construction_Output_Mln_AMD",
        "Construction_Real_Growth_YoY",
    ],
}


def merge_fields(active: pd.DataFrame, partial: pd.DataFrame, field_map: dict[pd.Timestamp, list[str]]) -> tuple[pd.DataFrame, list[tuple[str, str, object, object]]]:
    active = active.copy().set_index("Date").sort_index()
    partial = partial.copy().set_index("Date").sort_index()
    changes: list[tuple[str, str, object, object]] = []

    for dt, cols in field_map.items():
        if dt not in partial.index or dt not in active.index:
            continue
        for col in cols:
            if col not in active.columns or col not in partial.columns:
                continue
            pval = partial.at[dt, col]
            aval = active.at[dt, col]
            if pd.isna(pval):
                continue
            if pd.isna(aval) or aval != pval:
                active.at[dt, col] = pval
                changes.append((dt.strftime("%Y-%m-%d"), col, aval, pval))

    return active.reset_index(), changes


def main() -> None:
    active_q = pd.read_excel(ACTIVE_PATH, sheet_name="Quarterly")
    active_m = pd.read_excel(ACTIVE_PATH, sheet_name="Monthly")
    partial_q = pd.read_excel(PARTIAL_PATH, sheet_name="Quarterly")
    partial_m = pd.read_excel(PARTIAL_PATH, sheet_name="Monthly")

    for df in (active_q, active_m, partial_q, partial_m):
        df["Date"] = pd.to_datetime(df["Date"])

    active_q, q_changes = merge_fields(active_q, partial_q, QUARTERLY_FIELDS)
    active_m, m_changes = merge_fields(active_m, partial_m, MONTHLY_FIELDS)

    with pd.ExcelWriter(ACTIVE_PATH, engine="openpyxl") as writer:
        active_q.to_excel(writer, sheet_name="Quarterly", index=False)
        active_m.to_excel(writer, sheet_name="Monthly", index=False)

    print("Merged documented source-backed values from partial workbook.")
    print(f"Quarterly changes: {len(q_changes)}")
    for date, col, old, new in q_changes:
        print(f"{date} | {col}: {old} -> {new}")
    print(f"\nMonthly changes: {len(m_changes)}")
    for date, col, old, new in m_changes:
        print(f"{date} | {col}: {old} -> {new}")


if __name__ == "__main__":
    main()
