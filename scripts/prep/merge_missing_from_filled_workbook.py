from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
ACTIVE_PATH = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.xlsx"
FILLED_PATH = BASE_DIR / "Nowcasting_Data_Filled.xlsx"
BACKUP_PATH = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.backup_before_filled_merge.xlsx"


def merge_missing(active: pd.DataFrame, filled: pd.DataFrame, start_date: str) -> tuple[pd.DataFrame, list[tuple[str, str, object]]]:
    active = active.copy().set_index("Date").sort_index()
    filled = filled.copy().set_index("Date").sort_index()
    changes: list[tuple[str, str, object]] = []

    for dt in active.index[active.index >= pd.Timestamp(start_date)]:
        if dt not in filled.index:
            continue
        for col in active.columns:
            if col not in filled.columns:
                continue
            aval = active.at[dt, col]
            fval = filled.at[dt, col]
            if pd.isna(aval) and pd.notna(fval):
                active.at[dt, col] = fval
                changes.append((dt.strftime("%Y-%m-%d"), col, fval))

    return active.reset_index(), changes


def main() -> None:
    if not BACKUP_PATH.exists():
        shutil.copy2(ACTIVE_PATH, BACKUP_PATH)

    active_q = pd.read_excel(ACTIVE_PATH, sheet_name="Quarterly")
    active_m = pd.read_excel(ACTIVE_PATH, sheet_name="Monthly")
    filled_q = pd.read_excel(FILLED_PATH, sheet_name="Quarterly")
    filled_m = pd.read_excel(FILLED_PATH, sheet_name="Monthly")

    for df in (active_q, active_m, filled_q, filled_m):
        df["Date"] = pd.to_datetime(df["Date"])

    active_q, q_changes = merge_missing(active_q, filled_q, "2025-01-01")
    active_m, m_changes = merge_missing(active_m, filled_m, "2025-01-01")

    with pd.ExcelWriter(ACTIVE_PATH, engine="openpyxl") as writer:
        active_q.to_excel(writer, sheet_name="Quarterly", index=False)
        active_m.to_excel(writer, sheet_name="Monthly", index=False)

    print(f"Backup: {BACKUP_PATH}")
    print(f"Quarterly missing values filled: {len(q_changes)}")
    for date, col, val in q_changes[:80]:
        print(f"{date} | {col} -> {val}")
    print(f"\nMonthly missing values filled: {len(m_changes)}")
    for date, col, val in m_changes[:120]:
        print(f"{date} | {col} -> {val}")


if __name__ == "__main__":
    main()
