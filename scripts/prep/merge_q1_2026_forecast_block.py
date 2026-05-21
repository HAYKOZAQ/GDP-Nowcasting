from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
ACTIVE_PATH = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.xlsx"
FORECAST_PATH = BASE_DIR / "Nowcasting_Data_filled_actuals_plus_forecasts.xlsx"
BACKUP_PATH = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.backup_before_q1_2026_forecast_merge.xlsx"


def main() -> None:
    if not BACKUP_PATH.exists():
        shutil.copy2(ACTIVE_PATH, BACKUP_PATH)

    active_q = pd.read_excel(ACTIVE_PATH, sheet_name="Quarterly")
    active_m = pd.read_excel(ACTIVE_PATH, sheet_name="Monthly")
    forecast_q = pd.read_excel(FORECAST_PATH, sheet_name="Quarterly")

    active_q["Date"] = pd.to_datetime(active_q["Date"])
    active_m["Date"] = pd.to_datetime(active_m["Date"])
    forecast_q["Date"] = pd.to_datetime(forecast_q["Date"])

    active_q = active_q.set_index("Date").sort_index()
    forecast_q = forecast_q.set_index("Date").sort_index()

    q1 = pd.Timestamp("2026-01-01")
    if q1 not in forecast_q.index:
        raise KeyError("2026-01-01 not found in forecast workbook quarterly sheet.")
    if q1 not in active_q.index:
        active_q.loc[q1] = pd.Series(dtype=float)

    changes: list[tuple[str, object, object]] = []
    for col in active_q.columns:
        if col not in forecast_q.columns:
            continue
        aval = active_q.at[q1, col]
        fval = forecast_q.at[q1, col]
        if pd.isna(fval):
            continue
        if pd.isna(aval) or aval != fval:
            active_q.at[q1, col] = fval
            changes.append((col, aval, fval))

    with pd.ExcelWriter(ACTIVE_PATH, engine="openpyxl") as writer:
        active_q.reset_index().to_excel(writer, sheet_name="Quarterly", index=False)
        active_m.sort_values("Date").to_excel(writer, sheet_name="Monthly", index=False)

    print(f"Backup written to: {BACKUP_PATH}")
    print(f"Merged {len(changes)} quarterly 2026 Q1 forecast values into the active workbook.")
    for col, old, new in changes:
        print(f"{col}: {old} -> {new}")


if __name__ == "__main__":
    main()
