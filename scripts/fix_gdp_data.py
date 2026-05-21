import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_XLSX = BASE_DIR / "data" / "raw" / "Translated_Cleaned_Nowcasting_Data.xlsx"
PROC_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "backtests"

TARGET_DATE = "2026-01-01"
OLD_VAL = 105.9
NEW_VAL = 107.1

def update_csv(path, target_date, old_val, new_val, date_col='date'):
    if not path.exists():
        print(f"File not found: {path}")
        return
    df = pd.read_csv(path)
    if date_col not in df.columns:
        # Try to find date column
        for col in ['Date', 'prediction_date', 'date']:
            if col in df.columns:
                date_col = col
                break
    
    mask = (df[date_col].astype(str).str.startswith(target_date))
    if 'actual' in df.columns:
        # For backtest_predictions.csv
        df.loc[mask, 'actual'] = new_val
        # Recalculate errors
        df.loc[mask, 'abs_error'] = (df.loc[mask, 'actual'] - df.loc[mask, 'prediction']).abs()
        df.loc[mask, 'abs_pct_error'] = (df.loc[mask, 'abs_error'] / df.loc[mask, 'actual'].abs()) * 100
        df.loc[mask, 'squared_error'] = (df.loc[mask, 'actual'] - df.loc[mask, 'prediction'])**2
        df.loc[mask, 'residual'] = df.loc[mask, 'actual'] - df.loc[mask, 'prediction']
    else:
        # For data files, look for the column containing 105.9
        for col in df.columns:
            if col != date_col:
                col_mask = mask & (df[col] == old_val)
                if col_mask.any():
                    print(f"Updating column '{col}' in {path.name}")
                    df.loc[col_mask, col] = new_val
    
    df.to_csv(path, index=False)
    print(f"Updated {path.name}")

def update_excel():
    if not RAW_XLSX.exists():
        print(f"Excel not found: {RAW_XLSX}")
        return
    
    xls = pd.ExcelFile(RAW_XLSX)
    df_q = pd.read_excel(xls, sheet_name="Quarterly")
    df_m = pd.read_excel(xls, sheet_name="Monthly")
    
    q_mask = (df_q['Date'].astype(str).str.startswith(TARGET_DATE))
    if 'Real_GDP_Armenia_YoY' in df_q.columns:
        df_q.loc[q_mask, 'Real_GDP_Armenia_YoY'] = NEW_VAL
        print("Updated Real_GDP_Armenia_YoY in Excel Quarterly sheet")
    else:
        # Search for 105.9
        for col in df_q.columns:
            if col != 'Date':
                col_mask = q_mask & (df_q[col] == OLD_VAL)
                if col_mask.any():
                    df_q.loc[col_mask, col] = NEW_VAL
                    print(f"Updated column '{col}' in Excel Quarterly sheet")

    with pd.ExcelWriter(RAW_XLSX, engine="openpyxl") as writer:
        df_q.to_excel(writer, sheet_name="Quarterly", index=False)
        df_m.to_excel(writer, sheet_name="Monthly", index=False)
    print("Updated Excel file")

def main():
    print("Starting data correction...")
    
    # 1. Update Excel
    update_excel()
    
    # 2. Update Processed CSVs
    update_csv(PROC_DIR / "armstat_nowcast_extension_quarterly.csv", TARGET_DATE, OLD_VAL, NEW_VAL)
    update_csv(PROC_DIR / "armstat_nowcast_extension_monthly.csv", TARGET_DATE, OLD_VAL, NEW_VAL)
    
    # 3. Update Results CSVs (Predictions)
    update_csv(RESULTS_DIR / "backtest_predictions.csv", TARGET_DATE, OLD_VAL, NEW_VAL, date_col='prediction_date')
    
    print("Data correction completed. Now you should run the pipeline to refresh summaries and figures.")

if __name__ == "__main__":
    main()
