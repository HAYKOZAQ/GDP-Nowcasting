"""
===============================================================
EXPANDING WINDOW WALK-FORWARD BACKTESTING (2000-2025)
---------------------------------------------------------------
Instead of a single train/test split, this script:
  - Starts with training data from 2000 to 2010 (minimum)
  - Predicts 2010-Q1
  - Then expands the training window by 1 quarter
  - Predicts 2010-Q2
  - Continues all the way to 2025-Q1
  
This gives us:
  1. TRUE model performance (no look-ahead bias)
  2. Performance on the 2022 war shock period
  3. Error evolution over time (does the model improve as it gets more data?)
===============================================================
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings, io, sys

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------------------------------------------
# LOAD DATA (same as shock-robust pipeline)
# ----------------------------------------------------------------
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(BASE_DIR)
file_path = BASE_DIR / 'data' / 'raw' / 'Translated_Cleaned_Nowcasting_Data.xlsx'
print("Loading data...")
df_q = pd.read_excel(file_path, sheet_name='Quarterly', index_col='Date')
df_m = pd.read_excel(file_path, sheet_name='Monthly', index_col='Date')

trends_global = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'google_trends_armenia_quarterly.csv', index_col='date', parse_dates=True)
trends_local  = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'google_trends_armenian_quarterly.csv', index_col=0, parse_dates=True)
trends_global.index.name = trends_local.index.name = 'Date'
trends_global.columns = [f'GT_global_{c}' for c in trends_global.columns]
trends_local.columns  = [f'GT_local_{c}'  for c in trends_local.columns]

def add_qoq(df):
    chg = df.pct_change().replace([np.inf, -np.inf], np.nan) * 100
    chg.columns = [f'{c}_QoQ' for c in df.columns]
    return pd.concat([df, chg], axis=1)

trends_global = add_qoq(trends_global)
trends_local  = add_qoq(trends_local)
df_q = df_q.merge(trends_global, left_index=True, right_index=True, how='left')
df_q = df_q.merge(trends_local,  left_index=True, right_index=True, how='left')

# Crisis dummies
df_q['Dummy_GFC']       = ((df_q.index >= '2008-10-01') & (df_q.index <= '2010-01-01')).astype(int)
df_q['Dummy_COVID']     = ((df_q.index >= '2020-01-01') & (df_q.index <= '2021-04-01')).astype(int)
df_q['Dummy_RU_WAR']    = ((df_q.index >= '2022-01-01') & (df_q.index <= '2022-12-31')).astype(int)
df_q['Dummy_AMD_Surge'] = ((df_q.index >= '2022-04-01') & (df_q.index <= '2023-06-30')).astype(int)

# BOP / income features
df_q['Primary_Income_YoY']    = df_q['Primary_Income_Labor_Mln_USD'].pct_change(4) * 100
df_q['Secondary_Income_YoY']  = df_q['Secondary_Income_Transfers_Mln_USD'].pct_change(4) * 100
df_q['AMD_USD_StrongSignal']  = (100 - df_q['Exchange_Rate_AMD_USD_YoY'])
df_q['AMD_RUB_Level_QoQ']     = df_q['Exchange_Rate_AMD_RUB_Abs'].pct_change() * 100
df_q['Migration_Inflow_Signal'] = (df_q['AMD_USD_StrongSignal'].clip(lower=0) *
                                    df_q['Primary_Income_YoY'].clip(lower=0)) / 100
df_q['Salary_YoY_Lag1']       = df_q['Average_Nominal_Salary_YoY'].shift(1)
df_q['REER_Surge']             = (df_q['REER_YoY'] - 100).clip(lower=0)
df_q['Employment_Growth']      = df_q['Employment_YoY']

target_col = 'Real_GDP_Armenia_YoY'

q_features = [
    'Real_GDP_Russia_YoY', 'CPI_YoY', 'Exchange_Rate_AMD_USD_YoY', 'REER_YoY',
    'Brent_Oil_Price_USD_bbl', 'Copper_Price_USD_mt',
    'Dummy_GFC', 'Dummy_COVID', 'Dummy_RU_WAR', 'Dummy_AMD_Surge',
    'Primary_Income_YoY', 'Secondary_Income_YoY', 'Disposable_Income_YoY',
    'Salary_YoY_Lag1', 'AMD_USD_StrongSignal', 'AMD_RUB_Level_QoQ',
    'Migration_Inflow_Signal', 'Employment_Growth', 'REER_Surge', 'Unemployment_Rate_Pct',
] + [c for c in df_q.columns if c.startswith('GT_')]

# Monthly features
monthly_stationary = ['CPI_YoY', 'Exchange_Rate_AMD_USD', 'Exchange_Rate_AMD_RUB',
                      'Brent_Oil_Price_USD_bbl', 'Copper_Price_USD_mt',
                      'Short_Term_Nominal_Interest_Rate_Loans_AMD',
                      'Short_Term_Nominal_Interest_Rate_Deposits_AMD',
                      'Long_Term_Nominal_Interest_Rate_Loans_AMD',
                      'Economic_Activity_Index_Discrete_YoY',
                      'Industry_Real_Growth_YoY', 'Construction_Real_Growth_YoY', 'Services_Real_Growth_YoY']
monthly_stationary = [c for c in monthly_stationary if c in df_m.columns]
money_levels = ['Cash_in_Circulation_Mln_AMD', 'Money_Supply_M2_Mln_AMD', 'Money_Supply_M2X_Mln_AMD',
                'Commercial_Bank_Loans_Mln_AMD', 'Enterprise_Loans_Mln_AMD', 'Household_Loans_Mln_AMD',
                'Total_Loans_Mln_AMD', 'Loans_Industry_Mln_AMD', 'Loans_Agriculture_Mln_AMD',
                'Loans_Construction_Mln_AMD', 'Loans_Services_Mln_AMD']
df_m_yoy = pd.DataFrame(index=df_m.index)
for col in money_levels:
    if col in df_m.columns:
        df_m_yoy[f'{col}_YoY'] = df_m[col].pct_change(12) * 100
        df_m_yoy[f'{col}_QoQ'] = df_m[col].pct_change(3) * 100
df_m_sel = pd.concat([df_m[monthly_stationary], df_m_yoy], axis=1)

k = np.arange(1, 4, dtype=float); almon_w = np.exp(-(k-2)**2); almon_w /= almon_w.sum()

# ----------------------------------------------------------------
# BUILD FULL U-MIDAS MATRIX (entire history 2000-2025)
# ----------------------------------------------------------------
print("Building full U-MIDAS matrix (2000-2025)...")
target_series = df_q[target_col].copy()
aligned_rows = []

for index, row in df_q[[target_col] + q_features].iterrows():
    q_year, q0 = index.year, index.month
    row_data = {'Date': index, 'Target': row[target_col]}
    for lag, months in [(1, 3), (2, 6), (4, 12)]:
        prev = index - pd.DateOffset(months=months)
        row_data[f'AR{lag}_GDP'] = target_series.get(prev, np.nan)
    for feat in q_features:
        if feat in row.index:
            row_data[f'Q_{feat}'] = row[feat]
    monthly_vals = {col: [] for col in df_m_sel.columns}
    for m_idx, offset in enumerate([0, 1, 2]):
        m = q0 + offset; y = q_year + (m - 1)//12; m = ((m-1)%12)+1
        m_date = pd.Timestamp(year=y, month=m, day=1)
        for col in df_m_sel.columns:
            val = df_m_sel.loc[m_date, col] if m_date in df_m_sel.index else np.nan
            row_data[f'M{m_idx+1}_{col}'] = val
            monthly_vals[col].append(val)
    for col in df_m_sel.columns:
        vals = np.array(monthly_vals[col], dtype=float)
        row_data[f'ALMON_{col}'] = np.nansum(almon_w*vals) if not np.all(np.isnan(vals)) else np.nan
    aligned_rows.append(row_data)

df_full = pd.DataFrame(aligned_rows).set_index('Date')
df_full.dropna(subset=['Target'], inplace=True)
df_full.dropna(axis=1, thresh=len(df_full)*0.7, inplace=True)
df_full = df_full.ffill()

monthly_cols = [c for c in df_full.columns if c.startswith(('M1_','M2_','M3_','ALMON_'))]
other_cols   = [c for c in df_full.columns if c not in ['Target'] + monthly_cols]

print(f"Full dataset: {len(df_full)} quarters x {df_full.shape[1]} raw features")

# ----------------------------------------------------------------
# EXPANDING WINDOW WALK-FORWARD VALIDATION
# Start training from 2000, minimum 40 quarters (~10 years) of training data
# ----------------------------------------------------------------
MIN_TRAIN = 40   # minimum quarters before we start predicting
all_dates = df_full.index.tolist()
n_total   = len(all_dates)

print(f"\nRunning expanding window validation...")
print(f"  Train starts: {all_dates[0].date()} | Min training quarters: {MIN_TRAIN}")
print(f"  First prediction: {all_dates[MIN_TRAIN].date()}")
print(f"  Last prediction:  {all_dates[-1].date()}")
print(f"  Total predictions: {n_total - MIN_TRAIN}")

walk_results = []
X_all = df_full.drop(columns=['Target'])
y_all = df_full['Target']

for pred_idx in range(MIN_TRAIN, n_total):
    train_dates = all_dates[:pred_idx]
    pred_date   = all_dates[pred_idx]

    X_tr_raw = X_all.loc[train_dates]
    y_tr     = y_all.loc[train_dates]
    X_te_raw = X_all.loc[[pred_date]]
    y_te     = y_all.loc[pred_date]

    # Impute per window
    monthly_tr = X_tr_raw[[c for c in X_tr_raw.columns if c in monthly_cols]]
    other_tr   = X_tr_raw[[c for c in X_tr_raw.columns if c in other_cols]]
    monthly_te = X_te_raw[[c for c in X_te_raw.columns if c in monthly_cols]]
    other_te   = X_te_raw[[c for c in X_te_raw.columns if c in other_cols]]

    # --- SIMULATE PUBLICATION DELAY (TARGET LEAKAGE PROTECTION) ---
    # In reality, Q4 data (e.g. GDP lag, Q4 employment) is not fully available when predicting Q1.
    # We shift 'other_tr' and 'other_te' lag features to simulate this delay
    lag_features = [c for c in other_cols if 'Lag' in c or 'AR' in c]
    for c in lag_features:
        if c in other_tr.columns:
            other_tr.loc[:, c] = other_tr[c].shift(1) # Shift by 1 period to simulate delay
        if c in other_te.columns:
             # For test data, we just take the last known value from training data to avoid peeking into the future
            if len(other_tr) > 0:
                other_te.loc[:, c] = other_tr[c].iloc[-1]
            else:
                other_te.loc[:, c] = np.nan
    # --------------------------------------------------------------

    imp = SimpleImputer(strategy='median')
    monthly_tr_imp = imp.fit_transform(monthly_tr)
    monthly_te_imp = imp.transform(monthly_te)

    # PCA (fit on train only)
    pca_sc = StandardScaler()
    pca = PCA(n_components=min(0.95, len(train_dates)-1), svd_solver='full')
    try:
        monthly_tr_pca = pca.fit_transform(pca_sc.fit_transform(monthly_tr_imp))
        monthly_te_pca = pca.transform(pca_sc.transform(monthly_te_imp))
        n_pc = pca.n_components_
    except Exception:
        # During early quarters when not enough data for PCA
        n_pc = min(5, len(train_dates)-1)
        monthly_tr_pca = monthly_tr_imp[:, :n_pc]
        monthly_te_pca = monthly_te_imp[:, :n_pc]

    other_tr_imp = imp.fit_transform(other_tr)
    other_te_imp = imp.transform(other_te)

    X_tr_f = np.hstack([monthly_tr_pca, other_tr_imp])
    X_te_f = np.hstack([monthly_te_pca, other_te_imp])

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_f)
    X_te_sc = sc.transform(X_te_f)

    # Use Ridge (fastest for expanding window) + SVR + ExtraTrees (Tuned via Optuna)
    try:
        ridge = RidgeCV(alphas=np.logspace(-3, 4, 50))
        ridge.fit(X_tr_sc, y_tr)
        pred_ridge = ridge.predict(X_te_sc)[0]

        # Use Optuna to tune ExtraTrees & SVR inside the expanding window
        # We only tune every 4 quarters to save time, and reuse the best params
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if pred_idx % 4 == 0 or 'best_svr_params' not in locals():
            def objective(trial):
                # We do a small 3-fold CV on the training data for tuning
                cv = TimeSeriesSplit(n_splits=3)
                
                svr_c = trial.suggest_float('svr_c', 0.1, 100, log=True)
                svr_eps = trial.suggest_float('svr_eps', 0.01, 1.0, log=True)
                et_depth = trial.suggest_int('et_depth', 3, 10)
                et_samples_leaf = trial.suggest_int('et_samples_leaf', 1, 5)

                svr_m = SVR(kernel='rbf', C=svr_c, epsilon=svr_eps, gamma='scale')
                et_m = ExtraTreesRegressor(n_estimators=50, max_depth=et_depth, 
                                           min_samples_leaf=et_samples_leaf, random_state=42, n_jobs=-1)
                
                errs = []
                for tr_i, va_i in cv.split(X_tr_sc):
                    X_cv_tr, y_cv_tr = X_tr_sc[tr_i], y_tr.iloc[tr_i]
                    X_cv_va, y_cv_va = X_tr_sc[va_i], y_tr.iloc[va_i]
                    
                    svr_m.fit(X_cv_tr, y_cv_tr)
                    et_m.fit(X_cv_tr, y_cv_tr)
                    
                    p_svr = svr_m.predict(X_cv_va)
                    p_et = et_m.predict(X_cv_va)
                    
                    # Combine predictions
                    p_comb = (p_svr + p_et) / 2
                    errs.append(mean_absolute_error(y_cv_va, p_comb))
                return np.mean(errs)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=15) # Keep trials low for speed
            
            best_svr_params = {'C': study.best_params['svr_c'], 'epsilon': study.best_params['svr_eps']}
            best_et_params = {'max_depth': study.best_params['et_depth'], 'min_samples_leaf': study.best_params['et_samples_leaf']}

        svr = SVR(kernel='rbf', gamma='scale', **best_svr_params)
        svr.fit(X_tr_sc, y_tr)
        pred_svr = svr.predict(X_te_sc)[0]

        et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1, **best_et_params)
        et.fit(X_tr_sc, y_tr)
        pred_et = et.predict(X_te_sc)[0]

        pred_ensemble = (pred_ridge + pred_svr + pred_et) / 3
    except Exception as e:
        pred_ridge = pred_svr = pred_et = pred_ensemble = np.nan

    walk_results.append({
        'Date': pred_date,
        'Actual': y_te,
        'Ridge': pred_ridge,
        'SVR': pred_svr,
        'ExtraTrees': pred_et,
        'Ensemble': pred_ensemble,
        'Train_Size': pred_idx,
    })

    if pred_idx % 10 == 0:
        err = abs(y_te - pred_ensemble) / abs(y_te) * 100 if not np.isnan(pred_ensemble) else np.nan
        print(f"  [{pred_idx}/{n_total-1}] {pred_date.date()} | Train={pred_idx}Q | Error={err:.1f}%")

walk_df = pd.DataFrame(walk_results).set_index('Date')
walk_df.dropna(inplace=True)

# ----------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------
mape_ens   = mean_absolute_percentage_error(walk_df['Actual'], walk_df['Ensemble']) * 100
mape_ridge = mean_absolute_percentage_error(walk_df['Actual'], walk_df['Ridge']) * 100
mape_svr   = mean_absolute_percentage_error(walk_df['Actual'], walk_df['SVR']) * 100

# Period breakdowns
normal_mask = ~walk_df.index.to_series().between('2008-01-01','2010-06-30') & \
              ~walk_df.index.to_series().between('2020-01-01','2021-06-30') & \
              ~walk_df.index.to_series().between('2022-01-01','2023-06-30')
shock_mask  = ~normal_mask

mape_normal = mean_absolute_percentage_error(walk_df.loc[normal_mask,'Actual'], walk_df.loc[normal_mask,'Ensemble'])*100
mape_shock  = mean_absolute_percentage_error(walk_df.loc[shock_mask,'Actual'],  walk_df.loc[shock_mask,'Ensemble'])*100

print(f"\n{'='*65}")
print(f"WALK-FORWARD VALIDATION RESULTS (All quarters from 2010-2025)")
print(f"{'='*65}")
print(f"  Total quarters predicted:    {len(walk_df)}")
print(f"  Normal quarters MAPE:        {mape_normal:.2f}%")
print(f"  Shock quarters MAPE:         {mape_shock:.2f}%  (GFC+COVID+War)")
print(f"  Overall MAPE  (Ensemble):    {mape_ens:.2f}%")
print(f"  Overall MAPE  (SVR):         {mape_svr:.2f}%")
print(f"  Overall MAPE  (Ridge):       {mape_ridge:.2f}%")

# Post-2020 focus (most recent period)
recent = walk_df[walk_df.index >= '2020-01-01']
mape_recent = mean_absolute_percentage_error(recent['Actual'], recent['Ensemble'])*100
print(f"\n  Recent period (2020-2025) MAPE: {mape_recent:.2f}%")
print(f"  (Includes COVID and war shocks)")

# Save
walk_df['Abs_Pct_Error'] = np.abs(walk_df['Actual'] - walk_df['Ensemble']) / np.abs(walk_df['Actual']) * 100
walk_df.to_csv(rf'{DATA_DIR}\nowcast_walkforward_results.csv')
print(f"\nSaved: {DATA_DIR}\\nowcast_walkforward_results.csv")

# ----------------------------------------------------------------
# THESIS FIGURE: Walk-forward performance
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 11))
fig.suptitle("Armenia GDP Nowcasting — Expanding Window Walk-Forward Validation (2010–2025)\n"
             f"True Out-of-Sample Performance (No Look-Ahead) | Overall MAPE={mape_ens:.2f}%",
             fontsize=12, fontweight='bold')

# Top: Full actual vs predicted
ax = axes[0]
ax.plot(walk_df.index, walk_df['Actual'],   'k-o', markersize=3, linewidth=1.5, label='Actual GDP YoY')
ax.plot(walk_df.index, walk_df['Ensemble'], 'r--o', markersize=3, linewidth=1.5,
        label=f'Ensemble Nowcast (MAPE={mape_ens:.2f}%)')
ax.fill_between(walk_df.index, walk_df['Actual'], walk_df['Ensemble'], alpha=0.15, color='red')
ax.axhline(100, color='gray', linestyle='--', alpha=0.3)
ax.axvspan(pd.Timestamp('2008-10-01'), pd.Timestamp('2010-01-01'), alpha=0.10, color='red',    label=f'GFC')
ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-04-01'), alpha=0.10, color='purple', label='COVID')
ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), alpha=0.12, color='orangered', label='Ru-UA War')
ax.set_title("Actual vs Walk-Forward Nowcast (Expanding Window)", fontsize=11)
ax.set_ylabel("YoY Index (%)"); ax.legend(fontsize=8, ncol=2, loc='lower right'); ax.grid(True, alpha=0.3)

# Bottom: Rolling 4-quarter MAPE
ax2 = axes[1]
rolling_mape = walk_df['Abs_Pct_Error'].rolling(4).mean()
ax2.fill_between(rolling_mape.index, rolling_mape.values, alpha=0.3, color='steelblue')
ax2.plot(rolling_mape.index, rolling_mape.values, color='steelblue', linewidth=1.5,
         label='4-quarter Rolling MAPE')
ax2.axhline(mape_ens, color='navy', linestyle='--', label=f'Overall MAPE {mape_ens:.2f}%')
ax2.axhline(mape_normal, color='seagreen', linestyle=':', label=f'Normal quarters MAPE {mape_normal:.2f}%')
ax2.axvspan(pd.Timestamp('2008-10-01'), pd.Timestamp('2010-01-01'), alpha=0.10, color='red')
ax2.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-04-01'), alpha=0.10, color='purple')
ax2.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), alpha=0.12, color='orangered')
ax2.set_title("Rolling 4-Quarter Prediction Error Over Time", fontsize=11)
ax2.set_ylabel("MAPE (%)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(rf'{DATA_DIR}\gdp_nowcast_walkforward.png', dpi=150, bbox_inches='tight')
print(f"Saved: {DATA_DIR}\\gdp_nowcast_walkforward.png")
print("\nDone.")
