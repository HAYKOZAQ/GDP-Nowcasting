"""
COMPREHENSIVE DATA ANALYSIS FOR ARMENIA GDP NOWCASTING
Answers: Which variables minimize prediction error?
Covers:
  1. Correlation ranking (all quarterly variables vs GDP YoY)
  2. Feature importance from Random Forest (trained on full data)
  3. Lagged correlations (which variables LEAD GDP by 1-4 quarters?)
  4. Shock-period analysis (what spiked during GFC, COVID, War?)
  5. Monthly variables correlation with quarterly GDP
  6. Google Trends correlation with GDP
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, io, sys

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = str(BASE_DIR)
file_path = BASE_DIR / 'data' / 'raw' / 'Translated_Cleaned_Nowcasting_Data.xlsx'
print("Loading data...")
df_q = pd.read_excel(file_path, sheet_name='Quarterly', index_col='Date')
df_m = pd.read_excel(file_path, sheet_name='Monthly',   index_col='Date')
trends_global = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'google_trends_armenia_quarterly.csv',   index_col='date', parse_dates=True)
trends_local  = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'google_trends_armenian_quarterly.csv',  index_col=0,      parse_dates=True)
trends_global.index.name = trends_local.index.name = 'Date'

# Add BOP/income features
df_q['Primary_Income_YoY']     = df_q['Primary_Income_Labor_Mln_USD'].pct_change(4) * 100
df_q['Secondary_Income_YoY']   = df_q['Secondary_Income_Transfers_Mln_USD'].pct_change(4) * 100
df_q['AMD_USD_StrongSignal']   = (100 - df_q['Exchange_Rate_AMD_USD_YoY'])
df_q['AMD_RUB_Level_QoQ']      = df_q['Exchange_Rate_AMD_RUB_Abs'].pct_change() * 100
df_q['Migration_Inflow_Signal']= (df_q['AMD_USD_StrongSignal'].clip(lower=0) *
                                   df_q['Primary_Income_YoY'].clip(lower=0)) / 100
df_q['Salary_YoY_Lag1']        = df_q['Average_Nominal_Salary_YoY'].shift(1)
df_q['REER_Surge']             = (df_q['REER_YoY'] - 100).clip(lower=0)

TARGET = 'Real_GDP_Armenia_YoY'

# ================================================================
# 1. CORRELATION RANKING — ALL QUARTERLY VARIABLES
# ================================================================
print("\n" + "="*70)
print("1. CORRELATION WITH GDP YoY (Quarterly Variables)")
print("="*70)

# Exclude GDP components (leakage) and absolute GDP
exclude = ['Real_GDP_Armenia_Abs','Real_GDP_Armenia_YoY','Nominal_GDP_Mln_AMD',
           'Industry_Real_Growth_YoY','Agriculture_Real_Growth_YoY',
           'Construction_Real_Growth_YoY','Services_Real_Growth_YoY',
           'Net_Indirect_Taxes_YoY','Real_Consumption_YoY','Real_Private_Consumption_YoY',
           'Real_Government_Consumption_YoY','Real_Aggregate_Investments_YoY',
           'Real_Fixed_Capital_Investments_YoY','Real_Private_Investments_YoY',
           'Real_Government_Investments_YoY','Real_Exports_YoY','Real_Imports_YoY',
           'Real_Private_Consumption_Abs','Real_Private_Investments_Abs',
           'Real_Construction_Abs','Real_Disposable_Income_Abs',
           'Primary_Income_Mln_AMD','Secondary_Income_Mln_AMD','Disposable_Income_Mln_AMD']

safe_cols = [c for c in df_q.columns if c not in exclude and '_QoQ' not in c]
corr_data = df_q[[TARGET] + [c for c in safe_cols if c != TARGET]].dropna(subset=[TARGET])
corrs = corr_data.corr()[TARGET].drop(TARGET).dropna().sort_values(key=abs, ascending=False)

print(f"\nTop 30 correlated quarterly variables:")
print(corrs.head(30).to_string())

# ================================================================
# 2. LAGGED CORRELATION — WHICH VARIABLES LEAD GDP?
# ================================================================
print("\n" + "="*70)
print("2. LAGGED CORRELATION (Leading Indicators)")
print("="*70)
print("Variables with highest correlation with GDP 1-4 quarters AHEAD")

lead_results = []
for col in safe_cols:
    if col == TARGET:
        continue
    series = df_q[col].dropna()
    for lag in [1, 2, 3, 4]:
        shifted = series.shift(lag)
        combined = pd.concat([df_q[TARGET], shifted], axis=1).dropna()
        if len(combined) > 20:
            c = combined.corr().iloc[0, 1]
            lead_results.append({'Variable': col, 'Lead_Quarters': lag, 'Correlation': c})

lead_df = pd.DataFrame(lead_results)
lead_df['Abs_Corr'] = lead_df['Correlation'].abs()
lead_df = lead_df.sort_values('Abs_Corr', ascending=False)
print(f"\nTop 20 leading quarterly indicators:")
print(lead_df.head(20)[['Variable','Lead_Quarters','Correlation']].to_string(index=False))

# ================================================================
# 3. MONTHLY VARIABLES CORRELATION WITH QUARTERLY GDP
# ================================================================
print("\n" + "="*70)
print("3. MONTHLY VARIABLES CORRELATION WITH QUARTERLY GDP")
print("="*70)

# Aggregate monthly to quarterly
df_m_q = df_m.resample('QS').mean()
# Compute YoY for monetary levels
money_levels = ['Cash_in_Circulation_Mln_AMD','Money_Supply_M2_Mln_AMD','Money_Supply_M2X_Mln_AMD',
                'Commercial_Bank_Loans_Mln_AMD','Household_Loans_Mln_AMD','Total_Loans_Mln_AMD']
for col in money_levels:
    if col in df_m.columns:
        m_yoy = df_m[col].pct_change(12)*100
        df_m_q[f'{col}_YoY'] = m_yoy.resample('QS').mean()

monthly_corr_data = pd.concat([df_q[TARGET], df_m_q], axis=1).dropna(subset=[TARGET])
m_corrs = monthly_corr_data.corr()[TARGET].drop(TARGET).dropna().sort_values(key=abs, ascending=False)
print(f"\nTop 20 correlated monthly variables (aggregated to quarterly):")
print(m_corrs.head(20).to_string())

# ================================================================
# 4. GOOGLE TRENDS CORRELATION
# ================================================================
print("\n" + "="*70)
print("4. GOOGLE TRENDS CORRELATION WITH GDP YoY")
print("="*70)

gt_combined = pd.concat([df_q[TARGET], trends_global, trends_local], axis=1).dropna(subset=[TARGET])
gt_corrs = gt_combined.corr()[TARGET].drop(TARGET).dropna().sort_values(key=abs, ascending=False)
print(f"\nAll Google Trends correlations with GDP YoY:")
print(gt_corrs.to_string())

# ================================================================
# 5. SHOCK-PERIOD ANALYSIS — What variables spiked during shocks?
# ================================================================
print("\n" + "="*70)
print("5. SHOCK-PERIOD VARIABLE BEHAVIOR vs NORMAL PERIODS")
print("="*70)

gfc   = df_q[(df_q.index >= '2008-10-01') & (df_q.index <= '2010-01-01')]
covid = df_q[(df_q.index >= '2020-01-01') & (df_q.index <= '2021-04-01')]
war   = df_q[(df_q.index >= '2022-01-01') & (df_q.index <= '2023-06-30')]
normal= df_q[~df_q.index.to_series().between('2008-10-01','2010-01-01') &
             ~df_q.index.to_series().between('2020-01-01','2021-04-01') &
             ~df_q.index.to_series().between('2022-01-01','2023-06-30')]

key_shock_vars = ['Real_GDP_Russia_YoY','Exchange_Rate_AMD_USD_YoY','REER_YoY',
                  'CPI_YoY','Brent_Oil_Price_USD_bbl','Primary_Income_YoY',
                  'Migration_Inflow_Signal','Disposable_Income_YoY',
                  'AMD_USD_StrongSignal','Employment_YoY','Average_Nominal_Salary_YoY']

print(f"\n{'Variable':<35} {'Normal':>10} {'GFC':>10} {'COVID':>10} {'RU-War':>10}")
print("-"*75)
for v in key_shock_vars:
    if v in df_q.columns:
        nm = normal[v].mean(); gf = gfc[v].mean(); cv = covid[v].mean(); wr = war[v].mean()
        print(f"{v:<35} {nm:>10.1f} {gf:>10.1f} {cv:>10.1f} {wr:>10.1f}")

# ================================================================
# 6. FEATURE IMPORTANCE (Random Forest on full dataset)
# ================================================================
print("\n" + "="*70)
print("6. TREE-BASED FEATURE IMPORTANCE (Random Forest, full data)")
print("="*70)

safe_feats = [c for c in safe_cols if c != TARGET and c in df_q.columns]
rf_data = df_q[[TARGET]+safe_feats].dropna(subset=[TARGET]).copy()
imp_imp = SimpleImputer(strategy='median')
X_rf = pd.DataFrame(imp_imp.fit_transform(rf_data.drop(columns=[TARGET])),
                    columns=rf_data.columns[1:], index=rf_data.index)
y_rf = rf_data[TARGET]

rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(StandardScaler().fit_transform(X_rf), y_rf)

fi = pd.DataFrame({'Feature': X_rf.columns, 'Importance': rf.feature_importances_})
fi = fi.sort_values('Importance', ascending=False)
print("\nTop 25 features by Random Forest importance (quarterly features only):")
print(fi.head(25).to_string(index=False))

# ================================================================
# VISUALIZATION
# ================================================================
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)
fig.suptitle("Data Analysis: Feature Predictive Power for Armenian GDP Nowcasting",
             fontsize=13, fontweight='bold')

# Panel 1: Top correlations (quarterly)
ax1 = fig.add_subplot(gs[0, 0])
top_corr = corrs.head(15)
colors1 = ['seagreen' if v > 0 else 'tomato' for v in top_corr.values]
short_names = [n.replace('Real_GDP_Russia_YoY','RU GDP YoY')
               .replace('Exchange_Rate_AMD_USD_YoY','AMD/USD YoY')
               .replace('Migration_Inflow_Signal','Migration Signal')
               .replace('Primary_Income_YoY','Primary Income YoY')
               .replace('Disposable_Income_YoY','Disposable Income YoY')
               .replace('REER_Surge','REER Surge')
               .replace('Employment_YoY','Employment YoY')
               .replace('AMD_USD_StrongSignal','AMD Strong')
               .replace('Unemployment_Rate_Pct','Unemployment %')[:35]
               for n in top_corr.index]
ax1.barh(range(len(short_names)), top_corr.values, color=colors1, edgecolor='white')
ax1.set_yticks(range(len(short_names))); ax1.set_yticklabels(short_names[::-1][::-1], fontsize=8)
ax1.set_title("Top 15 Correlated Quarterly Variables", fontsize=10)
ax1.set_xlabel("Pearson Correlation with GDP YoY"); ax1.grid(True, alpha=0.3, axis='x')
ax1.axvline(0, color='black', linewidth=0.8)

# Panel 2: Monthly correlations
ax2 = fig.add_subplot(gs[0, 1])
top_m = m_corrs.head(15)
colors2 = ['steelblue' if v > 0 else 'tomato' for v in top_m.values]
short_m = [n.replace('_Mln_AMD_YoY','_YoY').replace('_YoY','').replace('_AMD','')
           .replace('Short_Term_Nominal_Interest_Rate_','IR_')
           .replace('Long_Term_Nominal_Interest_Rate_','IR_LT_')
           .replace('Money_Supply_M2X','M2X').replace('Money_Supply_M2','M2')
           .replace('Cash_in_Circulation','Cash')[:30]
           for n in top_m.index]
ax2.barh(range(len(short_m)), top_m.values, color=colors2, edgecolor='white')
ax2.set_yticks(range(len(short_m))); ax2.set_yticklabels(short_m[::-1][::-1], fontsize=8)
ax2.set_title("Top 15 Correlated Monthly Variables", fontsize=10)
ax2.set_xlabel("Pearson Correlation with GDP YoY"); ax2.grid(True, alpha=0.3, axis='x')
ax2.axvline(0, color='black', linewidth=0.8)

# Panel 3: RF Feature importances
ax3 = fig.add_subplot(gs[1, 0])
top_fi = fi.head(12)
short_fi = [n.replace('Real_GDP_Russia_YoY','RU GDP YoY')
            .replace('Migration_Inflow_Signal','Migration Signal')
            .replace('Primary_Income_YoY','Primary Income YoY')
            .replace('Disposable_Income_YoY','Disposable Inc YoY')
            .replace('Exchange_Rate_AMD_USD_YoY','AMD/USD YoY')
            .replace('REER_Surge','REER Surge')
            .replace('AMD_USD_StrongSignal','AMD Strong')
            .replace('Average_Nominal_Salary_YoY','Salary YoY')
            .replace('Salary_YoY_Lag1','Salary YoY Lag1')
            .replace('Employment_YoY','Employment YoY')
            .replace('Unemployment_Rate_Pct','Unemployment %')[:35]
            for n in top_fi['Feature']]
ax3.barh(range(len(top_fi)), top_fi['Importance'].values, color='mediumpurple', edgecolor='white')
ax3.set_yticks(range(len(top_fi))); ax3.set_yticklabels(short_fi[::-1][::-1], fontsize=8)
ax3.set_title("Random Forest Feature Importance (Quarterly)", fontsize=10)
ax3.set_xlabel("Importance"); ax3.grid(True, alpha=0.3, axis='x')

# Panel 4: Google Trends correlations
ax4 = fig.add_subplot(gs[1, 1])
top_gt = gt_corrs.head(15)
colors4 = ['darkorange' if v > 0 else 'tomato' for v in top_gt.values]
short_gt = [n[:35] for n in top_gt.index]
ax4.barh(range(len(top_gt)), top_gt.values, color=colors4, edgecolor='white')
ax4.set_yticks(range(len(top_gt))); ax4.set_yticklabels(short_gt[::-1][::-1], fontsize=8)
ax4.set_title("Top 15 Google Trends Correlations with GDP", fontsize=10)
ax4.set_xlabel("Pearson Correlation with GDP YoY"); ax4.grid(True, alpha=0.3, axis='x')
ax4.axvline(0, color='black', linewidth=0.8)

plt.savefig(rf'{DATA_DIR}\feature_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {DATA_DIR}\\feature_analysis.png")

# Save summary to CSV
summary = pd.DataFrame({
    'Quarterly_Var': corrs.index[:30].tolist() + ['']*(30-min(30,len(corrs))),
    'Q_Correlation': corrs.values[:30].tolist() + [np.nan]*(30-min(30,len(corrs))),
    'Monthly_Var': m_corrs.index[:30].tolist() + ['']*(30-min(30,len(m_corrs))),
    'M_Correlation': m_corrs.values[:30].tolist() + [np.nan]*(30-min(30,len(m_corrs))),
})
summary.to_csv(rf'{DATA_DIR}\feature_correlation_summary.csv', index=False)
print(f"Saved: {DATA_DIR}\\feature_correlation_summary.csv")
