import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = r'D:\DATA\Translated_Cleaned_Nowcasting_Data.xlsx'
print("Loading Translated Data for Diagnostics...")
df_q = pd.read_excel(file_path, sheet_name='Quarterly', index_col='Date')
df_m = pd.read_excel(file_path, sheet_name='Monthly', index_col='Date')

target_abs = 'Real_GDP_Armenia_Abs'
target_yoy = 'Real_GDP_Armenia_YoY'

# ============================================================
# 1. VISUALIZE THE TARGET VARIABLE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('GDP Data Diagnostics for Armenia', fontsize=14)

# Absolute GDP
df_q[target_abs].dropna().plot(ax=axes[0, 0], title='Absolute Real GDP (Mln AMD)', color='steelblue')
axes[0, 0].set_ylabel('Mln AMD')
axes[0, 0].grid(True, alpha=0.3)

# YoY Growth
df_q[target_yoy].dropna().plot(ax=axes[0, 1], title='Real GDP YoY Growth (%)', color='green')
axes[0, 1].axhline(100, color='gray', linestyle='--', alpha=0.6)
axes[0, 1].set_ylabel('%')
axes[0, 1].grid(True, alpha=0.3)

# Log of Absolute GDP
log_gdp = np.log(df_q[target_abs].dropna())
log_gdp.plot(ax=axes[1, 0], title='Log(Absolute Real GDP)', color='orange')
axes[1, 0].set_ylabel('Log-scale')
axes[1, 0].grid(True, alpha=0.3)

# First-difference of Log GDP (= quarterly growth rate)
log_gdp_diff = log_gdp.diff().dropna() * 100
log_gdp_diff.plot(ax=axes[1, 1], title='QoQ Real GDP Growth (Log-Diff, %)', color='red')
axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.6)
axes[1, 1].set_ylabel('%')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'D:\DATA\diagnostics_gdp_forms.png', dpi=150)
plt.close()
print("Saved GDP forms chart: D:\\DATA\\diagnostics_gdp_forms.png")

# ============================================================
# 2. STATIONARITY TESTS (ADF & KPSS) on target variants
# ============================================================
print("\n" + "="*70)
print("STATIONARITY TESTS")
print("="*70)

def run_adf_kpss(series, name):
    s = series.dropna()
    adf_res = adfuller(s, autolag='AIC')
    kpss_res = kpss(s, regression='c', nlags='auto')
    adf_stat, adf_p = adf_res[0], adf_res[1]
    kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
    adf_conclusion = "Stationary" if adf_p < 0.05 else "NON-Stationary"
    kpss_conclusion = "Stationary" if kpss_p > 0.05 else "NON-Stationary"
    print(f"\n[{name}]")
    print(f"  ADF  p-value: {adf_p:.4f} → {adf_conclusion}")
    print(f"  KPSS p-value: {kpss_p:.4f} → {kpss_conclusion}")
    return adf_conclusion, kpss_conclusion

run_adf_kpss(df_q[target_abs], "Absolute Real GDP")
run_adf_kpss(np.log(df_q[target_abs].dropna()), "Log Absolute Real GDP")
run_adf_kpss(np.log(df_q[target_abs].dropna()).diff().dropna(), "Log-Diff Real GDP (QoQ Growth)")
run_adf_kpss(df_q[target_yoy].dropna(), "Real GDP YoY Growth (%)")

# ============================================================
# 3. OUTLIER DETECTION (Z-score for YoY growth)
# ============================================================
print("\n" + "="*70)
print("OUTLIER DETECTION (Z-score > 3 on YoY GDP Growth)")
print("="*70)

yoy = df_q[target_yoy].dropna()
z_scores = (yoy - yoy.mean()) / yoy.std()
outliers = z_scores[np.abs(z_scores) > 2]
print(f"\nObservations with |Z-score| > 2 (potential outliers / shocks):")
for d, z in outliers.items():
    print(f"  {d.strftime('%Y-Q')}{d.quarter}: YoY={yoy[d]:.1f}%, Z={z:.2f}")

# ============================================================
# 4. CORRELATION HEATMAP w/ exogenous quarterly variables
# ============================================================
exog_q = ['Real_GDP_Russia_YoY', 'CPI_YoY', 'Exchange_Rate_AMD_USD_YoY',
          'REER_YoY', 'Brent_Oil_Price_USD_bbl', 'Copper_Price_USD_mt']

corr_data = df_q[[target_yoy] + exog_q].dropna()
corr = corr_data.corr()
print("\n" + "="*70)
print("CORRELATIONS with GDP YoY Growth (Top Predictors):")
print("="*70)
print(corr[[target_yoy]].sort_values(target_yoy, ascending=False).to_string())

# ============================================================
# 5. MISSING VALUES SUMMARY
# ============================================================
print("\n" + "="*70)
print("MISSING VALUES SUMMARY (Quarterly Sheet):")
print("="*70)
missing = df_q.isnull().sum()
missing_pct = (missing / len(df_q) * 100).round(1)
mv = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(mv[mv['Missing Count'] > 0].sort_values('Missing %', ascending=False).to_string())

print("\n" + "="*70)
print("MISSING VALUES SUMMARY (Monthly Sheet):")
print("="*70)
missing_m = df_m.isnull().sum()
missing_pct_m = (missing_m / len(df_m) * 100).round(1)
mv_m = pd.DataFrame({'Missing Count': missing_m, 'Missing %': missing_pct_m})
print(mv_m[mv_m['Missing Count'] > 0].sort_values('Missing %', ascending=False).to_string())
