"""
=================================================================
COMPREHENSIVE WEB DATA SCRAPER FOR ARMENIA GDP NOWCASTING
=================================================================
Sources:
  1. World Bank API — Annual indicators for Armenia + trading partners
     (Armenia GDP, Remittances, FDI, Trade, Unemployment, Inflation)
     (Russia GDP, Russia Imports, Georgia GDP, EU GDP)
  2. IMF World Economic Outlook API — IMF forecasts & indicators
  3. FRED API (St. Louis Fed) — Global commodity prices & rates
  4. CBA Armenia (JSON endpoint) — Monthly remittance data

Key shock-capture variables:
  - Annual remittances (clearly shows 2022 surge)
  - China/EU trade as diversification signals
  - FDI flows into Armenia
  - Russia unemployment (signals economic stress → Armenian migration)
  - Global uncertainty (VIX, oil price volatility)
=================================================================
"""
import requests
import pandas as pd
import numpy as np
import json
import time
import io, sys
from io import StringIO

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_WB  = "https://api.worldbank.org/v2"
SAVE_DIR  = r"D:\DATA"

def wb_fetch(country, indicator, per_page=100, mrv=30, label=None):
    """Fetch World Bank indicator for a country."""
    url = f"{BASE_WB}/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": per_page, "mrv": mrv}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if len(data) < 2 or not data[1]:
            return None
        records = data[1]
        series = {rec["date"]: rec["value"] for rec in records if rec["value"] is not None}
        name = label or f"{country}_{indicator}"
        return pd.Series(series, name=name).sort_index()
    except Exception as e:
        print(f"  ❌ Error {country}/{indicator}: {e}")
        return None

print("=" * 65)
print("WORLD BANK API — Fetching shock-relevant annual indicators")
print("=" * 65)

# ================================================================
# ARMENIA INDICATORS
# ================================================================
print("\n📊 Armenia indicators...")

ARM_INDICATORS = {
    # Core shock signals
    "BX.TRF.PWKR.CD.DT":  "ARM_Remittances_USD",          # Personal remittances received ★ KEY
    "BX.KLT.DINV.CD.WD":  "ARM_FDI_Inflows_USD",           # FDI inflows
    "NE.TRD.GNFS.ZS":     "ARM_Trade_Openness_GDP_pct",    # Trade as % of GDP
    "BN.CAB.XOKA.GD.ZS":  "ARM_CurrentAccount_GDP_pct",    # Current account balance
    "FI.RES.TOTL.CD":     "ARM_FX_Reserves_USD",           # FX reserves
    "SL.UEM.TOTL.ZS":     "ARM_Unemployment_Pct",          # Unemployment rate
    "FP.CPI.TOTL.ZG":     "ARM_CPI_Inflation_Pct",         # CPI inflation
    "PA.NUS.FCRF":        "ARM_ExchangeRate_USD",           # Official exchange rate
    "NY.GDP.PCAP.KD.ZG":  "ARM_GDP_PerCapita_Growth",      # GDP per capita growth
    "SP.POP.TOTL":        "ARM_Population",                 # Population (migration proxy)
    "MS.MIL.XPND.GD.ZS":  "ARM_Military_Spend_GDP_pct",   # Military spend (conflict proxy)
}

# ================================================================
# TRADING PARTNERS — GDP and financial conditions
# ================================================================
print("📊 Trading partner indicators...")

PARTNER_INDICATORS = {
    ("RU", "NY.GDP.MKTP.KD.ZG"):  "RU_GDP_Growth",          # Russia GDP ★ already have
    ("RU", "SL.UEM.TOTL.ZS"):     "RU_Unemployment",         # Russia unemployment ★ migration signal
    ("RU", "FP.CPI.TOTL.ZG"):     "RU_Inflation",            # Russia inflation → migration push
    ("RU", "PA.NUS.FCRF"):        "RU_ExchangeRate_USD",     # Ruble/USD rate
    ("GE", "NY.GDP.MKTP.KD.ZG"):  "GE_GDP_Growth",           # Georgia GDP (competitor)
    ("TR", "NY.GDP.MKTP.KD.ZG"):  "TR_GDP_Growth",           # Turkey GDP (major trade partner)
    ("IR", "NY.GDP.MKTP.KD.ZG"):  "IR_GDP_Growth",           # Iran GDP (neighbor)
    ("EU", "NY.GDP.MKTP.KD.ZG"):  "EU_GDP_Growth",           # EU GDP (export market)
    ("CN", "NY.GDP.MKTP.KD.ZG"):  "CN_GDP_Growth",           # China GDP (global demand)
    ("US", "NY.GDP.MKTP.KD.ZG"):  "US_GDP_Growth",           # US GDP (global benchmark)
}

# ================================================================
# GLOBAL COMMODITY + FINANCIAL INDICATORS
# ================================================================
GLOBAL_INDICATORS = {
    ("WLD", "PCOPP_USD"):          None,  # handled separately
    ("WLD", "POIL_USD"):           None,
}

# Fetch Armenia indicators
arm_series = {}
for code, label in ARM_INDICATORS.items():
    s = wb_fetch("AM", code, label=label)
    if s is not None and len(s) > 0:
        arm_series[label] = s
        print(f"  ✅ {label}: {len(s)} years ({s.index.min()}–{s.index.max()})")
    time.sleep(0.5)

# Compute YoY growth for level variables
derived = {}
for key, series in arm_series.items():
    if series.dtype == float and 'Growth' not in key and 'Pct' not in key and 'Rate' not in key:
        pct = series.pct_change() * 100
        pct.name = key + '_YoY'
        derived[key + '_YoY'] = pct

arm_series.update(derived)

# Fetch trading partner indicators
print()
for (country, code), label in PARTNER_INDICATORS.items():
    s = wb_fetch(country, code, label=label)
    if s is not None and len(s) > 0:
        arm_series[label] = s
        print(f"  ✅ {label}: {len(s)} years")
    time.sleep(0.5)

# ================================================================
# COMPILE INTO ANNUAL DATAFRAME
# ================================================================
print("\nCompiling annual dataframe...")
all_series = []
for name, series in arm_series.items():
    s = series.copy()
    s.index = pd.to_numeric(s.index, errors='coerce')
    s = s.dropna()
    s.index = pd.to_datetime(s.index.astype(int).astype(str) + "-01-01")
    s.name = name
    all_series.append(s)

df_annual = pd.concat(all_series, axis=1).sort_index()
df_annual = df_annual[df_annual.index >= '2000-01-01']

# Key ratios and shock indicators
if 'ARM_Remittances_USD' in df_annual.columns and 'ARM_Population' in df_annual.columns:
    df_annual['ARM_Remittances_PerCapita'] = df_annual['ARM_Remittances_USD'] / df_annual['ARM_Population']

if 'ARM_Remittances_USD' in df_annual.columns:
    df_annual['ARM_Remittances_YoY'] = df_annual['ARM_Remittances_USD'].pct_change() * 100

if 'RU_Unemployment' in df_annual.columns and 'ARM_Unemployment_Pct' in df_annual.columns:
    # Higher Russia unemployment relative to Armenia = more migration pressure
    df_annual['RU_minus_ARM_Unemployment'] = df_annual['RU_Unemployment'] - df_annual['ARM_Unemployment_Pct']

# Save annual data
df_annual.to_csv(rf"{SAVE_DIR}\worldbank_annual_data.csv")
print(f"\nSaved annual data: {df_annual.shape[0]} years x {df_annual.shape[1]} variables")
print(f"  {SAVE_DIR}\\worldbank_annual_data.csv")

# ================================================================
# QUARTERLY STEP SERIES
# Convert annual WB data to a lagged quarterly step series instead of a
# smooth interpolation. This is less likely to create artificial within-year
# variation or leak annual information into earlier quarters.
# ================================================================
print("\nConverting annual → lagged quarterly step series...")
quarterly_idx = pd.date_range('2000-01-01', '2025-04-01', freq='QS')
df_q_interp = df_annual.reindex(df_annual.index.union(quarterly_idx)).sort_index().ffill()
df_q_interp = df_q_interp.reindex(quarterly_idx).shift(4)

# Save quarterly interpolated
df_q_interp.to_csv(rf"{SAVE_DIR}\worldbank_quarterly_interpolated.csv")
print(f"Saved quarterly data: {df_q_interp.shape[0]} quarters x {df_q_interp.shape[1]} variables")
print(f"  {SAVE_DIR}\\worldbank_quarterly_interpolated.csv")

# ================================================================
# DISPLAY KEY SHOCK SIGNALS
# ================================================================
print("\n" + "="*65)
print("KEY SHOCK SIGNALS — comparison normal vs shock years")
print("="*65)

shock_years = ['2009', '2020', '2021', '2022', '2023']
normal_years = ['2014', '2015', '2016', '2017', '2018', '2019']

key_vars = ['ARM_Remittances_YoY', 'ARM_FDI_Inflows_USD', 'ARM_Unemployment_Pct',
            'RU_Unemployment', 'RU_GDP_Growth', 'RU_Inflation',
            'ARM_CurrentAccount_GDP_pct', 'ARM_FX_Reserves_USD_YoY']
key_vars = [v for v in key_vars if v in df_annual.columns]

print(f"\n{'Variable':<40} {'Normal Avg':>12} {'2009':>8} {'2020-21 Avg':>12} {'2022':>8} {'2023':>8}")
print("-"*90)
for v in key_vars:
    def safe_mean(yr_list):
        vals = [df_annual.loc[f'{y}-01-01', v] for y in yr_list
                if f'{y}-01-01' in df_annual.index and not np.isnan(df_annual.loc[f'{y}-01-01', v])]
        return np.mean(vals) if vals else np.nan

    nm = safe_mean(normal_years)
    g09 = df_annual.get(pd.Timestamp('2009-01-01'), {}).get(v, np.nan) if hasattr(df_annual, 'get') else np.nan
    try: g09 = df_annual.loc['2009-01-01', v]
    except: g09 = np.nan
    try: g2021 = (df_annual.loc['2020-01-01', v] + df_annual.loc['2021-01-01', v]) / 2
    except: g2021 = np.nan
    try: g22 = df_annual.loc['2022-01-01', v]
    except: g22 = np.nan
    try: g23 = df_annual.loc['2023-01-01', v]
    except: g23 = np.nan

    print(f"  {v:<38} {nm:>12.1f} {g09:>8.1f} {g2021:>12.1f} {g22:>8.1f} {g23:>8.1f}")

print(f"\n✅ Done. Use worldbank_quarterly_interpolated.csv in nowcasting model.")
