"""
Google Trends extractor for Armenia GDP Nowcasting.
Downloads monthly search volume for shock-relevant keywords.
Saves to D:\DATA\google_trends_armenia.csv
"""
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import time
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / 'data' / 'processed'
END_DATE = (pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
TIMEFRAME = f'2018-01-01 {END_DATE}'

# ============================================================
# KEYWORD STRATEGY (from Kohns & Bhattacharjee 2022 paper)
# ---------------------------------------------------------------
# The paper shows search terms with HIGH inclusion probability
# reflect: (a) Leading economic signals, (b) Anxiety/uncertainty
#
# For ARMENIA specifically, the most informative terms are:
#
# A. Migration/Relocation (key driver of 2022 GDP surge)
#    - "relocation Armenia" (English)
#    - "переезд Армения" (Russian: relocation to Armenia)
#    - "квартира Ереван" (Russian: apartment Yerevan)
#
# B. Remittances (key driver of Armenian GDP)
#    - "money transfer Armenia" (English)
#    - "перевод денег Армения" (Russian: money transfer Armenia)
#
# C. Economic Uncertainty (classic nowcasting signals)
#    - "Armenia economy"
#    - "Armenian dram"
#    - "USD AMD" (exchange rate)
#
# D. Trade Activity
#    - "business Armenia"
# ============================================================

# We collect in 5 separate batches of 5 (max per pytrends request = 5)
# Strategy: pull monthly data from 2018 to present (max range for monthly)
# Then chain with yearly data for 2004-2018 as relative index

all_results = {}

def fetch_trends(keywords, timeframe=TIMEFRAME, geo_filter='', cat=0):
    """Fetch Google Trends for a list of up to 5 keywords."""
    try:
        pytrends.build_payload(keywords, timeframe=timeframe, geo=geo_filter, cat=cat)
        df = pytrends.interest_over_time()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        print(f"  ✅ Fetched: {keywords}")
        time.sleep(3)  # Respect rate limits
        return df
    except Exception as e:
        print(f"  ❌ Error fetching {keywords}: {e}")
        time.sleep(5)
        return pd.DataFrame()

print("="*60)
print("Fetching Google Trends Data for Armenia Nowcasting")
print("="*60)

# === BATCH 1: Relocation & Migration (English) ===
print("\nBatch 1: Relocation & Migration (English)...")
batch1 = fetch_trends(
    ['Armenia relocation', 'move to Armenia', 'live in Armenia', 'Yerevan apartment', 'visa Armenia'],
    timeframe=TIMEFRAME
)

# === BATCH 2: Remittances & Finance ===
print("Batch 2: Remittances & Finance...")
batch2 = fetch_trends(
    ['money transfer Armenia', 'Armenian dram', 'USD AMD', 'Armenia investment', 'Armenia GDP'],
    timeframe=TIMEFRAME
)

# === BATCH 3: Russian-language relocation search terms (key 2022 signal) ===
print("Batch 3: Russian-language relocation terms...")
batch3 = fetch_trends(
    ['переезд Армения', 'квартира Ереван', 'работа Армения', 'Армения недвижимость', 'эмиграция Армения'],
    timeframe=TIMEFRAME
)

# === BATCH 4: Economic uncertainty & inflation ===
print("Batch 4: Economic uncertainty...")
batch4 = fetch_trends(
    ['Armenia economy', 'Yerevan prices', 'Armenia business', 'Armenia inflation', 'Armenia tourism'],
    timeframe=TIMEFRAME
)

# === BATCH 5: Russian geo-targeted search for "Armenia" ===
print("Batch 5: Russia-origin search for Армения (geo=RU)...")
batch5 = fetch_trends(
    ['Армения', 'Ереван', 'виза Армения', 'банк Армения', 'цены Ереван'],
    timeframe=TIMEFRAME,
    geo_filter='RU'  # searches originating from Russia are most predictive for Armenia GDP
)

# === COMBINE ALL BATCHES ===
print("\nCombining all batches...")
all_batches = [b for b in [batch1, batch2, batch3, batch4, batch5] if not b.empty]

if all_batches:
    trends_monthly = pd.concat(all_batches, axis=1)
    trends_monthly.index = pd.to_datetime(trends_monthly.index)
    trends_monthly.sort_index(inplace=True)
    # Resample to quarterly
    trends_quarterly = trends_monthly.resample('QS').mean()
    
    print(f"\n✅ Downloaded {len(trends_monthly.columns)} trend series")
    print(f"   Monthly range: {trends_monthly.index.min().date()} to {trends_monthly.index.max().date()}")
    print(f"   Total months: {len(trends_monthly)}")
    
    trends_monthly.to_csv(PROC_DIR / 'google_trends_armenia_monthly.csv')
    trends_quarterly.to_csv(PROC_DIR / 'google_trends_armenia_quarterly.csv')
    print(f"\nSaved:")
    print(f"  Monthly:   {PROC_DIR / 'google_trends_armenia_monthly.csv'}")
    print(f"  Quarterly: {PROC_DIR / 'google_trends_armenia_quarterly.csv'}")
    
    # Show head
    print("\nFirst rows of quarterly data:")
    print(trends_quarterly.tail(10).to_string())
else:
    print("❌ No data fetched! Possible rate-limiting from Google. Try again later.")
