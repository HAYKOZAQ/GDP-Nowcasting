"""
Fetch additional shock-specific Google Trends.
Focus: Russian-language terms searched FROM Russia about relocating to Armenia.
These terms spiked dramatically in Feb-March 2022 = perfect leading indicator.
"""
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import time, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
pytrends = TrendReq(hl='ru-RU', tz=240, timeout=(10,30), retries=3, backoff_factor=1)
BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / 'data' / 'processed'
END_DATE = (pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
TIMEFRAME = f'2018-01-01 {END_DATE}'

# ============================================================
# NEW TERMS — specifically about 2022 Russian migration to Armenia
# All fetched geo=RU (searches originating from Russia)
# These DEFINE the shock: Russians searching for how to move to Armenia
# ============================================================
batches_ru = [
    # Relocation intent signals
    ['снять квартиру Ереван', 'переехать Армения', 'работа Ереван', 'ВНЖ Армения', 'жить Армения'],
    # Banking/money signals
    ['открыть счет Армения', 'перевести деньги Ереван', 'банк Армения открыть', 'Amera', 'Инидрам'],
    # Property signals
    ['купить квартиру Ереван', 'недвижимость Ереван купить', 'аренда Ереван', 'ипотека Армения', 'риелтор Ереван'],
    # Work/business signals
    ['работа в Ереване', 'ИТ Армения', 'бизнес Армения', 'налоги Армения', 'самозанятый Армения'],
    # Migration admin
    ['виза Армения россиянам', 'гражданство Армения', 'регистрация Ереван', 'ОВИР Армения', 'загранпаспорт Армения'],
]

# Also Armenia-domestic terms about Russian migration
batches_am = [
    # Armenian domestic signals of Russian arrival
    ['Russian expats Yerevan', 'Yerevan expats', 'Russian community Armenia', 'IT park Armenia', 'startups Armenia'],
]

print("="*60)
print("Fetching Russia-origin relocation signals (geo=RU)")
print("="*60)

all_ru_dfs = []
for i, batch in enumerate(batches_ru):
    print(f"\nBatch RU-{i+1}: {batch}")
    try:
        pytrends.build_payload(batch, timeframe=TIMEFRAME, geo='RU', cat=0)
        df = pytrends.interest_over_time()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        if not df.empty:
            all_ru_dfs.append(df)
            print(f"  ✅ Got {len(df)} months")
        else:
            print(f"  ⚠️  No data")
        time.sleep(4)
    except Exception as e:
        print(f"  ❌ {e}")
        time.sleep(8)

print("\nFetching Armenia-origin expat signals (geo=AM)")
all_am_extra = []
for batch in batches_am:
    print(f"Batch AM-expat: {batch}")
    try:
        pytrends.build_payload(batch, timeframe=TIMEFRAME, geo='AM', cat=0)
        df = pytrends.interest_over_time()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        if not df.empty:
            all_am_extra.append(df)
            print(f"  ✅ Got {len(df)} months")
        time.sleep(4)
    except Exception as e:
        print(f"  ❌ {e}")
        time.sleep(6)

# Combine and save
all_new = all_ru_dfs + all_am_extra
if all_new:
    trends_new = pd.concat(all_new, axis=1)
    trends_new.index = pd.to_datetime(trends_new.index)
    # Drop all-zero columns
    nonzero = trends_new.columns[trends_new.sum() > 0]
    trends_new = trends_new[nonzero]
    trends_q = trends_new.resample('QS').mean()
    
    print(f"\n✅ {len(nonzero)} usable new trend series")
    trends_new.to_csv(PROC_DIR / 'google_trends_shock_monthly.csv')
    trends_q.to_csv(PROC_DIR / 'google_trends_shock_quarterly.csv')
    print("Saved:")
    print(f"  {PROC_DIR / 'google_trends_shock_monthly.csv'}")
    print(f"  {PROC_DIR / 'google_trends_shock_quarterly.csv'}")
    
    print("\nTop terms by avg volume (2022 Q2 peak):")
    period_22 = trends_q[(trends_q.index >= '2022-01-01') & (trends_q.index <= '2022-12-31')]
    print(period_22.mean().sort_values(ascending=False).head(15).to_string())
else:
    print("❌ No data fetched")
