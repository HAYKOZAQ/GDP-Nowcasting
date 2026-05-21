"""
Fetch Armenian-language & locally-relevant Google Trends for GDP Nowcasting.
48 exact terms provided by the user, geo-filtered to Armenia (geo='AM').
"""
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import time, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pytrends = TrendReq(hl='hy-AM', tz=240, timeout=(10, 30), retries=3, backoff_factor=1)
BASE_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = BASE_DIR / 'data' / 'processed'
END_DATE = (pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
TIMEFRAME = f'2018-01-01 {END_DATE}'

# ----------------------------------------------------------------
# 48 exact user-provided keywords (translated category notes):
# ----------------------------------------------------------------
terms = [
    # Financial products
    'վarkeliq',             # loan
    'աshkhataank',          # work/job search
    'բnakaran',             # apartment
    'hipoteq',              # mortgage
    'pokhandzum',           # transfer/remittance
    'shnararutyun',         # construction
    'ansharzh guyq',        # real estate
    'ashkhatavarzeq',       # salary/wage
    'mekena',               # car
    'zbosashrzhutyun',      # tourism
    'hyuranots',            # hotel
    'nerdrum',              # investment
    'tokosadruyk',          # interest rate
    'pokharzhekh',          # exchange rate
    'spavrokanakin varkh',  # consumer loan
    'bankain varkh',        # bank loan
    'avandh',               # deposit
    'tapvur ashkhatatagekh', # job vacancy
    'gnach',                # inflation
    'fnanser',              # finance
    # Platforms & brands
    'MoneyGram',
    'turizm',               # tourism
    'gazi gin',             # gas price
    'harker',               # taxes
    'artahanum',            # export
    'nermutum',             # import
    'voski',                # gold
    'apahovagrutjun',       # insurance
    'varkhayin patmutyun',  # credit history
    'ekamut',               # income
    'Wildberries',          # Russian marketplace popular in Armenia
    'dolari kurs',          # dollar rate
    'Rate.am',              # Armenian financial comparison site
    'Auto.am',              # Armenian car marketplace
    'avtomekena',           # automobile
    'iPhone',               # iPhone (consumer spending proxy)
    'onlayn varkh',         # online loan
    'List.am',              # Armenian classifieds
    'oravarzov',            # daily rent
    'Estate.am',            # Armenian real estate site
    'Ideal system',         # Armenian payment system
    'Staff.am',             # Armenian jobs platform
    'работа в армении',     # work in Armenia (Russian)
    'Zvartnots',            # Yerevan airport (tourism proxy)
    'Aviasales',            # flight booking
    'Booking.com',          # hotel booking
    'Idram',                # Armenian e-wallet
    'Telcell',              # Armenian payment system
]

# Split into batches of 5 (pytrends limit)
def make_batches(lst, n=5):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

batches = make_batches(terms, 5)
print("="*60)
print(f"Fetching {len(terms)} terms in {len(batches)} batches (geo=AM, 2018-2025)")
print("="*60)

all_dfs = []
for i, batch in enumerate(batches):
    print(f"\nBatch {i+1}/{len(batches)}: {batch}")
    try:
        pytrends.build_payload(
            batch,
            timeframe=TIMEFRAME,
            geo='AM',   # Armenia only
            cat=0
        )
        df = pytrends.interest_over_time()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        if not df.empty:
            all_dfs.append(df)
            print(f"  ✅ Got {len(df)} months")
        else:
            print(f"  ⚠️  No data returned")
        time.sleep(4)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        time.sleep(8)

if not all_dfs:
    print("\n❌ No data was fetched. Possible reasons:")
    print("  - Rate limiting by Google. Wait 10 min and retry.")
    sys.exit(1)

# Combine all batches
trends_monthly = pd.concat(all_dfs, axis=1)
trends_monthly.index = pd.to_datetime(trends_monthly.index)
trends_monthly.sort_index(inplace=True)

# Remove columns that are entirely zero (no search data in Armenia)
nonzero = trends_monthly.columns[trends_monthly.sum() > 0]
dropped = set(trends_monthly.columns) - set(nonzero)
if dropped:
    print(f"\n  ⚠️  Dropped {len(dropped)} all-zero columns: {list(dropped)}")
trends_monthly = trends_monthly[nonzero]

# Quarterly average
trends_q = trends_monthly.resample('QS').mean()

print(f"\n{'='*60}")
print(f"RESULT: {len(nonzero)} usable series from {len(terms)} attempted")
print(f"Date range: {trends_monthly.index.min().date()} to {trends_monthly.index.max().date()}")
print(f"\nTop 20 terms by average monthly search volume in Armenia:")
print(trends_monthly.mean().sort_values(ascending=False).head(20).to_string())

# Save
trends_monthly.to_csv(PROC_DIR / 'google_trends_armenian_monthly.csv')
trends_q.to_csv(PROC_DIR / 'google_trends_armenian_quarterly.csv')
print(f"\nSaved:")
print(f"  {PROC_DIR / 'google_trends_armenian_monthly.csv'}")
print(f"  {PROC_DIR / 'google_trends_armenian_quarterly.csv'}")
