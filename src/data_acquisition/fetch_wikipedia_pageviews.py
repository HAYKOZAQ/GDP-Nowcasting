"""
Fetch Wikipedia pageviews for Armenian-related articles on Russian Wikipedia.
The Russian Wikipedia page for 'Армения' spiked dramatically in Sep 2022
when Putin announced mobilization and Russians began searching for Armenia.
"""
import requests
import pandas as pd
import numpy as np
import time, io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
HEADERS = {"User-Agent": "Armenia-GDP-Nowcasting-Research/1.0"}

def get_pageviews(project, article, start="2015070100", end="2025030100"):
    url = f"{BASE}/{project}/all-access/user/{article}/monthly/{start}/{end}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30) # Increased timeout
        data = r.json()
        if 'items' not in data:
            return None
        records = {item['timestamp'][:6]: item['views'] for item in data['items']}
        return pd.Series(records, name=article.replace('%20','_').replace('%D0','').replace('%90',''))
    except Exception as e:
        print(f"  ❌ {article}: {e}")
        return None

# Articles to fetch (all Russian Wikipedia for Russian-language searches)
articles_ru = {
    # Core articles about Armenia
    'ru.wikipedia': [
        '%D0%90%D1%80%D0%BC%D0%B5%D0%BD%D0%B8%D1%8F',           # "Армения"
        '%D0%95%D1%80%D0%B5%D0%B2%D0%B0%D0%BD',                  # "Ереван"
        '%D0%90%D0%BC%D1%8F%D0%BD%D1%81%D0%BA%D0%B8%D0%B9_%D0%B4%D1%80%D0%B0%D0%BC', # "Армянский драм"
        '%D0%93%D1%80%D0%B0%D0%B6%D0%B4%D0%B0%D0%BD%D1%81%D1%82%D0%B2%D0%BE_%D0%90%D1%80%D0%BC%D0%B5%D0%BD%D0%B8%D0%B8', # "Гражданство Армении"
    ],
    # English Wikipedia for global interest
    'en.wikipedia': [
        'Armenia',
        'Yerevan',
        'Russian_mobilization_2022',
        'Armenian_dram',
    ]
}

print("Fetching Wikipedia pageviews...")
all_series = []

for project, articles in articles_ru.items():
    for article in articles:
        label = f"WIKI_{project.split('.')[0].upper()}_{article[:20].replace('%','').replace('D0','').replace('D1','')}"
        print(f"  {project}: {article[:30]}...")
        s = get_pageviews(project, article)
        if s is not None and len(s) > 0:
            s.name = label
            s.index = pd.to_datetime(s.index, format='%Y%m')
            all_series.append(s)
            print(f"    ✅ {len(s)} months, max={s.max():,.0f} views")
        time.sleep(1)

if all_series:
    df_wiki = pd.concat(all_series, axis=1)
    df_wiki.index = pd.to_datetime(df_wiki.index)
    df_wiki = df_wiki.sort_index()
    
    # Show 2022 spike
    print("\nWikipedia 2022 monthly views (key spike period):")
    period = df_wiki[(df_wiki.index >= '2022-01-01') & (df_wiki.index <= '2023-01-01')]
    print(period.to_string())
    
    # Save monthly
    df_wiki.to_csv(r'D:\DATA\wikipedia_pageviews_monthly.csv')
    
    # Aggregate to quarterly
    wiki_q = df_wiki.resample('QS').mean()
    # Add YoY growth
    for col in wiki_q.columns:
        wiki_q[f'{col}_YoY'] = wiki_q[col].pct_change(4) * 100
    wiki_q.to_csv(r'D:\DATA\wikipedia_pageviews_quarterly.csv')
    
    print(f"\nSaved:")
    print(f"  D:\\DATA\\wikipedia_pageviews_monthly.csv")
    print(f"  D:\\DATA\\wikipedia_pageviews_quarterly.csv")
    print(f"  {len(df_wiki.columns)} articles tracked")
else:
    print("No data fetched.")
