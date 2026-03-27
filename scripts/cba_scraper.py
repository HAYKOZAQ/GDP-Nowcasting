import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os

# Configuration
BASE_DIR = 'D:/DATA/GDP_NOWCASTING_DASHBOARD'
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data_raw')
os.makedirs(DATA_RAW_DIR, exist_ok=True)

def fetch_exchange_rates(date_from, date_to):
    """Fetches exchange rates from CBA SOAP API using ExchangeRatesByDateRangeByISO."""
    url = "https://api.cba.am/exchangerates.asmx"
    headers = {
        'Content-Type': 'text/xml; charset=utf-8',
        'SOAPAction': 'http://www.cba.am/ExchangeRatesByDateRangeByISO'
    }
    
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <ExchangeRatesByDateRangeByISO xmlns="http://www.cba.am/">
      <ISOCodes>USD,RUB</ISOCodes>
      <DateFrom>{date_from}</DateFrom>
      <DateTo>{date_to}</DateTo>
    </ExchangeRatesByDateRangeByISO>
  </soap:Body>
</soap:Envelope>"""

    try:
        response = requests.post(url, data=soap_body, headers=headers, timeout=30)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            ns = {'cba': 'http://www.cba.am/'}
            data = []
            # The returned structure for ExchangeRatesByDateRangeByISO might differ slightly
            # It usually returns a DataSet or a collection of ExchangeRates
            for day in root.findall('.//cba:ExchangeRates', ns):
                date = day.find('cba:Date', ns).text
                for rate in day.findall('cba:Rates/cba:ExchangeRate', ns):
                    iso = rate.find('cba:ISO', ns).text
                    value = rate.find('cba:Rate', ns).text
                    data.append({'Date': date, 'ISO': iso, 'Rate': value})
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(DATA_RAW_DIR, 'exchange_rates_raw.csv'), index=False)
                print(f"Exchange rates saved. Count: {len(df)}")
            else:
                print("No exchange rate data found in response.")
        else:
            print(f"Error fetching exchange rates: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception in fetch_exchange_rates: {e}")

def download_cba_excel(url, filename):
    """Downloads Excel files from CBA with User-Agent."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            path = os.path.join(DATA_RAW_DIR, filename)
            with open(path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Error downloading {filename}: {response.status_code} for URL {url}")
    except Exception as e:
        print(f"Exception downloading {filename}: {e}")

if __name__ == "__main__":
    # Date range for 2024-2025 (Standard ISO YYYY-MM-DD)
    today = datetime.now().strftime("%Y-%m-%d")
    start_date = "2024-01-01"
    
    print(f"Fetching USD/RUB from {start_date} to {today}...")
    fetch_exchange_rates(start_date, today)
    
    # Banking indicators
    print("Downloading banking indicators...")
    download_cba_excel("https://www.cba.am/stat/stat_data_eng/6_loans%20by%20sectors_eng.xlsx", "loans_sectors.xlsx")
    download_cba_excel("https://www.cba.am/stat/stat_data_eng/5_Deposits%20by%20sectors_eng.xlsx", "deposits_sectors.xlsx")
