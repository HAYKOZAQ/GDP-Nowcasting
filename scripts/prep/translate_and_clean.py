import pandas as pd
import io
import sys
import numpy as np

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = r'D:\DATA\GDP_NOWCASTING\Nowcasting DATA_for NIFI(1).xlsx'

print("Loading data...")
xls = pd.ExcelFile(file_path)
df_monthly = pd.read_excel(xls, 'Monthly')
df_quarterly = pd.read_excel(xls, 'Quartly')

def clean_cols(df):
    df.columns = [str(col).replace('\n', ' ').strip() for col in df.columns]
    return df

df_monthly = clean_cols(df_monthly)
df_quarterly = clean_cols(df_quarterly)

# Dictionary for Quarterly Translations
quarterly_translations = {
    'Unnamed: 0': 'Quarter',
    'Реальный  ВВП РФ, YoY': 'Real_GDP_Russia_YoY',
    'Реальный  ВВП  РА, YoY': 'Real_GDP_Armenia_YoY',
    'Реальный  располагаемых  доход, YoY': 'Real_Disposable_Income_YoY_1',
    'Промышленность  (реальный рост), YoY': 'Industry_Real_Growth_YoY',
    'Сельско хозяйства (реальный рост),  YoY': 'Agriculture_Real_Growth_YoY',
    'Строительство (реальный рост), YoY': 'Construction_Real_Growth_YoY',
    'Сфера услуг (реальный рост), YoY': 'Services_Real_Growth_YoY',
    'Чистые  косвенные  налоги,  YoY': 'Net_Indirect_Taxes_YoY',
    'Реальная  потребления,  YoY': 'Real_Consumption_YoY',
    'Реальная  частная  потребления,  YoY': 'Real_Private_Consumption_YoY',
    'Реальная  государственная  потребления,  YoY': 'Real_Government_Consumption_YoY',
    'Реальные   совокупные  инвестиции,  YoY': 'Real_Aggregate_Investments_YoY',
    'Реальные   инвестиции  в основном  капитале,  YoY': 'Real_Fixed_Capital_Investments_YoY',
    'Реальные   частные  инвестиции,  YoY': 'Real_Private_Investments_YoY',
    'Реальные   государственные  инвестиции,  YoY': 'Real_Government_Investments_YoY',
    'Реальный  экспорт,  YoY': 'Real_Exports_YoY',
    'Реальный  импорт,  YoY': 'Real_Imports_YoY',
    'Рост  безработицы,  YoY': 'Unemployment_Growth_YoY',
    'Занятость,  YoY': 'Employment_YoY',
    'Наёмные  работники,  YoY': 'Hired_Workers_YoY',
    'Средная  номинальная  зарплата,  YoY': 'Average_Nominal_Salary_YoY',
    'ИПЦ,  YoY': 'CPI_YoY',
    'Обменный  курс  драм/доллар  США,  YoY': 'Exchange_Rate_AMD_USD_YoY',
    'Реальный  эффективный  обменный  курс,  YoY': 'REER_YoY',
    'Уровень  безработицы,  %': 'Unemployment_Rate_Pct',
    'Реальный  располагаемый доход': 'Real_Disposable_Income_Abs',
    'Реальный  ВВП': 'Real_GDP_Armenia_Abs',
    'Реальное  частное  потребление': 'Real_Private_Consumption_Abs',
    'Реальные частные  инвестиции': 'Real_Private_Investments_Abs',
    'Реальное  строительство': 'Real_Construction_Abs',
    'Обменный курс  драм/доллар  США': 'Exchange_Rate_AMD_USD_Abs',
    'Обменный  курс драм/рубль  РФ': 'Exchange_Rate_AMD_RUB_Abs',
    'Цена на нефть марки Brent, $/bbl': 'Brent_Oil_Price_USD_bbl',
    'Цена на медь, $/mt': 'Copper_Price_USD_mt',
    'Индикатор  экономической  активности, YoY': 'Economic_Activity_Indicator_YoY',
    'Номинальный ВВП, млн драмов': 'Nominal_GDP_Mln_AMD',
    'Первичные доходы –  Оплата труда,  млн долларов': 'Primary_Income_Labor_Mln_USD',
    'Вторичные доходы –  Финансовые,  нефинансовые организации,  домохозяйства и НКО, млн долларов': 'Secondary_Income_Transfers_Mln_USD',
    'Чистый некоммерческий  приток, номинальный  в долларах,  YoY': 'Net_Non_Commercial_Inflow_Nominal_USD_YoY',
    'Первичные доходы, млн драмов': 'Primary_Income_Mln_AMD',
    'Вторичные доходы, млн драмов': 'Secondary_Income_Mln_AMD',
    'Располагаемый  доход, млн драмов': 'Disposable_Income_Mln_AMD',
    'Располагаемый доход, YoY': 'Disposable_Income_YoY',
    'Реальный располагаемый доход, YoY': 'Real_Disposable_Income_YoY_2'
}

# Dictionary for Monthly Translations (Inferring from column outputs from earlier)
monthly_translations_map = {
    'Unnamed: 0': 'Month',
    'ИПЦ, YoY': 'CPI_YoY',
    'Курс армянский  драм/доллар США': 'Exchange_Rate_AMD_USD',
    'Курс армянский  драм/российский рубль': 'Exchange_Rate_AMD_RUB',
    'Цена на нефть марки Brent, $/bbl': 'Brent_Oil_Price_USD_bbl',
    'Цена на медь, $/mt': 'Copper_Price_USD_mt',
    'Краткосрочные  номинальные  процентные ставки  по кредитам  в армянских драмах': 'Short_Term_Nominal_Interest_Rate_Loans_AMD',
    'Краткосрочные  номинальные  процентные ставки  по кредитам  в долларах США': 'Short_Term_Nominal_Interest_Rate_Loans_USD',
    'Долгосрочные  номинальные  процентные ставки  по кредитам  в армянских драмах': 'Long_Term_Nominal_Interest_Rate_Loans_AMD',
    'Долгосрочные  номинальные  процентные ставки  по кредитам  в долларах США': 'Long_Term_Nominal_Interest_Rate_Loans_USD',
    'Краткосрочные  номинальные  процентные ставки  по депозитам  в армянских драмах': 'Short_Term_Nominal_Interest_Rate_Deposits_AMD',
    'Краткосрочные  номинальные  процентные ставки  по депозитам  в долларах США': 'Short_Term_Nominal_Interest_Rate_Deposits_USD',
    'Долгосрочные  номинальные  процентные ставки  по депозитам  в армянских драмах': 'Long_Term_Nominal_Interest_Rate_Deposits_AMD',
    'Долгосрочные  номинальные  процентные ставки  по депозитам  в долларах США': 'Long_Term_Nominal_Interest_Rate_Deposits_USD',
    'Наличные  в обращении, млн. драмов': 'Cash_in_Circulation_Mln_AMD',
    'Денежная  масса M2, млн. драмов': 'Money_Supply_M2_Mln_AMD',
    'Денежная  масса M2X, млн. драмов': 'Money_Supply_M2X_Mln_AMD',
    'Кредиты  коммерческих  банков, млн. драмов': 'Commercial_Bank_Loans_Mln_AMD',
    'Кредиты  предприятиям, млн. драмов': 'Enterprise_Loans_Mln_AMD',
    'Кредиты частным  предприятиям,  млн. драмов': 'Private_Enterprise_Loans_Mln_AMD',
    'Кредиты  домохозяйствам, млн. драмов': 'Household_Loans_Mln_AMD',
    'Кредиты  в промышленность, млн. драмов': 'Loans_Industry_Mln_AMD',
    'Кредиты  в сельское хозяйство, млн. драмов': 'Loans_Agriculture_Mln_AMD',
    'Кредиты  в строительство, млн. драмов': 'Loans_Construction_Mln_AMD',
    'Кредиты  в сферу транспорта  и связи, млн. драмов': 'Loans_Transport_Communication_Mln_AMD',
    'Кредиты  в торговлю, млн. драмов': 'Loans_Trade_Mln_AMD',
    'Кредиты  в сферу услуг, млн. драмов': 'Loans_Services_Mln_AMD',
    'Потребительские  кредиты, млн. драмов': 'Consumer_Loans_Mln_AMD',
    'Ипотечные  кредиты, млн. драмов': 'Mortgage_Loans_Mln_AMD',
    'Прочие  кредиты, млн. драмов': 'Other_Loans_Mln_AMD',
    'Общий объем  кредитов, млн. драмов': 'Total_Loans_Mln_AMD',
    'Кредиты резидентам,  банки, млн.  драмов': 'Loans_Residents_Banks_Mln_AMD',
    'Кредиты  резидентам  (кредитные  организации), млн. драмов': 'Loans_Residents_Credit_Orgs_Mln_AMD',
    'Наемные  работники': 'Hired_Workers',
    'Индекс  экономической  активности,  дискретные,  YoY': 'Economic_Activity_Index_Discrete_YoY',
    'Промышленность  (выпуск, реальный рост), YoY': 'Industry_Real_Growth_YoY',
    'Строительство (выпуск, реальный рост), YoY': 'Construction_Real_Growth_YoY',
    'Сфера услуг (выпуск, реальный рост), YoY': 'Services_Real_Growth_YoY',
    'Частное  строительство (выпуск, реальный рост), YoY': 'Private_Construction_Real_Growth_YoY',
    'Промышленность, выпуск, млн. драмов': 'Industry_Output_Mln_AMD',
    'Строительство, выпуск, млн. драмов': 'Construction_Output_Mln_AMD',
    'Сфера услуг, выпуск,  млн. драмов': 'Services_Output_Mln_AMD'
}

# Translate columns
df_quarterly.rename(columns=quarterly_translations, inplace=True)
df_monthly.rename(columns=monthly_translations_map, inplace=True)

# Function to parse quarterly dates
def parse_quarter(q_str):
    try:
        year = int(q_str[:4])
        q = int(q_str[-1])
        month = (q - 1) * 3 + 1
        return pd.Timestamp(year=year, month=month, day=1)
    except:
        return np.nan

# Function to parse monthly dates
def parse_month(m_str):
    try:
        return pd.to_datetime(m_str.replace('M', '-'), format='%Y-%m')
    except:
        return np.nan

# Apply date parsing
df_quarterly['Date'] = df_quarterly['Quarter'].apply(parse_quarter)
df_quarterly.set_index('Date', inplace=True)
df_quarterly.drop(columns=['Quarter'], inplace=True, errors='ignore')

df_monthly['Date'] = df_monthly['Month'].apply(parse_month)
df_monthly.set_index('Date', inplace=True)
df_monthly.drop(columns=['Month'], inplace=True, errors='ignore')

# Save translated data to a new Excel file to ensure perfectly clean base layer
translated_file_path = r'D:\DATA\Translated_Cleaned_Nowcasting_Data.xlsx'
with pd.ExcelWriter(translated_file_path) as writer:
    df_quarterly.to_excel(writer, sheet_name='Quarterly')
    df_monthly.to_excel(writer, sheet_name='Monthly')

print(f"Successfully translated and saved to {translated_file_path}")
print(f"Quarterly columns: {len(df_quarterly.columns)}")
print(f"Monthly columns: {len(df_monthly.columns)}")
