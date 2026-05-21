import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import os

os.makedirs(f'{BASE_DIR}/data', exist_ok=True)

# --- Page 1 ---
df1_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p1_1.xlsx')
# 2023-01 starts at index 60, 2026-01 is at index 96 (37 rows)
df1_1 = pd.DataFrame({
    'cu': df1_1_raw.iloc[60:97, 3].values,
    'oil': df1_1_raw.iloc[60:97, 2].values,
    'gold': df1_1_raw.iloc[60:97, 7].values
})
df1_1.to_csv(f'{BASE_DIR}/data/p1_commodities.csv', index=False, float_format="%.2f")

df1_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p1_3.xlsx')
# 2024-01 is at index 60. Odd rows from 60 to 86 are months 1,3,5,7,9,11 in 2024 and 2025, and 84 is 2026-01: [60,62,64,66,68,70, 72,74,76,78,80,82, 84]
idx_p1 = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84]
df1_2 = pd.DataFrame({
    'month': ["1", "3", "5", "7", "9", "11", "1", "3", "5", "7", "9", "11", "1"],
    'meat': df1_3_raw.iloc[idx_p1, 3].values,
    'dairy': df1_3_raw.iloc[idx_p1, 4].values,
    'cereals': df1_3_raw.iloc[idx_p1, 5].values,
    'oil': df1_3_raw.iloc[idx_p1, 6].values,
    'sugar': df1_3_raw.iloc[idx_p1, 7].values,
    'food': df1_3_raw.iloc[idx_p1, 2].values
})
df1_2.to_csv(f'{BASE_DIR}/data/p1_food.csv', index=False, float_format="%.2f")

# --- Page 2 ---
df2_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p2_1.xlsx')
df2_1 = pd.DataFrame({
    "Ուղղություն": ["Ընդհանուր","ՌԴ","Այլ երկր."]*4,
    "Տարեթիվ": ["2019"]*3+["2023"]*3+["2024"]*3+["2025"]*3,
    "Քանակ": [
        df2_1_raw.iloc[0,2], df2_1_raw.iloc[1,2], df2_1_raw.iloc[4,2],
        df2_1_raw.iloc[0,6], df2_1_raw.iloc[1,6], df2_1_raw.iloc[4,6],
        df2_1_raw.iloc[0,7], df2_1_raw.iloc[1,7], df2_1_raw.iloc[4,7],
        df2_1_raw.iloc[0,8], df2_1_raw.iloc[1,8], df2_1_raw.iloc[4,8]
    ]
})
df2_1.to_csv(f'{BASE_DIR}/data/p2_tourism_counts.csv', index=False, float_format="%.2f")

# Tourism growth isn't directly in p2_1 as a table, using its original.
df2_2 = pd.DataFrame({"Երկիր": ["Վրաստան", "Չինաստան", "Ֆրանսիա", "Իրան", "Հնդկաստան", "ՌԴ"], "Աճ": [2.5, 1.8, 1.2, 0.4, -0.7, -0.03]})
df2_2.to_csv(f'{BASE_DIR}/data/p2_tourism_growth.csv', index=False)

# --- Page 3 ---
df3_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p3_1.xlsx')
df3_2_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p3_2.xlsx')
df3_1 = pd.DataFrame({
    'month': list(range(1, 13)),
    'in_2021': df3_1_raw.iloc[0:12, 1].values,
    'in_2022': df3_1_raw.iloc[0:12, 2].values,
    'in_2023': df3_1_raw.iloc[0:12, 3].values,
    'in_2024': df3_1_raw.iloc[0:12, 4].values,
    'in_2025': df3_1_raw.iloc[0:12, 5].values,
    'out_2021': df3_1_raw.iloc[14:26, 1].values,
    'out_2022': df3_1_raw.iloc[14:26, 2].values,
    'out_2023': df3_1_raw.iloc[14:26, 3].values,
    'out_2024': df3_1_raw.iloc[14:26, 4].values,
    'out_2025': df3_1_raw.iloc[14:26, 5].values,
    'net_2021': df3_2_raw.iloc[0:12, 1].values,
    'net_2022': df3_2_raw.iloc[0:12, 2].values,
    'net_2023': df3_2_raw.iloc[0:12, 3].values,
    'net_2024': df3_2_raw.iloc[0:12, 4].values,
    'net_2025': df3_2_raw.iloc[0:12, 5].values
})
# Excel data is already cumulative — no cumsum needed
df3_1.to_csv(f'{BASE_DIR}/data/p3_remittances.csv', index=False, float_format="%.2f")

# --- Page 4 ---
df4_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p4_1.xlsx')
df4_1 = pd.DataFrame({
    "quarter": ["I", "II", "III", "IV"],
    "eai": list(df4_1_raw.iloc[0, 1:5].values),
    "eai_nosk": list(df4_1_raw.iloc[1, 1:5].values)
})
df4_1.to_csv(f'{BASE_DIR}/data/p4_eai_quarterly.csv', index=False, float_format="%.2f")

# p4_eai_monthly doesn't have an obvious excel mapped structure in my dump (maybe missing?). Will keep as-is.
df4_2 = pd.DataFrame({
    "month": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"],
    "val_2024": [10.7, 13.6, 15.3, 10.4, 5.2, 7.1, 6.2, 5.6, 7.0, 4.2, 4.3, 4.0],
    "val_2025": [13.2, 10.5, 9.4, 7.5, 5.5, 3.8, 12.3, 13.9, 13.8, 12.0, 11.2, 9.0]
})
df4_2.to_csv(f'{BASE_DIR}/data/p4_eai_monthly.csv', index=False)

df4_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p4_3.xlsx')
# Row 8 is ՏԱՑ summary: contribution=8.9, growth=9.17
df4_3 = pd.DataFrame({
    "sector": ["ՏԱՑ", "ծառայություններ", "ֆին․և ապահով․ գործ․", "տեղեկ․ և կապ", "շինարարություն", "արդյունաբերություն", "մշակող արդյունաբեր․", "գյուղատնտեսություն*", "առևտուր"],
    "growth": [df4_3_raw.iloc[8,2], df4_3_raw.iloc[7,2], None, None, df4_3_raw.iloc[4,2], df4_3_raw.iloc[3,2], None, df4_3_raw.iloc[1,2], df4_3_raw.iloc[0,2]],
    "contribution": [df4_3_raw.iloc[8,1], df4_3_raw.iloc[7,1], df4_3_raw.iloc[6,1], df4_3_raw.iloc[5,1], df4_3_raw.iloc[4,1], df4_3_raw.iloc[3,1], df4_3_raw.iloc[2,1], df4_3_raw.iloc[1,1], df4_3_raw.iloc[0,1]]
})
df4_3.to_csv(f'{BASE_DIR}/data/p4_sectors.csv', index=False, float_format="%.2f")

# --- Page 5 ---
df5_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p5_1.xlsx')
df5_1 = pd.DataFrame({
    "month": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"],
    "val_2025": list(df5_1_raw.iloc[1, 1:13].values),
    "val_2024": list(df5_1_raw.iloc[0, 1:13].values),
    "water": list(df5_1_raw.iloc[5, 1:13].values),
    "energy": list(df5_1_raw.iloc[4, 1:13].values),
    "manuf": list(df5_1_raw.iloc[3, 1:13].values),
    "mining": list(df5_1_raw.iloc[2, 1:13].values)
})
df5_1.to_csv(f'{BASE_DIR}/data/p5_industry.csv', index=False, float_format="%.2f")

df5_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p5_3.xlsx')
df5_2 = pd.DataFrame({
    "sector": ["Մշակող արդյունաբեր․", "ծխախոտ", "սննդամթերքի արտադրություն", "խմիչքների արտադրություն", "ոսկերչական արտադրատեսակներ", "հիմնային մետաղներ"], 
    "val": [df5_3_raw.iloc[5,1], df5_3_raw.iloc[4,1], df5_3_raw.iloc[3,1], df5_3_raw.iloc[2,1], df5_3_raw.iloc[1,1], df5_3_raw.iloc[0,1]]
})
df5_2.to_csv(f'{BASE_DIR}/data/p5_manufacturing.csv', index=False, float_format="%.2f")

# --- Page 6 ---
df6_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p6_1.xlsx')
df6_1 = pd.DataFrame({
    "period": ["I-III", "I-VI", "I-IX", "I-XII"],
    "l25": list(df6_1_raw.iloc[0, 1:5].values),
    "l24": list(df6_1_raw.iloc[1, 1:5].values),
    "crop": list(df6_1_raw.iloc[2, 1:5].values),
    "animal": list(df6_1_raw.iloc[3, 1:5].values),
    "forest": list(df6_1_raw.iloc[4, 1:5].values),
    "fish": list(df6_1_raw.iloc[5, 1:5].values)
})
df6_1.to_csv(f'{BASE_DIR}/data/p6_agriculture.csv', index=False, float_format="%.2f")

df6_2_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p6_2.xlsx')
df6_2 = pd.DataFrame({
    "sector": ["Բուսաբուծություն", "Անասնաբուծություն", "Ձկնորսություն", "Ամբողջ գյուղ."], 
    "growth": [df6_2_raw.iloc[2,1], df6_2_raw.iloc[1,1], df6_2_raw.iloc[0,1], df6_2_raw.iloc[3,1]], 
    "contrib": [df6_2_raw.iloc[2,1], df6_2_raw.iloc[1,1], df6_2_raw.iloc[0,1], df6_2_raw.iloc[3,1]]
})
df6_2.to_csv(f'{BASE_DIR}/data/p6_sectors.csv', index=False, float_format="%.2f")

# --- Page 7 ---
df7_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p7_1.xlsx')
df7_1 = pd.DataFrame({
    "month": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"],
    "gr24": list(df7_1_raw.iloc[1, 1:13].values),
    "gr25": list(df7_1_raw.iloc[0, 1:13].values),
    "state": list(df7_1_raw.iloc[2, 1:13].values),
    "comm": list(df7_1_raw.iloc[3, 1:13].values),
    "hum": list(df7_1_raw.iloc[4, 1:13].values),
    "org": list(df7_1_raw.iloc[5, 1:13].values),
    "pop": list(df7_1_raw.iloc[6, 1:13].values)
})
df7_1.to_csv(f'{BASE_DIR}/data/p7_construction_monthly.csv', index=False, float_format="%.2f")

df7_2_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p7_2.xlsx')
df7_2 = pd.DataFrame({
    "source": ["Համայնք միջոցներ", "Բնակչ․ միջոցներ", "Կազմ․ միջոցներ", "Պետական բյուջե", "Շինարարություն"], 
    "val": [df7_2_raw.iloc[0,1], df7_2_raw.iloc[1,1], df7_2_raw.iloc[2,1], df7_2_raw.iloc[3,1], df7_2_raw.iloc[4,1]]
})
df7_2.to_csv(f'{BASE_DIR}/data/p7_funding.csv', index=False, float_format="%.2f")

df7_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p7_3.xlsx')
df7_3 = pd.DataFrame({
    "sector": ["էլեկտր․, գազի...", "Կրթություն", "Անշարժ գույք", "Շինարարություն"], 
    "val": [df7_3_raw.iloc[0,1], df7_3_raw.iloc[1,1], df7_3_raw.iloc[2,1], df7_3_raw.iloc[3,1]]
})
df7_3.to_csv(f'{BASE_DIR}/data/p7_sectors.csv', index=False, float_format="%.2f")

# --- Page 8 ---
q_lbl = []
for y in ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
    for q in ["I", "II", "III", "IV"]:
        if y == "2025" and q == "IV": continue
        q_lbl.append(f"{y}-{q}")

df8_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p8_1.xlsx')
df8_2_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p8_2.xlsx')
df8_1 = pd.DataFrame({
    "quarter": q_lbl,
    "trend_blue": list(df8_1_raw.iloc[0, 1:32].values),
    "trend_red": list(df8_1_raw.iloc[1, 1:32].values),
    "t_blue": list(df8_2_raw.iloc[0, 1:32].values),
    "t_red": list(df8_2_raw.iloc[1, 1:32].values)
})
df8_1.to_csv(f'{BASE_DIR}/data/p8_real_estate.csv', index=False, float_format="%.2f")

df8_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p8_3.xlsx')
df8_2 = pd.DataFrame({
    "quarter": ["I եռ.", "II եռ.", "III եռ.", "IV եռ."], 
    "p2022": list(df8_3_raw.iloc[0:4, 1].values), 
    "p2023": list(df8_3_raw.iloc[0:4, 2].values), 
    "p2024": list(df8_3_raw.iloc[0:4, 3].values), 
    "p2025": list(df8_3_raw.iloc[0:4, 4].values)
})
df8_2.to_csv(f'{BASE_DIR}/data/p8_permits.csv', index=False, float_format="%.2f")

# --- Page 9 ---
df9_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p9_1.xlsx')
df9_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p9_3.xlsx')
df9_1 = pd.DataFrame({
    "period": ["I եռ.", "II եռ.", "III եռ.", "IV եռ.", " ", "Տարեկան"],
    "emp25": list(df9_1_raw.iloc[2, 1:5].values) + [None, df9_1_raw.iloc[2, 6]],
    "emp24": list(df9_1_raw.iloc[1, 1:5].values) + [None, df9_1_raw.iloc[1, 6]],
    "emp23": list(df9_1_raw.iloc[0, 1:5].values) + [None, df9_1_raw.iloc[0, 6]],
    "lvl25": list(df9_3_raw.iloc[2, 1:5].values) + [None, df9_3_raw.iloc[2, 6]],
    "lvl24": list(df9_3_raw.iloc[1, 1:5].values) + [None, df9_3_raw.iloc[1, 6]],
    "lvl23": list(df9_3_raw.iloc[0, 1:5].values) + [None, df9_3_raw.iloc[0, 6]]
})
df9_1.to_csv(f'{BASE_DIR}/data/p9_employment.csv', index=False, float_format="%.2f")

df9_2_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p9_2.xlsx')
df9_2 = pd.DataFrame({
    "type": ["2024", "2025"], 
    "no_wage": list(df9_2_raw.iloc[0:2, 1].values), 
    "wage": list(df9_2_raw.iloc[0:2, 2].values)
})
df9_2.to_csv(f'{BASE_DIR}/data/p9_employment_structure.csv', index=False, float_format="%.2f")

# --- Page 10 ---
df10_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p10_1.xlsx')
df10_2_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p10_2.xlsx')
df10_1 = pd.DataFrame({
    "period": ["I եռ.", "II եռ.", "III եռ.", "IV եռ.", " ", "Տարեկան"],
    "unemp25": list(df10_1_raw.iloc[2, 1:5].values) + [None, df10_1_raw.iloc[2, 6]],
    "unemp24": list(df10_1_raw.iloc[1, 1:5].values) + [None, df10_1_raw.iloc[1, 6]],
    "unemp23": list(df10_1_raw.iloc[0, 1:5].values) + [None, df10_1_raw.iloc[0, 6]],
    "lvl25": list(df10_2_raw.iloc[2, 1:5].values) + [None, df10_2_raw.iloc[2, 6]],
    "lvl24": list(df10_2_raw.iloc[1, 1:5].values) + [None, df10_2_raw.iloc[1, 6]],
    "lvl23": list(df10_2_raw.iloc[0, 1:5].values) + [None, df10_2_raw.iloc[0, 6]]
})
df10_1.to_csv(f'{BASE_DIR}/data/p10_unemployment.csv', index=False, float_format="%.2f")

df10_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p10_3.xlsx')
df10_2 = pd.DataFrame({
    "category": ["Գործազուրկներ", "Աշխատուժի առաջարկ", "Զբաղվածներ", "Աշխատուժից դուրս"], 
    "val": [df10_3_raw.iloc[3,1], df10_3_raw.iloc[2,1], df10_3_raw.iloc[1,1], df10_3_raw.iloc[0,1]]
})
df10_2.to_csv(f'{BASE_DIR}/data/p10_changes.csv', index=False, float_format="%.2f")

df10_4_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p10_4.xlsx')
df10_3 = pd.DataFrame({
    "month": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"],
    "r25": list(df10_4_raw.iloc[1, 1:13].values),
    "r24": list(df10_4_raw.iloc[0, 1:13].values)
})
df10_3.to_csv(f'{BASE_DIR}/data/p10_registered.csv', index=False, float_format="%.2f")

# --- Page 11 ---
df11_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p11_1.xlsx')
df11_2_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p11_2.xlsx')
df11_3_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p11_3.xlsx')
df11_1 = pd.DataFrame({
    "period": ["I եռ.", "II եռ.", "III եռ.", "IV եռ.", " ", "Տարեկան"],
    "res25": list(df11_1_raw.iloc[2, 1:5].values) + [None, df11_1_raw.iloc[2, 6]],
    "res24": list(df11_1_raw.iloc[1, 1:5].values) + [None, df11_1_raw.iloc[1, 6]],
    "res23": list(df11_1_raw.iloc[0, 1:5].values) + [None, df11_1_raw.iloc[0, 6]],
    "sup25": list(df11_2_raw.iloc[2, 1:5].values) + [None, df11_2_raw.iloc[2, 6]],
    "sup24": list(df11_2_raw.iloc[1, 1:5].values) + [None, df11_2_raw.iloc[1, 6]],
    "sup23": list(df11_2_raw.iloc[0, 1:5].values) + [None, df11_2_raw.iloc[0, 6]],
    "out25": list(df11_3_raw.iloc[2, 1:5].values) + [None, df11_3_raw.iloc[2, 6]],
    "out24": list(df11_3_raw.iloc[1, 1:5].values) + [None, df11_3_raw.iloc[1, 6]],
    "out23": list(df11_3_raw.iloc[0, 1:5].values) + [None, df11_3_raw.iloc[0, 6]]
})
df11_1.to_csv(f'{BASE_DIR}/data/p11_labor_resources.csv', index=False, float_format="%.2f")

# --- Page 12 ---
df12_1_raw = pd.read_excel(f'{BASE_DIR}/data/data_xlsx/p12_1.xlsx')
df12_1 = pd.DataFrame({
    "sector": ["Ընդամենը", "Գյուղատնտեսություն", "Հանքագործություն", "Մշակող արդյունաբեր.", "Էլեկտրականություն", "Ջրամատակարարում", "Շինարարություն", "Առևտուր", "Փոխադրումներ", "Կացություն", "Տեղեկատվություն", "Ֆինանսական", "Անշարժ գույք", "Մասնագիտական", "Վարչարարական", "Պետական կառավարում", "Կրթություն", "Առողջապահություն", "Մշակույթ", "Սպասարկման այլ"],
    "wg": list(df12_1_raw.iloc[0:20, 2].values),
    "em": list(df12_1_raw.iloc[0:20, 1].values)
})
df12_1.to_csv(f'{BASE_DIR}/data/p12_wages.csv', index=False, float_format="%.2f")

print("CSVs generated from raw Excel export!")
