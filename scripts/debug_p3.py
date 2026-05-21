import pandas as pd

out = 'd:/DATA/GDP_NOWCASTING_DASHBOARD/debug_p3_output.txt'
with open(out, 'w', encoding='utf-8') as f:
    f.write("=== p3_1.xlsx ===\n")
    df1 = pd.read_excel('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/data_xlsx/p3_1.xlsx')
    f.write(f'Shape: {df1.shape}\n')
    f.write(f'Columns: {list(df1.columns)}\n')
    f.write(df1.to_string() + '\n\n')

    f.write("=== p3_2.xlsx ===\n")
    df2 = pd.read_excel('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/data_xlsx/p3_2.xlsx')
    f.write(f'Shape: {df2.shape}\n')
    f.write(f'Columns: {list(df2.columns)}\n')
    f.write(df2.to_string() + '\n\n')

    # Also show current CSV
    f.write("=== CURRENT p3_remittances.csv ===\n")
    df3 = pd.read_csv('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/p3_remittances.csv')
    f.write(f'Shape: {df3.shape}\n')
    f.write(f'Columns: {list(df3.columns)}\n')
    f.write(df3.to_string() + '\n\n')

print("Done")
