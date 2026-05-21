import pandas as pd

# Check p6_2 structure
df = pd.read_excel('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/data_xlsx/p6_2.xlsx')
with open('d:/DATA/GDP_NOWCASTING_DASHBOARD/excel_debug.txt', 'w', encoding='utf-8') as f:
    f.write(f'=== p6_2.xlsx === shape={df.shape}\n')
    f.write(f'Columns: {list(df.columns)}\n')
    f.write(df.to_string() + '\n\n')

    # Check p4_1 structure for the 2024 line
    df2 = pd.read_excel('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/data_xlsx/p4_1.xlsx')
    f.write(f'=== p4_1.xlsx === shape={df2.shape}\n')
    f.write(f'Columns: {list(df2.columns)}\n')
    f.write(df2.to_string() + '\n\n')

    # Check p4_3 structure
    df3 = pd.read_excel('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/data_xlsx/p4_3.xlsx')
    f.write(f'=== p4_3.xlsx === shape={df3.shape}\n')
    f.write(f'Columns: {list(df3.columns)}\n')
    f.write(df3.to_string() + '\n\n')

    # Check p7_2 structure  
    df4 = pd.read_excel('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/data_xlsx/p7_2.xlsx')
    f.write(f'=== p7_2.xlsx === shape={df4.shape}\n')
    f.write(f'Columns: {list(df4.columns)}\n')
    f.write(df4.to_string() + '\n\n')

    # Check p7_3 structure
    df5 = pd.read_excel('d:/DATA/GDP_NOWCASTING_DASHBOARD/data/data_xlsx/p7_3.xlsx')
    f.write(f'=== p7_3.xlsx === shape={df5.shape}\n')
    f.write(f'Columns: {list(df5.columns)}\n')
    f.write(df5.to_string() + '\n\n')

print("Done")
