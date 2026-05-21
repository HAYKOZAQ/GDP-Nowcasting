import pandas as pd
import os

data_dir = 'd:/DATA/GDP_NOWCASTING_DASHBOARD/data'
out_path = 'd:/DATA/GDP_NOWCASTING_DASHBOARD/audit_output.txt'

files = sorted(os.listdir(data_dir))
with open(out_path, 'w', encoding='utf-8') as f:
    for fn in files:
        path = os.path.join(data_dir, fn)
        df = pd.read_csv(path)
        f.write(f'=== {fn} === shape={df.shape}\n')
        f.write(df.to_string() + '\n\n')
print("Done writing to", out_path)
