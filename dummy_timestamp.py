import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def find_header_row(file_path, keyword, sheet_name):
        return_index = None
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        for index, row in df.iterrows():
            if row.astype(str).str.contains(keyword, case=False, na=False).any():
                return_index = index
                break
        return return_index

file_path = "HKFoods.xlsx"
keyword = "BATCH no."

skip_rows = find_header_row(file_path=file_path, keyword=keyword, sheet_name="FAITH STORAGE AFTER COOKING")
df = pd.read_excel(file_path, sheet_name="FAITH STORAGE AFTER COOKING", skiprows=skip_rows)

for index, row in df.iterrows():
    start_time = row[1] + timedelta(hours=np.random.randint(13,15), minutes=np.random.randint(0,60), seconds=np.random.randint(0,60))
    end_time = row[2] + timedelta(hours=np.random.randint(15,18), minutes=np.random.randint(0,60), seconds=np.random.randint(0,60))
    df.at[index, 'start_time'] = start_time
    df.at[index, 'end_time'] = end_time
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

print(df.head())

df.to_excel("faith_storage.xlsx", columns=["BATCH no.", "BATCH WEIGHT LEAVING STORAGE (KG)","start_time", "end_time"], index=False)