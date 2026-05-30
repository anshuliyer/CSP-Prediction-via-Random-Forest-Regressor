import pandas as pd
import requests
import io
import warnings
warnings.filterwarnings('ignore')

print("Loading existing NHANES dataset...")
df_main = pd.read_csv('Nhanes_cvd_raw.csv')

def fetch_xpt(url, columns):
    print(f"Downloading {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}. Status code: {response.status_code}")
    
    # Read the XPT file
    df = pd.read_sas(io.BytesIO(response.content), format='xport')
    
    # Some columns might not exist if we mix and match incorrectly, handle safely
    existing_cols = [c for c in columns if c in df.columns]
    return df[existing_cols].copy()

# Define columns needed
demo_cols = ['SEQN', 'RIDRETH1']
bmx_cols = ['SEQN', 'BMXWT', 'BMXHT']

# 2017-March 2020 Data
demo_p = fetch_xpt('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.xpt', demo_cols)
bmx_p = fetch_xpt('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.xpt', bmx_cols)

# 2021-August 2023 Data
demo_l = fetch_xpt('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt', demo_cols)
bmx_l = fetch_xpt('https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt', bmx_cols)

# Combine cycles
demo_all = pd.concat([demo_p, demo_l])
bmx_all = pd.concat([bmx_p, bmx_l])

# Merge Demo and BMX
extra_data = pd.merge(demo_all, bmx_all, on='SEQN', how='inner')

# Map Ethnicity codes to strings
ethnicity_map = {
    1.0: 'MA',
    2.0: 'OH',
    3.0: 'NHW',
    4.0: 'NHB',
    5.0: 'OR'
}
extra_data['RIDRETH1'] = extra_data['RIDRETH1'].map(ethnicity_map)

# Rename columns
extra_data = extra_data.rename(columns={
    'RIDRETH1': 'Ethnicity',
    'BMXWT': 'Weight_kg',
    'BMXHT': 'Height_cm'
})

print("Merging with main dataset...")
# Merge with the main CSV
df_merged = pd.merge(df_main, extra_data, on='SEQN', how='left')

# Save to a new CSV
output_file = 'Nhanes_cvd_full.csv'
df_merged.to_csv(output_file, index=False, na_rep='NAN')
print(f"Successfully saved merged dataset to {output_file}!")
