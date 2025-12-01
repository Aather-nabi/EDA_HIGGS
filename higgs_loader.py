import pandas as pd
import numpy as np
import os

def load_data(file_path='HIGGS.csv.gz', nrows=None):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Column names based on UCI Machine Learning Repository description
    # 1st column is label
    # 21 low-level features
    # 7 high-level features
    
    low_level = [
        'lepton_pT', 'lepton_eta', 'lepton_phi', 
        'missing_energy_magnitude', 'missing_energy_phi',
        'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_b-tag',
        'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_b-tag',
        'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_b-tag',
        'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_b-tag'
    ]
    
    high_level = [
        'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]
    
    columns = ['label'] + low_level + high_level 
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, names=columns,  nrows=nrows)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    print(df.head(10))
    #print(df.shape)
    return df

if __name__ == "__main__":
    # Test loading a small chunk
    try:
        df = load_data(nrows=1000)
        print("\nFirst 5 rows:")
        print(df.head(5))
        print("\nData types:")
        print(df.dtypes)
    except Exception as e:
        print(f"Error loading data: {e}")
