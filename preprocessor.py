import pandas as pd
import numpy as np 

def preprocess(df):
    # Drop the 'customerID' column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(float)
    df['tenure'] = df['tenure'].astype(float)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)  
    # Strip whitespace from object columns
    
    df.replace('', np.nan, inplace=True)
    # Replace empty strings with NaN
    
    df = df.dropna().reset_index(drop=True)
    # Drop rows with NaN values
        
    return df