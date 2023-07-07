import pandas as pd
import numpy as np 

def preprocess(df):
    
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)  
    # Strip whitespace from object columns
    
    df.replace('', np.nan, inplace=True)  
    # Replace empty strings with NaN
    
    df.dropna(inplace=True)  
    # Drop rows with NaN values
    
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    else:
        pass
    
    # Drop the 'customerID' column
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(float)
    df['tenure'] = df['tenure'].astype(float)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    # Convert 'TotalCharges' column to float
    
    new_df = pd.get_dummies(df, drop_first=True)
    
    return new_df