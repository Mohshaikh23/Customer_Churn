import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def X_encoder(X):    
    
    numerical_columns = X.select_dtypes(include='float').columns
    categorical_columns = X.select_dtypes(include='object').columns
    
    X_num = X[numerical_columns].values
    
    ohe = OneHotEncoder(sparse_output = False, 
                        drop= 'first',
                        handle_unknown='ignore')
    
    X_cat = ohe.fit_transform(X[categorical_columns])
    
    X = np.hstack((X_num,X_cat))
    
    return X