import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

original_dataframe = pd.read_csv('dataset.csv')
df = original_dataframe.copy()

def preprocess(df):
    # Drop the 'customerID' column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)  
    # Strip whitespace from object columns
    
    df.replace('', np.nan, inplace=True)
    # Replace empty strings with NaN
    
    df = df.dropna().reset_index(drop=True)
    # Drop rows with NaN values
     
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(float)
    df['tenure'] = df['tenure'].astype(float)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    # Convert 'TotalCharges' column to float
        
    return df

new_df = preprocess(df)

X = new_df.iloc[:, :-1]
y = new_df.iloc[:, -1]

def X_encoder(X):    
    
    numerical_columns = X.select_dtypes(include='float').columns
    categorical_columns = X.select_dtypes(include='object').columns
    
    X_num = X[numerical_columns].values
    
    ohe = OneHotEncoder(sparse_output = False, drop= 'first', handle_unknown='ignore')
    X_cat = ohe.fit_transform(X[categorical_columns])
    
    X = np.hstack((X_num,X_cat))
    
    return X

X = X_encoder(X)

def y_encoder(y):

    lab = LabelEncoder()

    y = lab.fit_transform(y)

    return y

y = y_encoder(y)



def smoter(X, y):
    sm = SMOTE(random_state=102)
    
    X, y = sm.fit_resample(X, y.ravel())
    
    return X, y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=105)

from sklearn.pipeline import Pipeline

# Create the pipeline
pipeline = Pipeline([
    ('Standardization', StandardScaler()),
    ('Random_Forest', RandomForestClassifier())
     ])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their possible values
param_grid = {
    'Random_Forest__n_estimators': [100, 200, 300],
    'Random_Forest__max_depth': [None, 5, 10]
}

# Create the GridSearchCV object with the pipeline and parameter grid
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best hyperparameters and score
print("Best Hyperparameters:", best_params)
print("Best Score:", best_score)

# Make predictions on the test data using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# print(classification_report(y_test, y_pred))

import pickle

filename = 'model.pkl'

pickle.dump(best_model, open(filename, 'wb'))
