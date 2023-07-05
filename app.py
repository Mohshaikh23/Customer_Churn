import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('dataset.csv')
def categorical_column_fetch(df):
    categorical_columns = []
    for i in df.columns:
        if df[i].dtype == 'object':
            categorical_columns.append(i)
            continue
        else:
            pass
    return categorical_columns

categorical_columns = categorical_column_fetch(df)

df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)  
# Strip whitespace from object columns

df.replace('', np.nan, inplace=True)  
# Replace empty strings with NaN

df.dropna(inplace=True)  
# Drop rows with NaN values

df.drop('customerID', axis=1, inplace=True)  
# Drop the 'customerID' column

df['TotalCharges'] = df['TotalCharges'].astype(float)  
# Convert 'TotalCharges' column to float

# Sample equal number of 'No' instances as 'Yes' instances
yes_data = df[df['Churn'] == 'Yes']
no_data = df[df['Churn'] == 'No'].sample(n=len(yes_data), random_state=24)
df = pd.concat([yes_data, no_data], ignore_index=True)

df = df.sample(frac=1, random_state=24).reset_index(drop=True)  # Shuffle the dataframe

new_df = df.copy()

X= df.iloc[:, :-1]

y=df.iloc[:, -1]

le = LabelEncoder()
y = le.fit_transform(y)

cat_columns = [i for i in X.columns if df[i].dtype == 'object']
numerical_columns = ['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4)


target_transformer = Pipeline([('label_encoder',LabelEncoder())])

categorical_transformer = Pipeline([('one_hot_encoder',OneHotEncoder(drop='first',
                                                                     handle_unknown='ignore'))
                                    ])

# Create the ColumnTransformer to apply different transformers to different columns
preprocessor = ColumnTransformer([('categorical',categorical_transformer,cat_columns)],
                                                remainder='passthrough')

# Create the final pipeline
pipeline = Pipeline([('preprocessor', preprocessor),
                     ('Standard Scalar', StandardScaler()),
                     ('clf', LogisticRegression())
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)

# Print the classification report
# print(report)

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

