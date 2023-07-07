from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create the pipeline
pipeline = Pipeline([
    ('Standardization', StandardScaler()),
    ('Random_Forest', RandomForestClassifier())
     ])