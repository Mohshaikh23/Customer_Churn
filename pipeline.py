from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Create the pipeline
def pipeline(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=105)
    
    pipeline = Pipeline([
                ('Standardization', StandardScaler()),
                ('Random_Forest', RandomForestClassifier())
                        ])
    return pipeline

# # Fit the pipeline to the training data
# pipeline.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = pipeline.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
