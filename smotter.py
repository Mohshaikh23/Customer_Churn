from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def smoter(X, y):
    sm = SMOTE(random_state=102)
    
    X, y = sm.fit_resample(X, y.ravel())
    
    return X, y