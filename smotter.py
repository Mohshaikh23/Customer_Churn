from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def smotter(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size=0.2,
                                                        random_state=105)
    
    sm = SMOTE(random_state=102)

    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_res,
                                                        y_train_res, 
                                                        test_size=0.2,
                                                        random_state=105)
    
    return X_train, X_test, y_train, y_test
