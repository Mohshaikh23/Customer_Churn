from sklearn.preprocessing import LabelEncoder

def y_encoder(y):
    
    lab = LabelEncoder()

    y = lab.fit_transform(y)
    
    return y