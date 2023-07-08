from preprocessor import preprocess
import pickle
import pandas as pd
from x_encoder import X_encoder

model = pickle.load(open("model.pkl", "rb"))
original_dataframe = pd.read_csv('dataset.csv')

def model_prediction(df):
     
    df = pd.concat(
        [original_dataframe.iloc[:,1:len(original_dataframe.columns)-1],
        df], 
        ignore_index = True) 

    X = preprocess(df)
    
    X = X_encoder(X)

    X_test = X[-1].reshape(1, -1)
    
    prediction = model.predict(X_test)
    
    prob = model.predict_proba(X_test)
    
    probability_No,probability_yes = prob[0][0]*100 , prob[0][1]*100

    return prediction, probability_No, probability_yes