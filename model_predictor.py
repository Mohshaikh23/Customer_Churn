from preprocessor import preprocess
import pickle

def model_prediction(df):
    df = preprocess(df)
    model = pickle.load(open("model.pkl", "rb"))
    result = model.predict(df)
    probability = model.predict_proba(df)
    return probability, result