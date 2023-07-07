import streamlit as st
import pandas as pd
from model_predictor import model_prediction

st.title('Churn Check')

st.sidebar.title('csv file over here')
uploaded_file = st.sidebar.file_uploader('Choose a file')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

st.header("Welcome to the Chun Check App")
st.text('''You can insert the value of all the parameters you have to
        check if the customer is churn out or not''')

col =['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
      'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
      'InternetService','OnlineSecurity', 'OnlineBackup',
      'DeviceProtection', 'TechSupport','StreamingTV',
      'StreamingMovies', 'Contract', 'PaperlessBilling',
      'PaymentMethod', 'tenure']
data = []

for i in col:
    user_input = st.text_input(f"{i}", "")
    data.append(user_input)

df = pd.DataFrame(data = [data], columns=col)

if st.button('click for dataframe'):
    st.dataframe(data =df)

if st.button('Process'):
    probability, result = model_prediction(df)

    if result == 1:
        st.write("This customer is likely to be churned!!")
        st.write("Confidence: {}".format(probability*100))
    else:
        st.write("This customer is likely to continue!!")
        st.write("Confidence: {}".format(probability*100))