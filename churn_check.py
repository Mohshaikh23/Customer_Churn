import streamlit as st
import pandas as pd
from model_predictor import model_prediction

st.title('Churn Check')

st.sidebar.title('csv file over here')
#File upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.dataframe(df)


st.header("Welcome to the Chun Check App")
st.text('''You can insert the value of all the parameters you have to
        check if the customer is churn out or not''')

col =['gender', 'SeniorCitizen','Partner','Dependents',
      'tenure','PhoneService','MultipleLines','InternetService',
      'OnlineSecurity','OnlineBackup','DeviceProtection',
      'TechSupport','StreamingTV','StreamingMovies','Contract',
      'PaperlessBilling','PaymentMethod','MonthlyCharges',
      'TotalCharges']

data = []

for i in col:
    user_input = st.text_input(f"{i}", "")
    data.append(user_input)

df = pd.DataFrame(data = [data], columns=col)

if st.button('click for dataframe'):
    st.dataframe(data =df)
    st.write(df.info())

if st.button('Process'):
    prediction, probability_No, probability_yes = model_prediction(df)

    if prediction[0]==0:
        st.write('No Churn')
        st.write(f'{round(probability_No,2)}% Probability')
    else:
        st.write('yes Churn')
        st.write(f'{round(probability_yes,2)}% Probability')