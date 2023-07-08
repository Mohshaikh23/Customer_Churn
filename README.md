# Customer Churn Prediction

📋 This project focuses on predicting customer churn using a RandomForestClassifier.

## Overview

The goal of this project is to build a predictive model that can accurately predict customer churn. Customer churn refers to the phenomenon where customers stop using a product or service. By identifying customers who are likely to churn, businesses can take proactive measures to retain them and reduce churn rate.

## Dataset

📊 The dataset used for this project is `dataset.csv`. It contains various customer attributes such as gender, tenure, monthly charges, and churn status.

## Workflow

🔧 The project workflow can be summarized as follows:

1. **Data Preprocessing:** The dataset is preprocessed to handle missing values, convert data types, and encode categorical variables.

2. **Feature Engineering:** The features are transformed and prepared for model training using OneHotEncoder and LabelEncoder.

3. **Data Balancing:** The dataset is balanced using the SMOTE technique to address class imbalance.

4. **Model Training:** A RandomForestClassifier model is built using a pipeline that includes data standardization.

5. **Model Evaluation:** The model is evaluated using accuracy score and classification report on the test set.

6. **Hyperparameter Tuning:** GridSearchCV is used to find the best hyperparameters for the RandomForestClassifier model.

7. **Saving the Model:** The best model is saved as a pickle file (`model.pkl`) for future use.

## How to Run

🚀 To run the project:

1. Install the required libraries using `pip install -r requirements.txt`.

2. Run the script `main.py` to train the model and save it.

3. Use the saved model to make predictions on new data.

## Results

📊 The best model achieved an accuracy score of XX% on the test set. It shows promising performance in predicting customer churn.

## Future Improvements

🔍 Further improvements can be made to enhance the model's performance:

- Explore additional feature engineering techniques to capture more predictive information.
- Experiment with different machine learning algorithms to compare performance.
- Collect more data to improve the model's accuracy and generalization.

📚 Feel free to contribute to this project by suggesting improvements or adding new features!