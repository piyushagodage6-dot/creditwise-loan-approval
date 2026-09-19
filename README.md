# CreditWise – Loan Approval Prediction System

CreditWise is a Machine Learning-based Loan Approval Prediction System that predicts whether a loan application is likely to be approved based on applicant and loan-related information.

The project includes data preprocessing, feature engineering, Machine Learning model comparison, and an interactive Streamlit web application.

## 🚀 Live Demo

[Click here to try CreditWise]([YOUR_STREAMLIT_LINK](https://creditwise-loan-approval-cgh8hbqxldfxdiwzkgsmah.streamlit.app/))

## 📌 Features

- Loan approval prediction using Machine Learning
- Applicant and loan information input through an interactive UI
- Data preprocessing and missing-value handling
- Categorical feature encoding
- Feature scaling
- Feature engineering
- Model evaluation using multiple performance metrics
- Interactive Streamlit web application
- Online deployment

## 🤖 Machine Learning Models

The following classification models were implemented and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Gaussian Naive Bayes was selected as the final model based on the evaluation performed in the project.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn

## 📊 Data Preprocessing

The project includes:

- Missing value handling
- Numerical feature preprocessing
- Categorical feature encoding using Label Encoding and One-Hot Encoding
- Feature scaling using StandardScaler
- Feature engineering using Credit Score and DTI Ratio
- Train-test split for model evaluation

## 📂 Project Structure

```text
creditwise-loan-approval/
│
├── app.py
├── model.pkl
├── ohe.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
