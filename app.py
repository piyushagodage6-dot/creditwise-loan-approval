import streamlit as st
import joblib
import pandas as pd
import numpy as np
model = joblib.load("model.pkl")

st.title("CreditWise - Loan Approval System")

st.write("Enter applicant details to predict loan approval.")
age = st.number_input("Age", min_value=18, max_value=100)
applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
employment_status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

dependents = st.number_input(
    "Dependents",
    min_value=0,
    max_value=10,
    step=1
)

credit_score = st.number_input(
    "Credit Score",
    min_value=0.0,
    max_value=900.0
)

existing_loans = st.number_input(
    "Existing Loans",
    min_value=0,
    step=1
)

dti_ratio = st.number_input(
    "DTI Ratio",
    min_value=0.0
)

savings = st.number_input(
    "Savings",
    min_value=0.0
)

collateral_value = st.number_input(
    "Collateral Value",
    min_value=0.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0
)

loan_term = st.number_input(
    "Loan Term",
    min_value=1,
    step=1
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Personal", "Car", "Business"]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

education_level = st.selectbox(
    "Education Level",
    ["Graduate", "Not Graduate"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

employer_category = st.selectbox(
    "Employer Category",
    ["Private", "Government", "MNC", "Unemployed"]
)
if st.button("Predict Loan Approval"):

    # Create input dataframe
    input_data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": 0 if education_level == "Graduate" else 1,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category
    }

    input_df = pd.DataFrame([input_data])

    # Load preprocessing objects
    ohe = joblib.load("ohe.pkl")
    scaler = joblib.load("scaler.pkl")

    # Categorical columns
    cat_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    # One-hot encoding
    encoded = ohe.transform(input_df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cat_cols)
    )

    # Remove categorical columns
    input_df = input_df.drop(columns=cat_cols)

    # Combine
    input_df = pd.concat(
        [input_df.reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Feature engineering
    input_df["credit_score_sq"] = input_df["Credit_Score"] ** 2
    input_df["dti_ratio_sq"] = input_df["DTI_Ratio"] ** 2
    input_df["Applicant_income_log"] = np.log1p(input_df["Applicant_Income"])
    # Drop original features
    input_df = input_df.drop(
        columns=["Credit_Score", "DTI_Ratio"]
    )

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")