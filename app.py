# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

st.set_page_config(page_title="Maintenance Risk Dashboard", layout="wide")
st.title("🛠️ Halliburton Maintenance Analytics")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("abnormality_report_cleaned.csv")
    df['SafetyIssueBool'] = df['SafetyIssue'].str.lower() == 'yes'
    df['MachineDownBool'] = df['MachineDownNew'].str.lower() == 'yes'
    df['Notes'] = df['Maint Tech Notes/Abnormality Action Item Notes'].fillna("")
    df['Date_Created'] = pd.to_datetime(df['Date_Created'])
    df['DateClosed'] = pd.to_datetime(df['DateClosed'])
    df['Hour'] = df['Date_Created'].dt.hour
    df['Date'] = df['Date_Created'].dt.date
    return df

df = load_data()

# Load model and vectorizer
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
selected_code = st.sidebar.multiselect("Select Maintenance Problem Code(s):", df['MaintenanceProblemCode'].unique(), default=None)
selected_date = st.sidebar.date_input("Select Date Range:", [])

if selected_code:
    df = df[df['MaintenanceProblemCode'].isin(selected_code)]
if len(selected_date) == 2:
    df = df[(df['Date_Created'] >= pd.to_datetime(selected_date[0])) & (df['Date_Created'] <= pd.to_datetime(selected_date[1]))]

# Calculate metrics
total_incidents = df.shape[0]
avg_hours_lost = df['MachineHoursLost'].mean()
avg_close_time = df['TotalHoursToClose'].mean()
safety_issue_rate = df['SafetyIssueBool'].mean()
downtime_rate = df['MachineDownBool'].mean()
safety_downtime_rate = df[df['SafetyIssueBool']]['MachineDownBool'].mean()
top_problem_codes = df['MaintenanceProblemCode'].value_counts().nlargest(10)
top_action_owners = df['Action Owner'].value_counts().nlargest(10)
hourly_distribution = df['Hour'].value_counts().sort_index()
daily_incidents = df.groupby('Date').size()

# Key metrics ###############################################

st.header("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Incidents", f"{total_incidents}")
col2.metric("Avg. Downtime (hrs)", f"{avg_hours_lost:.2f}")
col3.metric("Avg. Close Time (hrs)", f"{avg_close_time:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Safety Issue Rate", f"{safety_issue_rate*100:.1f}%")
col5.metric("Downtime Rate", f"{downtime_rate*100:.1f}%")
col6.metric("Safety & Downtime", f"{safety_downtime_rate*100:.1f}%")

st.divider() 

# Static Model Predictions Table ###############################################
st.subheader("🤖 Model Predictions (Simulated)")

st.caption("This is a static example to demonstrate what predictive output might look like.")

prediction_data = pd.DataFrame({
    "Ticket ID": [1001, 1002, 1003, 1004, 1005],
    "Problem Description": [
        "Axis noise on startup",
        "Tool jammed during operation",
        "Oil leakage from spindle",
        "Digital readout not responding",
        "Unusual vibration detected"
    ],
    "Predicted Downtime Risk (%)": [92, 65, 74, 88, 59],
    "Predicted Safety Risk (%)": [77, 12, 45, 68, 30],
    "Suggested Priority": ["🔴 High", "🟡 Medium", "🟡 Medium", "🔴 High", "🟡 Medium"]
})
st.dataframe(prediction_data, use_container_width=True)

st.caption("Disclaimer: This dashboard is a proof-of-concept. Visualizations and models are based on limited or simulated data.")

st.divider() ###############################################

st.subheader("🔍 Predict Downtime from Technician Notes")
user_input = st.text_area("Enter maintenance note:")

if user_input:
    vect = vectorizer.transform([user_input]).toarray()
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0][1 if pred else 0] * 100
    st.success(f"Prediction: {'Machine Down' if pred else 'No Downtime'} ({prob:.1f}% confidence)")

st.divider()


