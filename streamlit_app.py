# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Maintenance Risk Dashboard", layout="wide")
st.title("🛠️ Maintenance Risk Dashboard")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("abnormality_report_cleaned.csv")
    df['SafetyIssueBool'] = df['SafetyIssue'].str.lower() == 'yes'
    df['MachineDownBool'] = df['MachineDownNew'].str.lower() == 'yes'
    df['Notes'] = df['Maint Tech Notes/Abnormality Action Item Notes'].fillna("")
    df['Date_Created'] = pd.to_datetime(df['Date_Created'])
    df['Hour'] = df['Date_Created'].dt.hour
    return df

df = load_data()

# Load model and vectorizer
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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

# Display metrics
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

# Top problem codes
st.subheader("🔧 Top Maintenance Problem Codes")
st.bar_chart(top_problem_codes)

# Top action owners
st.subheader("👷 Top Action Owners")
st.bar_chart(top_action_owners)

# Hourly incident distribution
st.subheader("🕒 Incidents by Hour")
st.bar_chart(hourly_distribution)

st.divider()

# Prediction
st.subheader("🔍 Predict Downtime from Technician Notes")
user_input = st.text_area("Enter maintenance note:")

if user_input:
    vect = vectorizer.transform([user_input]).toarray()
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0][1 if pred else 0] * 100
    st.success(f"Prediction: {'Machine Down' if pred else 'No Downtime'} ({prob:.1f}% confidence)")

