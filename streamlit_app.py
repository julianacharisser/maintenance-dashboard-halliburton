# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

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

st.subheader("🔧 Top Maintenance Problem Codes")
st.caption("This chart shows the most frequent types of problems reported.")
st.bar_chart(top_problem_codes)

st.subheader("👷 Top Action Owners")
st.caption("Who is most frequently handling maintenance incidents.")
st.bar_chart(top_action_owners)

st.subheader("🕒 Incidents by Hour")
st.caption("Understand at what time of day incidents occur most frequently.")
st.bar_chart(hourly_distribution)

st.subheader("📈 Daily Incident Trend")
st.caption("Time series analysis of maintenance incidents over time.")
fig, ax = plt.subplots(figsize=(10, 4))
daily_incidents.plot(ax=ax)
ax.set_ylabel("Incident Count")
ax.set_xlabel("Date")
ax.set_title("Daily Maintenance Incidents Over Time")
st.pyplot(fig)

# Simulated time series feed (example)
st.subheader("🕰️ Simulated Time Series Feed")
st.caption("Simulating sensor or event feed using NumPy and time.")
dates = pd.date_range(end=datetime.today(), periods=30)
simulated_data = np.random.poisson(lam=5, size=len(dates))
sim_df = pd.DataFrame({"Date": dates, "Incidents": simulated_data})
fig2, ax2 = plt.subplots(figsize=(10, 4))
sim_df.set_index("Date").plot(ax=ax2)
ax2.set_ylabel("Simulated Events")
ax2.set_title("Simulated Daily Feed")
st.pyplot(fig2)

st.divider()

st.subheader("🔍 Predict Downtime from Technician Notes")
user_input = st.text_area("Enter maintenance note:")

if user_input:
    vect = vectorizer.transform([user_input]).toarray()
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0][1 if pred else 0] * 100
    st.success(f"Prediction: {'Machine Down' if pred else 'No Downtime'} ({prob:.1f}% confidence)")

st.caption("Built with ❤️ using Streamlit and scikit-learn")
