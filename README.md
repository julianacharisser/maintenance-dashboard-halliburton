# 🛠️ Halliburton Maintenance Analytics Dashboard
A Streamlit-powered dashboard to analyze Halliburton’s maintenance logs, uncover downtime trends, and predict risk using natural language from technician notes.

## 🔍 Features

- **Key Metrics**  
  View total incidents, average downtime, safety issue rate, and closure time.

- **Top Problem Codes**  
  Identify the biggest downtime contributors and resolution bottlenecks.

- **Risk & Resolution Insights**  
  Visualize median resolution time and safety vs. productivity risk by issue type.

- **Downtime Prediction Tool**  
  Enter technician notes and predict machine downtime risk using a DataRobot-trained model.

- **Outlier Detection**  
  Detect abnormal incidents with unusually high downtime for further root cause analysis.

---

## 🧪 How to Use

1. Use the sidebar to filter data by maintenance code or date range.
2. Explore key metrics and visualizations to identify patterns and top risks.
3. Scroll through downtime and resolution insights to prioritize attention.
4. Enter a technician note into the predictor to estimate risk and triage smarter.

---

## ⚠️ Limitations

- Predictions depend heavily on the clarity and consistency of technician notes.
- This dashboard is a proof-of-concept and not yet validated for production deployment.

---

## 💡 Technologies Used

- Python, Streamlit, pandas, seaborn, matplotlib
- Scikit-learn (Random Forest model)
- Joblib for model loading
- DataRobot for model training and export
