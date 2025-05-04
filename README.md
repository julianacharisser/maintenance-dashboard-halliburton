# 🛠️ Maintenance Risk Dashboard
A Streamlit app to analyze technician maintenance logs and predict machine downtime using natural language in technician notes.

## 🔍 Features
- Key Metrics: View average downtime, safety incident rate, and closure time
- Top Problem Codes & Action Owners: Identify the most common failure types and who responds to them
- Time Trends: Spot hourly and daily spikes in incidents
- Downtime Prediction Tool: Enter a technician’s note and predict the likelihood of machine downtime using a DataRobot-trained model

✍️ How to Use
1. Use the sidebar to filter incidents by problem code or date
2. Scroll through the key metrics and visualizations to spot patterns
3. Enter a technician note into the prediction tool to simulate whether it may cause downtime
4. Use insights to prioritize failure types and allocate resources

🧩 Limitations
- Predictions depend on the quality of technician notes
- Further validation is needed before deployment in production
