# train_model.py

import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess data
df = pd.read_csv("abnormality_report_cleaned.csv")
df['SafetyIssueBool'] = df['SafetyIssue'].str.lower() == 'yes'
df['MachineDownBool'] = df['MachineDownNew'].str.lower() == 'yes'
df['Notes'] = df['Maint Tech Notes/Abnormality Action Item Notes'].fillna("")

# Feature extraction
vectorizer = CountVectorizer(max_features=300, stop_words='english')
X = vectorizer.fit_transform(df['Notes']).toarray()
y = df['MachineDownBool']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "rf_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")