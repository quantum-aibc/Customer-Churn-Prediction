# Customer-Churn-Prediction
 Goal: Predict customer churn using ML models (Logistic Regression, Random Forest, XGBoost).
 Use Case: Helps businesses retain customers by identifying potential churners.

 Customer-Churn-Prediction/
│── data/
│   ├── customer_data.csv
│── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│── src/
│   ├── train.py
│   ├── predict.py
│── app/
│   ├── app.py
│── README.md
│── requirements.txt
│── Dockerfile
Key Features
✔ ML models to classify customers as Churn/No-Churn
✔ Feature engineering and data preprocessing pipeline
✔ API-based real-time customer churn prediction

📌 Code for Model Training (src/train.py)

python
Copy
Edit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Data
df = pd.read_csv("../data/customer_data.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("../models/churn_model.pkl", "wb"))
print("Model Trained and Saved!")
