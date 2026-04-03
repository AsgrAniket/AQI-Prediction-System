import pandas as pd
import pickle
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fix paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'aqi_data.csv')
model_path = os.path.join(BASE_DIR, 'model', 'aqi_model.pkl')

# Load data
df = pd.read_csv(data_path)

# ---------------- CLEANING ---------------- #

# Drop useless columns
df = df.drop(columns=['City', 'Date', 'AQI_Bucket'], errors='ignore')

# Drop rows where AQI is missing
df = df.dropna(subset=['AQI'])

# Drop sparse column
df = df.drop(columns=['Xylene'], errors='ignore')

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# ---------------- MODEL ---------------- #

X = df.drop('AQI', axis=1)
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr = LinearRegression()
rf = RandomForestRegressor(
    n_estimators=80,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)


lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluation
lr_r2 = r2_score(y_test, lr.predict(X_test))
rf_r2 = r2_score(y_test, rf.predict(X_test))

print("Linear Regression R2:", lr_r2)
print("Random Forest R2:", rf_r2)

# Select best model
best_model = rf if rf_r2 > lr_r2 else lr

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
    joblib.dump(best_model, model_path, compress=3)

print("✅ Model saved successfully!")