import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from pathlib import Path

# --- Corrected File Path ---
# Construct a robust path to the data file.
# This assumes the script is run from the project's root directory.
data_path = Path("data/raw/salesmonthly.csv")

# Load the dataset using the corrected path
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: The file was not found at {data_path}")
    print("Please ensure the file exists in the 'data/raw' directory.")
    exit()


# --- Preprocessing ---
df.fillna(0, inplace=True)
df['datum'] = pd.to_datetime(df['datum'])

# --- Feature Engineering ---
# Create time-based features from the 'datum' column
# Note: 'hour' is removed as it's not relevant for monthly data
df['dayofweek'] = df['datum'].dt.dayofweek # Monday=0, Sunday=6
df['month'] = df['datum'].dt.month
df['year'] = df['datum'].dt.year

# --- Feature and Target Selection ---
# Use the new time features and the original drug sales columns
features = ['M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06', 'dayofweek', 'month', 'year']
target = 'M01AB'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# --- Model Training (Using Regressor) ---
# Use CatBoostRegressor for predicting a continuous value
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE', # Standard loss function for regression
    verbose=100,
    random_seed=42
)
catboost_model.fit(X_train, y_train)

# --- Model Evaluation (Using Regression Metrics) ---
y_pred = catboost_model.predict(X_test)

# Calculate regression metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# --- Feature Importance Visualization ---
feature_importance = catboost_model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 7))
plt.barh(np.array(X.columns)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("CatBoost Feature Importance for Regression")
plt.show()
