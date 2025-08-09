import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('data/raw/salesmonthly.csv')  # Replace with your file name

# Inspect the dataset to understand its structure
print(df.head())

# Preprocessing (example: handling missing values)
df.fillna(0, inplace=True)  # Replace missing values with 0

# Convert 'datum' column to datetime (if applicable)
if 'datum' in df.columns:
    df['datum'] = pd.to_datetime(df['datum'])

# Select features and target variable
X = df[['M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']]  # Features
y = df['M01AB']  # Target variable (forecasting M01AB)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,  # Number of boosting rounds
    learning_rate=0.1,  # Step size shrinkage
    max_depth=5,        # Maximum depth of a tree
    random_state=42     # For reproducibility
)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Feature importance visualization (optional)
import matplotlib.pyplot as plt

feature_importance = xgb_model.feature_importances_
plt.barh(X.columns, feature_importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()
