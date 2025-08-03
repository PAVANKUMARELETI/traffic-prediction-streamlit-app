# ------------------------------------------------------------------
#  Traffic Volume Prediction using Machine Learning
# ------------------------------------------------------------------
# This script loads the Metro Interstate Traffic Volume dataset,
# preprocesses the data, trains three different regression models
# (Linear Regression, Decision Tree, and Random Forest),
# evaluates their performance, and saves the trained models
# and column information for later use.
# ------------------------------------------------------------------

# 1. Import Libraries and Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- Create a directory for models if it doesn't exist ---
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Load the Dataset
try:
    data = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
except FileNotFoundError:
    print("Dataset 'Metro_Interstate_Traffic_Volume.csv' not found.")
    print("Please download it from the UCI Machine Learning Repository and place it in the same directory.")
    exit()

# 3. Explore the Dataset
print("----------------- Dataset Head -----------------")
print(data.head())
print("\n----------------- Dataset Info -----------------")
data.info()
print("\n----------------- Dataset Description -----------------")
print(data.describe())

# 4. Data Cleaning and Preprocessing

# Remove rows with 'None' holiday as they are very few and might not be informative
data = data[data['holiday'] != 'None']

# Remove Duplicate Values
print(f"\nNumber of duplicate rows before cleaning: {data.duplicated().sum()}")
data.drop_duplicates(inplace=True)
print("Duplicate rows removed.")

# 5. Feature Engineering
data['date_time'] = pd.to_datetime(data['date_time'])
data['year'] = data['date_time'].dt.year
data['month'] = data['date_time'].dt.month
data['day'] = data['date_time'].dt.day
data['hour'] = data['date_time'].dt.hour
data['day_of_week'] = data['date_time'].dt.dayofweek # Monday=0, Sunday=6

# Convert Categorical Columns to Numerical using one-hot encoding
data = pd.get_dummies(data, columns=['weather_main', 'weather_description'], drop_first=True)

# Drop the original date_time and holiday columns (holiday is now implicitly handled)
# We also drop year as it might not generalize well to future years not in the data.
data = data.drop(['date_time', 'holiday', 'year'], axis=1)

# 6. Create Input and Output Parameters
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']

# 7. Split the Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Build the Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_split=10)
}

# 9. Train and Evaluate the Models
results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    print(f"{name} trained.")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

    print(f"\n--- {name} Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.4f}")

# 10. Save the Models and Training Columns
# We will save the Random Forest model as it typically performs best for this task.
best_model = models['Random Forest']
joblib.dump(best_model, 'models/random_forest_model.pkl')
print("\nBest performing model (Random Forest) saved to 'models/random_forest_model.pkl'")

# Save the column names from the training set for the app
joblib.dump(X_train.columns, 'models/training_columns.pkl')
print("Training columns saved to 'models/training_columns.pkl'")

print("\nAll tasks completed successfully!")