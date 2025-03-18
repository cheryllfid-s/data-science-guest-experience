import pandas as pd
import numpy as np
from ctgan import CTGAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os  # Added import

# --- Synthetic Data Generation ---
def generate_synthetic_data(n_samples=1000, output_file="synthetic_theme_park_data.csv"):
    if os.path.exists(output_file):
        return pd.read_csv(output_file)

    np.random.seed(42)
    data = pd.DataFrame({
        'Visitor_ID': np.arange(1, n_samples + 1),
        'Wait_Time': np.random.normal(loc=30, scale=10, size=n_samples).astype(int),
        'Weather_Condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy'], size=n_samples, p=[0.6, 0.3, 0.1])
    })
    discrete_columns = ['Weather_Condition']
    model = CTGAN(epochs=500)
    model.fit(data, discrete_columns=discrete_columns)
    synthetic_data = model.sample(n_samples)
    synthetic_data['Timestamp'] = pd.date_range(start="2025-01-01", periods=n_samples, freq="h").date
    synthetic_data.to_csv(output_file, index=False)
    print("Synthetic Data - First 5 records:\n", synthetic_data.head())
    return synthetic_data

# --- Predict Wait Time ---
df = generate_synthetic_data()

# Dummy weather features (for standalone use; integrate NEA if needed)
df['TEMP'] = np.random.normal(28, 2, len(df))
df['PRCP'] = np.random.exponential(5, len(df))
df['HUMID'] = np.random.uniform(60, 90, len(df))

features = ['TEMP', 'PRCP', 'HUMID']
target = 'Wait_Time'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')