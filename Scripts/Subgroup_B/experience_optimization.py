import pandas as pd
import numpy as np
import requests
from ctgan import CTGAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# --- Synthetic Data Generation ---
def generate_synthetic_data(n_samples=1000, output_file="synthetic_theme_park_data.csv"):
    if os.path.exists(output_file):
        return pd.read_csv(output_file)

    np.random.seed(42)
    # Exclude Timestamp from CTGAN training
    data = pd.DataFrame({
        'Visitor_ID': np.arange(1, n_samples + 1),
        'Wait_Time': np.random.normal(loc=30, scale=10, size=n_samples).astype(int),
        'Weather_Condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy'], size=n_samples, p=[0.6, 0.3, 0.1])
    })
    discrete_columns = ['Weather_Condition']
    model = CTGAN(epochs=500)
    model.fit(data, discrete_columns=discrete_columns)
    synthetic_data = model.sample(n_samples)
    # Add Timestamp after generation
    synthetic_data['Timestamp'] = pd.date_range(start="2025-01-01", periods=n_samples, freq="h").date
    synthetic_data.to_csv(output_file, index=False)
    print("Synthetic Data - First 5 records:\n", synthetic_data.head())
    return synthetic_data

# --- Load Weather Data ---
def fetch_weather_data(df_synthetic, cache_file="weather_data.csv"):
    if os.path.exists(cache_file):
        df_weather = pd.read_csv(cache_file)
        df_weather["DATE"] = pd.to_datetime(df_weather["DATE"]).dt.date
        return df_weather

    survey_dates = pd.to_datetime(df_synthetic["Timestamp"]).dt.date.unique()
    endpoints = {
        "TEMP": "https://api.data.gov.sg/v1/environment/air-temperature",
        "PRCP": "https://api.data.gov.sg/v1/environment/rainfall",
        "HUMID": "https://api.data.gov.sg/v1/environment/relative-humidity"
    }
    weather_data = {"DATE": [], "TEMP": [], "PRCP": [], "HUMID": []}

    for date in survey_dates:
        proxy_date = date.replace(year=2024)
        date_str = proxy_date.strftime("%Y-%m-%d")
        daily_data = {"TEMP": [], "PRCP": [], "HUMID": []}

        for key, url in endpoints.items():
            try:
                response = requests.get(url, params={"date": date_str})
                data = response.json()
                items = data.get("items", [])
                for item in items:
                    timestamp = pd.to_datetime(item["timestamp"])
                    if timestamp.date() != proxy_date:
                        continue
                    readings = item.get("readings", [])
                    for reading in readings:
                        value = reading.get("value")
                        if value is not None:
                            daily_data[key].append(value)
            except Exception as e:
                print(f"Error fetching {key} for {date_str}: {e}")

        weather_data["DATE"].append(date)
        weather_data["TEMP"].append(np.mean(daily_data["TEMP"]) if daily_data["TEMP"] else 28)
        weather_data["PRCP"].append(np.mean(daily_data["PRCP"]) if daily_data["PRCP"] else 0)
        weather_data["HUMID"].append(np.mean(daily_data["HUMID"]) if daily_data["HUMID"] else 75)

    df_weather = pd.DataFrame(weather_data)
    df_weather.to_csv(cache_file, index=False)
    return df_weather

# --- Merge Datasets ---
def merge_datasets(df_synthetic, df_weather):
    df_synthetic["DATE"] = pd.to_datetime(df_synthetic["Timestamp"]).dt.date
    df_merged = pd.merge(df_synthetic, df_weather, on="DATE", how="left")
    df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)
    return df_merged

# --- Predict Wait Time ---
df_synthetic = generate_synthetic_data()
df_weather = fetch_weather_data(df_synthetic)
df = merge_datasets(df_synthetic, df_weather)

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