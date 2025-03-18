import pandas as pd
import numpy as np
import requests
from ctgan import CTGAN
import os

# --- Synthetic Data Generation ---
def generate_synthetic_data(n_samples=1000, output_file="synthetic_theme_park_data.csv"):
    if os.path.exists(output_file):
        return pd.read_csv(output_file)

    np.random.seed(42)
    data = pd.DataFrame({
        'Visitor_ID': np.arange(1, n_samples + 1),
        'Step_Count': np.random.normal(loc=12000, scale=3000, size=n_samples).astype(int),
        'Transaction_Amount': np.abs(np.random.normal(loc=125, scale=30, size=n_samples)).round(2),
        'Check_In_Time': np.random.uniform(9, 13, size=n_samples).round(2),
        'Check_Out_Time': np.random.uniform(17, 22, size=n_samples).round(2),
        'Loyalty_Member': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7]),
        'Weather_Condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy'], size=n_samples, p=[0.6, 0.3, 0.1]),
        'Age': np.random.normal(loc=35, scale=10, size=n_samples).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.5, 0.5]),
        'Guest_Satisfaction_Score': np.random.uniform(1, 5, size=n_samples).round(1),
        'Wait_Time': np.random.normal(loc=30, scale=10, size=n_samples).astype(int)
    })
    discrete_columns = ['Loyalty_Member', 'Weather_Condition', 'Gender']
    model = CTGAN(epochs=500)
    model.fit(data, discrete_columns=discrete_columns)
    synthetic_data = model.sample(n_samples)
    synthetic_data['Timestamp'] = pd.date_range(start="2025-01-01", periods=n_samples, freq="h").date
    synthetic_data.to_csv(output_file, index=False)
    print("Synthetic Data - First 5 records:\n", synthetic_data.head())
    return synthetic_data

# --- Load Weather Data ---
def fetch_weather_data(df_synthetic, cache_file="weather_data.csv"):
    if os.path.exists(cache_file):
        df_weather = pd.read_csv(cache_file)
        # Check if 'date' exists, otherwise try 'DATE' or raise an error
        if 'date' not in df_weather.columns and 'DATE' in df_weather.columns:
            df_weather = df_weather.rename(columns={'DATE': 'date'})
        elif 'date' not in df_weather.columns:
            raise KeyError("Cached weather_data.csv does not contain a 'date' or 'DATE' column.")
        df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date
        return df_weather

    survey_dates = pd.to_datetime(df_synthetic["Timestamp"]).dt.date.unique()
    endpoints = {
        "temperature": "https://api.data.gov.sg/v1/environment/air-temperature",
        "rainfall": "https://api.data.gov.sg/v1/environment/rainfall",
        "humidity": "https://api.data.gov.sg/v1/environment/relative-humidity"
    }
    weather_data = {"date": [], "temperature": [], "rainfall": [], "humidity": []}

    for date in survey_dates:
        proxy_date = date.replace(year=2024)
        date_str = proxy_date.strftime("%Y-%m-%d")
        daily_data = {"temperature": [], "rainfall": [], "humidity": []}

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

        weather_data["date"].append(date)
        weather_data["temperature"].append(np.mean(daily_data["temperature"]) if daily_data["temperature"] else 28)
        weather_data["rainfall"].append(np.mean(daily_data["rainfall"]) if daily_data["rainfall"] else 0)
        weather_data["humidity"].append(np.mean(daily_data["humidity"]) if daily_data["humidity"] else 75)

    df_weather = pd.DataFrame(weather_data)
    df_weather.to_csv(cache_file, index=False)
    print("Weather Data - First 5 records:\n", df_weather.head())
    return df_weather

# --- Merge and Display ---
df_synthetic = generate_synthetic_data()
df_weather = fetch_weather_data(df_synthetic)
df_synthetic["date"] = pd.to_datetime(df_synthetic["Timestamp"]).dt.date
df_merged = pd.merge(df_synthetic, df_weather, on="date", how="inner")
print("Merged Synthetic and Weather Data - First 5 records:\n", df_merged.head())