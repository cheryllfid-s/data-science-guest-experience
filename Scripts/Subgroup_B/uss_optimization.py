import pandas as pd
import numpy as np
import requests
from ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta
import os

# --- Synthetic Data Generation ---
def generate_synthetic_data(n_samples=1000, output_file="synthetic_theme_park_data.csv"):
    expected_columns = ['Visitor_ID', 'Guest_Satisfaction_Score', 'Wait_Time', 'Weather_Condition', 'Timestamp']
    if os.path.exists(output_file):
        df_synthetic = pd.read_csv(output_file)
        if all(col in df_synthetic.columns for col in expected_columns):
            return df_synthetic
        print("Warning: Cached synthetic data missing expected columns. Regenerating...")
        os.remove(output_file)

    np.random.seed(42)
    data = pd.DataFrame({
        'Visitor_ID': np.arange(1, n_samples + 1),
        'Guest_Satisfaction_Score': np.random.uniform(1, 5, size=n_samples).round(1),
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

# --- Load Weather Data ---
def fetch_weather_data(df_synthetic, cache_file="weather_data.csv"):
    if os.path.exists(cache_file):
        df_weather = pd.read_csv(cache_file)
        rename_dict = {'DATE': 'date', 'TEMP': 'temperature', 'PRCP': 'rainfall', 'HUMID': 'humidity'}
        df_weather = df_weather.rename(columns={k: v for k, v in rename_dict.items() if k in df_weather.columns})
        if 'date' not in df_weather.columns:
            raise KeyError("Cached weather_data.csv does not contain a 'date' or 'DATE' column.")
        if not all(col in df_weather.columns for col in ['temperature', 'rainfall', 'humidity']):
            print("Warning: Cached weather data missing expected columns. Regenerating...")
            os.remove(cache_file)
        else:
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

# --- Merge Datasets ---
def merge_datasets(df_weather, df_synthetic):
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date
    df_synthetic["date"] = pd.to_datetime(df_synthetic["Timestamp"]).dt.date
    df_merged = pd.merge(df_synthetic, df_weather, on="date", how="inner")
    print("Merged DataFrame columns:", df_merged.columns.tolist())
    return df_merged

# --- Predict Satisfaction ---
def predict_demand(df):
    if df.empty:
        print("Cannot predict demand: Merged dataset is empty.")
        return None

    X = df[["temperature", "rainfall", "humidity"]]
    y = df["Guest_Satisfaction_Score"].fillna(df["Guest_Satisfaction_Score"].mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Satisfaction Prediction - RMSE: {rmse:.2f}, R²: {r2:.2f}")

    forecast_url = "https://api.data.gov.sg/v1/environment/4-day-weather-forecast"
    response = requests.get(forecast_url)
    forecast_data = response.json().get("items", [{}])[0].get("forecasts", [])
    future_dates = pd.date_range(start=datetime.now(), periods=7, freq="D")
    n_days = len(future_dates)
    forecast_data_padded = forecast_data + [forecast_data[-1] if forecast_data else {"temperature": {"high": 28}, "forecast": "Partly cloudy"}] * (n_days - len(forecast_data))
    temperature = [f.get("temperature", {}).get("high", 28) for f in forecast_data_padded[:n_days]]
    rainfall = [5 if "rain" in f.get("forecast", "").lower() else 0 for f in forecast_data_padded[:n_days]]

    df_future = pd.DataFrame({
        "date": future_dates,
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": [75] * n_days
    })
    future_pred = model.predict(df_future[["temperature", "rainfall", "humidity"]])
    print("7-Day Satisfaction Forecast (1-5):\n", pd.DataFrame({"Date": future_dates, "Predicted Rating": future_pred}))
    return model

# --- Optimize Layouts ---
def simulate_guest_flow(env, num_guests, service_rate, wait_time_data, layout="single_queue", duration=1440):
    wait_times = []

    def guest(env, name, resource):
        arrival_time = env.now
        with resource.request() as req:
            yield req
            wait_time = env.now - arrival_time
            wait_times.append(wait_time)
            yield env.timeout(np.random.exponential(1 / service_rate))

    resource = simpy.Resource(env, capacity=1 if layout == "single_queue" else 2)
    for i in range(int(num_guests)):
        env.process(guest(env, f"Guest_{i}", resource))
        env.timeout(np.random.exponential(0.1))

    env.run(until=duration)
    avg_survey_wait = wait_time_data.mean() if not wait_time_data.isna().all() else 0
    print(f"Simulation Adjusted with Survey Avg Wait Time: {avg_survey_wait:.2f} minutes")
    return wait_times

def optimize_layout(df, model):
    if df.empty or model is None:
        print("Cannot optimize layout: Merged dataset is empty or model is not trained.")
        return

    env = simpy.Environment()
    avg_demand = len(df) / df["date"].nunique()

    wait_times_single = list(simulate_guest_flow(env, avg_demand, service_rate=5, wait_time_data=df["Wait_Time"], layout="single_queue", duration=1440))
    env = simpy.Environment()
    wait_times_multi = list(simulate_guest_flow(env, avg_demand, service_rate=5, wait_time_data=df["Wait_Time"], layout="multi_queue", duration=1440))

    print(f"Single Queue - Avg Wait Time: {np.mean(wait_times_single):.2f} minutes")
    print(f"Multi Queue - Avg Wait Time: {np.mean(wait_times_multi):.2f} minutes")

    plt.figure(figsize=(10, 6))
    plt.hist(wait_times_single, bins=20, alpha=0.5, label="Single Queue")
    plt.hist(wait_times_multi, bins=20, alpha=0.5, label="Multi Queue")
    plt.xlabel("Wait Time (minutes)")
    plt.ylabel("Frequency")
    plt.title("Wait Time Distribution by Layout")
    plt.legend()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    df_synthetic = generate_synthetic_data()
    df_weather = fetch_weather_data(df_synthetic)
    df_merged = merge_datasets(df_weather, df_synthetic)
    demand_model = predict_demand(df_merged)
    optimize_layout(df_merged, demand_model)