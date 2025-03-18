# To vary parameters and screenshot results for wiki use


import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta
import os

# --- Dataset 1: Load Singapore Weather API Data ---
def fetch_weather_data(start_date="2025-01-01", end_date="2025-03-31"):
    """Fetch historical weather data from SG NEA API."""
    # Define date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # API endpoints
    endpoints = {
        "temperature": "https://api.data.gov.sg/v1/environment/air-temperature",
        "rainfall": "https://api.data.gov.sg/v1/environment/rainfall",
        "humidity": "https://api.data.gov.sg/v1/environment/relative-humidity"
    }

    # Store daily averages
    weather_data = {"date": [], "temperature": [], "rainfall": [], "humidity": []}

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        daily_data = {"temperature": [], "rainfall": [], "humidity": []}

        # Fetch data for each endpoint
        for key, url in endpoints.items():
            try:
                response = requests.get(url, params={"date": date_str})
                data = response.json()
                items = data.get("items", [])

                if not items:
                    continue

                # Extract readings
                for item in items:
                    timestamp = pd.to_datetime(item["timestamp"])
                    if timestamp.date() != date.date():
                        continue

                    readings = item.get("readings", [])
                    for reading in readings:
                        value = reading.get("value")
                        if value is not None:
                            if key == "temperature":
                                daily_data["temperature"].append(value)
                            elif key == "rainfall":
                                daily_data["rainfall"].append(value)
                            elif key == "humidity":
                                daily_data["humidity"].append(value)
            except Exception as e:
                print(f"Error fetching {key} for {date_str}: {e}")

        # Calculate daily averages (or use fallback if no data)
        weather_data["date"].append(date)
        weather_data["temperature"].append(np.mean(daily_data["temperature"]) if daily_data["temperature"] else 28)
        weather_data["rainfall"].append(np.mean(daily_data["rainfall"]) if daily_data["rainfall"] else 0)
        weather_data["humidity"].append(np.mean(daily_data["humidity"]) if daily_data["humidity"] else 75)

    # Create DataFrame
    df_weather = pd.DataFrame(weather_data)
    print("Weather Data - First 5 records:\n", df_weather.head())
    return df_weather

# --- Dataset 2: Load USS Survey Data ---
def load_uss_data(file_path="survey.csv"):
    """Load USS survey data from specified CSV path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Survey file not found at {file_path}")

    df = pd.read_csv(file_path)
    print("USS Survey Data - First 5 records:\n", df.head())

    # Preprocess
    df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
    df["overall_experience"] = pd.to_numeric(df["On a scale of 1-5, how would you rate your overall experience at USS?"], errors="coerce")
    df["wait_time"] = df["How long did you wait in line for rides on average during your visit?"].str.extract(r'(\d+)').astype(float)
    return df

# --- Merge Datasets ---
def merge_datasets(df_weather, df_uss):
    """Join weather and survey data by date."""
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date
    df_merged = pd.merge(df_uss, df_weather, on="date", how="inner")
    if df_merged.empty:
        print("Warning: Merged DataFrame is empty.")
    else:
        print("Merged Data - First 5 records:\n", df_merged.head())
    return df_merged

# --- Question 1: Demand Prediction for Attractions and Services ---
def predict_demand(df):
    """Predict demand (proxied by experience) using weather and survey data."""
    if df.empty:
        print("Cannot predict demand: Merged dataset is empty.")
        return None

    X = df[["temperature", "rainfall", "humidity"]]
    y = df["overall_experience"].fillna(df["overall_experience"].mean())

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Demand/Experience Prediction - RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Forecast for next 7 days
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
    print("7-Day Experience Forecast (1-5):\n", pd.DataFrame({"Date": future_dates, "Predicted Rating": future_pred}))

    return model

# --- Question 2: Optimization of Attraction Layouts and Schedules ---
def simulate_guest_flow(env, num_guests, service_rate, wait_time_data, layout="single_queue", duration=1440):
    """Simulate guest flow for a fixed duration and return wait times."""
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
        env.timeout(np.random.exponential(0.1)) # remove yield, not a generator func

    env.run(until=duration)
    avg_survey_wait = wait_time_data.mean() if not wait_time_data.isna().all() else 0
    print(f"Simulation Adjusted with Survey Avg Wait Time: {avg_survey_wait:.2f} minutes")
    return wait_times

def optimize_layout(df, model):
    """Optimize layout based on predicted demand and survey wait times."""
    if df.empty or model is None:
        print("Cannot optimize layout: Merged dataset is empty or model is not trained.")
        return

    env = simpy.Environment()
    avg_demand = len(df) / df["date"].nunique()  # Proxy: avg visitors per day

    wait_times_single = list(simulate_guest_flow(env, avg_demand, service_rate=5, wait_time_data=df["wait_time"], layout="single_queue", duration=1440))
    env = simpy.Environment()
    wait_times_multi = list(simulate_guest_flow(env, avg_demand, service_rate=5, wait_time_data=df["wait_time"], layout="multi_queue", duration=1440))

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
    survey_path = "survey.csv"
    df_weather = fetch_weather_data()
    df_uss = load_uss_data(survey_path)
    df_merged = merge_datasets(df_weather, df_uss)
    demand_model = predict_demand(df_merged)
    optimize_layout(df_merged, demand_model)

# --- Documentation ---
"""
Installation:
1. Install Python 3.8+
2. Install dependencies: pip install pandas numpy scikit-learn matplotlib simpy requests

How to Run:
1. Save as `uss_optimization.py` in /Scripts/Subgroup_B/.
2. Ensure `survey.csv` is at /data/survey.csv.
3. Run from /Scripts/Subgroup_B/: `python uss_optimization.py`.
"""