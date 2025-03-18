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

# --- Load Survey Dataset ---
def load_survey_data(file_path="/Users/derr/Documents/DSA3101/Project/DSA3101 data-science-guest-experience/data-science-guest-experience/Scripts/Subgroup_B/survey.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please provide the survey dataset.")

    df = pd.read_csv(file_path)

    # Rename columns for consistency
    df = df.rename(columns={
        "On a scale of 1-5, how would you rate your overall experience at USS?": "Guest_Satisfaction_Score",
        "How long did you wait in line for rides on average during your visit?": "Wait_Time",
        "Timestamp": "Timestamp"
    })

    # Convert Wait_Time to numeric values (adjust based on actual survey options)
    wait_time_mapping = {
        "Less than 15 mins": 10,
        "15-30 mins": 22.5,
        "30-45 mins": 37.5,
        "45 mins - 1 hr": 52.5,
        "More than 1 hr": 75
    }
    df["Wait_Time"] = df["Wait_Time"].map(wait_time_mapping).fillna(37.5)  # Default to 37.5 if unmapped

    # Ensure Guest_Satisfaction_Score is numeric
    df["Guest_Satisfaction_Score"] = pd.to_numeric(df["Guest_Satisfaction_Score"], errors="coerce")

    # Convert Timestamp to date
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.date

    required_columns = ["Guest_Satisfaction_Score", "Wait_Time", "Timestamp"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    print("Survey Data - First 5 records:\n", df[required_columns].head())
    return df

# --- Fetch Weather Data ---
def fetch_weather_data(df_survey, cache_file="/Users/derr/Documents/DSA3101/Project/DSA3101 data-science-guest-experience/data-science-guest-experience/Scripts/Subgroup_B/weather_data.csv"):
    if os.path.exists(cache_file):
        df_weather = pd.read_csv(cache_file)
        rename_dict = {'DATE': 'date', 'TEMP': 'temperature', 'PRCP': 'rainfall', 'HUMID': 'humidity'}
        df_weather = df_weather.rename(columns={k: v for k, v in rename_dict.items() if k in df_weather.columns})
        if 'date' not in df_weather.columns:
            raise KeyError("Cached weather_data.csv does not contain a 'date' or 'DATE' column.")
        df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date
        return df_weather

    survey_dates = pd.to_datetime(df_survey["Timestamp"]).dt.date.unique()
    endpoints = {
        "temperature": "https://api.data.gov.sg/v1/environment/air-temperature",
        "rainfall": "https://api.data.gov.sg/v1/environment/rainfall",
        "humidity": "https://api.data.gov.sg/v1/environment/relative-humidity"
    }
    weather_data = {"date": [], "temperature": [], "rainfall": [], "humidity": []}

    for date in survey_dates:
        proxy_date = date  # Use survey date directly (assuming 2023–2024 data)
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
def merge_datasets(df_survey, df_weather):
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date
    df_survey["date"] = pd.to_datetime(df_survey["Timestamp"]).dt.date
    df_merged = pd.merge(df_survey, df_weather, on="date", how="inner")
    print("Merged DataFrame columns:", df_merged.columns.tolist())
    return df_merged

# --- Predict Satisfaction ---
def predict_demand(df):
    if df.empty:
        print("Cannot predict demand: Dataset is empty.")
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
        print("Cannot optimize layout: Dataset is empty or model is not trained.")
        return

    env = simpy.Environment()
    avg_demand = len(df) / df["Timestamp"].nunique()

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
    # Load survey data
    df_survey = load_survey_data()

    # Fetch weather data
    df_weather = fetch_weather_data(df_survey)

    # Merge datasets
    df_merged = merge_datasets(df_survey, df_weather)

    # Run prediction and optimization
    demand_model = predict_demand(df_merged)
    optimize_layout(df_merged, demand_model)