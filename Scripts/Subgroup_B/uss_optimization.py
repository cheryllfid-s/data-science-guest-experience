import pandas as pd
import numpy as np
import requests
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta

# --- Dataset 1: Load Singapore Weather API Data ---
def fetch_weather_data(collection_id=1459, start_date="2024-01-01", end_date="2024-12-31"):
    """Fetch historical weather data from SG Government API."""
    url = f"https://api-production.data.gov.sg/v2/public/api/collections/{collection_id}/metadata"
    response = requests.get(url)
    metadata = response.json()
    print("Weather API Metadata:", metadata)

    # Assuming this is a historical dataset (e.g., daily weather)
    # Replace with actual data endpoint if available (e.g., NEA forecast API)
    base_url = "https://api.data.gov.sg/v1/environment/24-hour-weather-forecast"
    params = {"date": start_date}  # Simplified; adjust for date range
    response = requests.get(base_url, params=params)
    weather_data = response.json()

    # Process into DataFrame (example structure)
    df_weather = pd.DataFrame({
        "date": pd.date_range(start_date, end_date, freq="D"),
        "temperature": np.random.normal(28, 2, 366),  # Placeholder
        "rainfall": np.random.exponential(5, 366),    # Placeholder
        "humidity": np.random.uniform(60, 90, 366)    # Placeholder
    })
    return df_weather

# --- Dataset 2: Load USS Data from Kaggle ---
def load_uss_data():
    """Load USS dataset from KaggleHub."""
    file_path = ""  # Specify file path if multiple files in dataset
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "ayushtankha/hackathon",
        file_path
    )
    print("USS Data - First 5 records:\n", df.head())

    # Filter to Singapore data and preprocess (assuming date and relevant columns exist)
    df_uss = df[df["location"] == "Singapore"].copy() if "location" in df.columns else df.copy()
    df_uss["date"] = pd.to_datetime(df_uss["date"])  # Adjust column name as needed
    return df_uss

# --- Merge Datasets ---
def merge_datasets(df_weather, df_uss):
    """Join weather and USS data by date."""
    df_merged = pd.merge(df_weather, df_uss, on="date", how="inner")
    print("Merged Data - First 5 records:\n", df_merged.head())
    return df_merged

# --- Question 1: Demand Prediction for Attractions and Services ---
def predict_demand(df):
    """Predict daily demand (e.g., foot traffic) using weather features."""
    # Features: Weather + historical demand
    X = df[["temperature", "rainfall", "humidity"]]
    y = df["demand"]  # Replace with actual column (e.g., foot traffic)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Demand Prediction - RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Forecast for next 7 days
    forecast_url = "https://api.data.gov.sg/v1/environment/24-hour-weather-forecast"
    response = requests.get(forecast_url)
    forecast_data = response.json()

    # Simplified forecast DataFrame
    future_dates = pd.date_range(start=datetime.now(), periods=7, freq="D")
    df_future = pd.DataFrame({
        "date": future_dates,
        "temperature": [28] * 7,  # Replace with actual forecast
        "rainfall": [5] * 7,
        "humidity": [75] * 7
    })
    future_pred = model.predict(df_future[["temperature", "rainfall", "humidity"]])
    print("7-Day Demand Forecast:\n", pd.DataFrame({"Date": future_dates, "Predicted Demand": future_pred}))

    return model

# --- Question 2: Optimization of Attraction Layouts and Schedules ---
def simulate_guest_flow(env, num_guests, service_rate, layout="single_queue"):
    """Simulate guest flow with Discrete Event Simulation."""
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
        yield env.timeout(np.random.exponential(0.1))

    return wait_times

def optimize_layout(df, model):
    """Optimize layout based on predicted demand."""
    env = simpy.Environment()
    future_demand = model.predict(df[["temperature", "rainfall", "humidity"]].iloc[-7:])  # Last 7 days as example
    avg_demand = np.mean(future_demand)

    # Simulate single vs multi-queue
    wait_times_single = env.run(until=simulate_guest_flow(env, avg_demand, service_rate=5, layout="single_queue"))
    env = simpy.Environment()
    wait_times_multi = env.run(until=simulate_guest_flow(env, avg_demand, service_rate=5, layout="multi_queue"))

    print(f"Single Queue - Avg Wait Time: {np.mean(wait_times_single):.2f} minutes")
    print(f"Multi Queue - Avg Wait Time: {np.mean(wait_times_multi):.2f} minutes")

    # Visualization
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
    # Load and merge datasets
    df_weather = fetch_weather_data()
    df_uss = load_uss_data()
    df_merged = merge_datasets(df_weather, df_uss)

    # Question 1
    demand_model = predict_demand(df_merged)

    # Question 2
    optimize_layout(df_merged, demand_model)

# --- Documentation ---
"""
Installation:
1. Install Python 3.8+
2. Install dependencies: pip install pandas numpy scikit-learn matplotlib simpy requests kagglehub[pandas-datasets]

How to Run:
1. Save as `uss_optimization.py`.
2. Run: `python uss_optimization.py`.

Notes:
- Replace placeholder weather data with actual API calls.
- Adjust USS dataset columns (e.g., 'demand') to match your data.
"""