import pandas as pd
import numpy as np
import requests
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta
import os

# --- Load Survey Dataset ---
def load_survey_data(file_path="survey.csv"):
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
def fetch_weather_data(df_survey, cache_file="weather_data.csv"):
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
class ThemePark:
    def __init__(self, env, attractions, layout="single_queue"):
        self.env = env
        self.layout = layout  
        self.attractions = {name: simpy.Resource(env, capacity=(2 if layout == "single_queue" else 4)) for name in attractions.keys()}
        self.wait_times = {name: [] for name in attractions.keys()}
        self.guest_movements = []  # Track guest paths

    def guest(self, env, name, attraction_name, service_rate):
        """
        Simulate guest visiting an attraction.
        """
        arrival_time = env.now
        with self.attractions[attraction_name].request() as req:
            yield req  # Wait for availability
            wait_time = env.now - arrival_time
            self.wait_times[attraction_name].append(wait_time)
            
            # Simulate ride duration based on service rate
            K = 7
            ride_duration = np.random.exponential(K / service_rate)
            yield env.timeout(ride_duration)

    def generate_guests(self, num_guests, arrival_rates):
        """
        Simulate guests arriving at different time periods (Morning, Afternoon, Evening).
        
        :param num_guests: Total guests for the day
        :param arrival_rates: Dictionary mapping time periods to arrival rates
        """
        for period, (start, end, rate) in arrival_rates.items():
            while self.env.now < end:
                attraction_name = random.choice(list(self.attractions.keys()))
                service_rate = 16 if self.layout == "multi_queue" else 4  # Multi-queue processes guests faster, especially for 3-4min rides
                self.env.process(self.guest(self.env, f"Guest_{int(self.env.now)}", attraction_name, service_rate))
                yield self.env.timeout(np.random.exponential(rate))

# --- Run Simulation for Different Attraction Layouts and Peak Hours ---
def run_simulation(num_guests, attractions, layout):
    """
    Execute the Theme Park Simulation with different layouts and peak hours.
    
    :param num_guests: Total guests for the day
    :param attractions: Dictionary with attraction capacities
    :param layout: "single_queue" or "multi_queue"
    :return: Average wait times per attraction
    """
    env = simpy.Environment()
    park = ThemePark(env, attractions, layout)

    # Peak hour arrival rates
    arrival_rates = {
        "Morning": (0, 120, 1),   # Guests arrive every 3 mins (low traffic) 10am-12pm
        "Afternoon": (120,420, 0.5),  # Guests arrive every 1 min (high traffic) 12pm-5pm
        "Evening": (420, 540, 3)  # Guests arrive every 2 mins (moderate traffic) 5pm - 7pm
    }

    env.process(park.generate_guests(num_guests, arrival_rates))
    env.run(until=540)  # Simulate 9 hours

    avg_wait_times = {name: (np.mean(times) if times else 0) for name, times in park.wait_times.items()}
    return avg_wait_times

# --- Define Attraction Layouts ---
attractions_layout_1 = {"Revenge of the Mummy": 2, "CYLON": 3, "Transformers": 1}  # Single Queue Layout
attractions_layout_2 = {"Revenge of the Mummy": 4, "CYLON": 5, "Transformers": 2}  # Multi Queue Layout

# Define attractions with their (x, y) coordinates
attractions_map = {
    "Revenge of the Mummy": (3, 5),  # Ancient Egypt zone
    "Battlestar Galactica: CYLON": (5, 3),  # Sci-Fi City zone
    "Transformers: The Ride": (6, 4),  # Sci-Fi City, near Battlestar Galactica
    "Puss In Boots' Giant Journey": (2, 6),  # Far Far Away zone
    "Sesame Street Spaghetti Space Chase": (4, 2)  # New York zone
}

def walking_time(attraction_1, attraction_2):
    """
    Calculate walking time between two attractions based on distance.
    """
    x1, y1 = attractions_map[attraction_1]
    x2, y2 = attractions_map[attraction_2]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance * 0.5  # Assume 0.5 minutes per unit distance

def compare_layouts():
    """
    Run simulations for two different attraction layouts and compare results.
    """
    env1 = simpy.Environment()
    env2 = simpy.Environment()

    # Layout 1 (Current setup)
    park1 = ThemePark(env1, attractions_map, layout="multi_queue")
    for i in range(13000):  # Simulating 15000 guests
        attraction_name = random.choice(list(park1.attractions.keys()))  
        service_rate = 4 if park1.layout == "single_queue" else 16
        env1.process(park1.guest(env1, f"Guest_{i}", attraction_name, service_rate))
    env1.run(until=540)

    # Layout 2 (Modified)
    attractions_map["Revenge of the Mummy"] = (4,2)  # Exchange RoTM and Sesame Street
    attractions_map["Sesame Street Spaghetti Space Chase"] = (3,5) # Exchange RoTM and Sesame Street
    park2 = ThemePark(env2, attractions_map, layout="multi_queue") # Also implement multi-queue
    for i in range(13000):
        attraction_name = random.choice(list(park2.attractions.keys()))  # Pick a random attraction
        service_rate = 4 if park2.layout == "single_queue" else 16  # Assign service rate
        env2.process(park2.guest(env2, f"Guest_{i}", attraction_name, service_rate))
    env2.run(until=540)

    # Compare Wait Times
    avg_wait_times_1 = {name: np.mean(times) if times else 0 for name, times in park1.wait_times.items()}
    avg_wait_times_2 = {name: np.mean(times) if times else 0 for name, times in park2.wait_times.items()}

    print("\n Current USS Layout - Average Wait Times:")
    for attraction, time in avg_wait_times_1.items():
        print(f"{attraction}: {time:.2f} min")

    print("\n Modified USS Layout - Average Wait Times:")
    for attraction, time in avg_wait_times_2.items():
        print(f"{attraction}: {time:.2f} min")


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
    # --- Run Simulations ---
    single_queue_wait_times = run_simulation(13000, attractions_layout_2, "single_queue")
    multi_queue_wait_times = run_simulation(13000, attractions_layout_2, "multi_queue") 
    # --- Display Results ---
    #print("\n🟢 Single Queue Layout (1 guest per ride cycle) - Average Wait Times (minutes):")
    for attraction, wait_time in single_queue_wait_times.items():
        print(f" - {attraction}: {wait_time:.2f} min")

    # print("\n🔵 Multi Queue Layout (Several guests per ride cycle) - Average Wait Times (minutes):")
    for attraction, wait_time in multi_queue_wait_times.items():
        print(f" - {attraction}: {wait_time:.2f} min")

    compare_layouts()
    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.bar(single_queue_wait_times.keys(), single_queue_wait_times.values(), alpha=0.6, label="Single Queue")
    plt.bar(multi_queue_wait_times.keys(), multi_queue_wait_times.values(), alpha=0.6, label="Multi Queue")
    plt.xlabel("Attractions")
    plt.ylabel("Average Wait Time (minutes)")
    plt.title("Comparison of Wait Times by Attraction Layout")
    plt.legend()
    plt.show()