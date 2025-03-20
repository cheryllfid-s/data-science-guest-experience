import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta
import os

# --- Load Survey Data ---
def load_survey_data(file_path="survey.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please provide the survey dataset.")

    df = pd.read_csv(file_path)

    # Print column names for debugging
    print("Columns in survey.csv:", df.columns.tolist())

    # Rename columns for consistency
    df = df.rename(columns={
        "On a scale of 1-5, how would you rate your overall experience at USS?": "Guest_Satisfaction_Score",
        "How long did you wait in line for rides on average during your visit?": "Wait_Time",
        "Timestamp": "Timestamp",
        "Which ride or attraction was your favourite?": "Attraction"
    })

    # Convert Wait_Time to numeric values
    wait_time_mapping = {
        "Less than 15 mins": 10,
        "15-30 mins": 22.5,
        "30-45 mins": 37.5,
        "45 mins - 1 hr": 52.5,
        "More than 1 hr": 75
    }
    df["Wait_Time"] = df["Wait_Time"].map(wait_time_mapping).fillna(37.5)

    # Add a simulated Event column
    np.random.seed(42)
    df["Event"] = np.random.choice(['None', 'Special Event'], size=len(df), p=[0.8, 0.2])

    # Enhance Wait_Time using long wait time responses with more granularity
    required_columns_for_long_wait = [
        'Attraction', 'Timestamp', 'Event', 'Wait_Time', 'Guest_Satisfaction_Score',
        'Which part of the year did you visit USS?', 'Did you purchase the Express Pass?',
        'What was the main purpose of your visit?'
    ]

    if 'Did you experience any rides with longer-than-expected wait times? If yes, which ride(s)?' not in df.columns:
        print("Warning: Long wait time column missing. Skipping wait time enhancement.")
        long_wait_df = pd.DataFrame(columns=required_columns_for_long_wait)
    else:
        long_wait_rides = df['Did you experience any rides with longer-than-expected wait times? If yes, which ride(s)?'].str.split(', ', expand=True).stack().reset_index()
        long_wait_rides.columns = ['original_index', 'split_index', 'Attraction']
        long_wait_rides = long_wait_rides[long_wait_rides['Attraction'].notna()]

        queue_worth_col = 'Did you feel that overall, the queueing time was worth the experience of the attraction?'
        if queue_worth_col in df.columns:
            wait_time_adjusted = []
            for idx in long_wait_rides['original_index']:
                queue_worth = df[queue_worth_col].iloc[idx]
                if queue_worth == 'No':
                    wait_time_adjusted.append(90)
                else:
                    wait_time_adjusted.append(75)
        else:
            print(f"Warning: Column '{queue_worth_col}' not found. Using default wait time of 75 minutes for long waits.")
            wait_time_adjusted = [75] * len(long_wait_rides)

        long_wait_df = pd.DataFrame({
            'Attraction': long_wait_rides['Attraction'],
            'Timestamp': df['Timestamp'].iloc[long_wait_rides['original_index']].values,
            'Event': df['Event'].iloc[long_wait_rides['original_index']].values,
            'Wait_Time': wait_time_adjusted,
            'Guest_Satisfaction_Score': df['Guest_Satisfaction_Score'].iloc[long_wait_rides['original_index']].values,
            'Which part of the year did you visit USS?': df['Which part of the year did you visit USS?'].iloc[long_wait_rides['original_index']].values,
            'Did you purchase the Express Pass?': df['Did you purchase the Express Pass?'].iloc[long_wait_rides['original_index']].values if 'Did you purchase the Express Pass?' in df.columns else [None] * len(long_wait_rides),
            'What was the main purpose of your visit?': df['What was the main purpose of your visit?'].iloc[long_wait_rides['original_index']].values if 'What was the main purpose of your visit?' in df.columns else [None] * len(long_wait_rides)
        })

    df = pd.concat([df[['Attraction', 'Wait_Time', 'Timestamp', 'Event', 'Guest_Satisfaction_Score', 'Which part of the year did you visit USS?', 'Did you purchase the Express Pass?', 'What was the main purpose of your visit?']], long_wait_df], ignore_index=True)

    # Print Wait_Time distribution
    print("Wait Time Distribution:\n", df["Wait_Time"].value_counts())

    # Ensure Guest_Satisfaction_Score is numeric
    df["Guest_Satisfaction_Score"] = pd.to_numeric(df["Guest_Satisfaction_Score"], errors="coerce")

    # Convert Timestamp to date
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.date

    # Add Day_of_Week
    df['Day_of_Week'] = pd.to_datetime(df['Timestamp']).dt.day_name()

    # Define valid attractions
    valid_attractions = [
        "Revenge of the Mummy",
        "Battlestar Galactica: CYLON",
        "Transformers: The Ride",
        "Puss In Boots' Giant Journey",
        "Sesame Street Spaghetti Space Chase"
    ]

    # Filter out invalid attraction names
    df = df[df['Attraction'].isin(valid_attractions)]
    if df.empty:
        raise ValueError("No valid attractions found in the survey data after filtering.")

    required_columns = ["Attraction", "Wait_Time", "Timestamp", "Event", "Guest_Satisfaction_Score", "Day_of_Week", "Which part of the year did you visit USS?"]
    if 'Did you purchase the Express Pass?' in df.columns:
        required_columns.append('Did you purchase the Express Pass?')
    if 'What was the main purpose of your visit?' in df.columns:
        required_columns.append('What was the main purpose of your visit?')

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    print("Survey Data - First 5 records:\n", df[required_columns].head())
    return df

# --- Load Weather Data ---
def fetch_weather_data(df_survey, cache_file="weather_data.csv"):
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

    survey_dates = pd.to_datetime(df_survey["Timestamp"]).dt.date.unique()
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
def merge_datasets(df_weather, df_survey):
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date
    df_survey["date"] = pd.to_datetime(df_survey["Timestamp"]).dt.date
    df_merged = pd.merge(df_survey, df_weather, on="date", how="inner")
    print("Merged DataFrame columns:", df_merged.columns.tolist())
    return df_merged

# --- Predict Demand ---
def predict_demand(df):
    if df.empty:
        print("Cannot predict demand: Merged dataset is empty.")
        return None, None, None

    # Aggregate data by date and attraction to calculate demand
    agg_dict = {
        'temperature': 'mean',
        'rainfall': 'mean',
        'humidity': 'mean',
        'Event': 'first',
        'Wait_Time': 'mean',
        'Guest_Satisfaction_Score': 'mean',
        'Day_of_Week': 'first',
        'Which part of the year did you visit USS?': 'first',
        'Attraction': 'size'
    }
    if 'Did you purchase the Express Pass?' in df.columns:
        agg_dict['Did you purchase the Express Pass?'] = 'first'
    if 'What was the main purpose of your visit?' in df.columns:
        agg_dict['What was the main purpose of your visit?'] = 'first'

    df_agg = df.groupby(['date', 'Attraction']).agg(agg_dict).rename(columns={'Attraction': 'Attraction_Visits'}).reset_index()

    # Encode categorical variables
    dummy_cols = ['Event', 'Day_of_Week', 'Which part of the year did you visit USS?']
    if 'Did you purchase the Express Pass?' in df_agg.columns:
        dummy_cols.append('Did you purchase the Express Pass?')
    if 'What was the main purpose of your visit?' in df_agg.columns:
        dummy_cols.append('What was the main purpose of your visit?')

    df_agg = pd.get_dummies(df_agg, columns=dummy_cols, drop_first=True)

    # Define features
    feature_cols = ['temperature', 'rainfall', 'humidity', 'Wait_Time', 'Guest_Satisfaction_Score'] + \
                   [col for col in df_agg.columns if col.startswith('Event_') or col.startswith('Day_of_Week_') or col.startswith('Which part of the year did you visit USS?_') or col.startswith('Did you purchase the Express Pass?_') or col.startswith('What was the main purpose of your visit?_')]
    X = df_agg[feature_cols]
    y = df_agg['Attraction_Visits']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Demand Prediction - RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-Validated R²: {scores.mean():.2f} ± {scores.std():.2f}")

    # Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Feature Importance:\n", feature_importance.head(10))

    # Forecast demand for the next 7 days
    forecast_url = "https://api.data.gov.sg/v1/environment/4-day-weather-forecast"
    response = requests.get(forecast_url)
    forecast_data = response.json().get("items", [{}])[0].get("forecasts", [])
    future_dates = pd.date_range(start=datetime.now(), periods=7, freq="D")
    n_days = len(future_dates)
    forecast_data_padded = forecast_data + [forecast_data[-1] if forecast_data else {"temperature": {"high": 28}, "forecast": "Partly cloudy"}] * (n_days - len(forecast_data))
    temperature = [f.get("temperature", {}).get("high", 28) for f in forecast_data_padded[:n_days]]
    rainfall = [5 if "rain" in f.get("forecast", "").lower() else 0 for f in forecast_data_padded[:n_days]]

    avg_wait_time = df['Wait_Time'].mean()
    avg_satisfaction = df['Guest_Satisfaction_Score'].mean()

    df_future = pd.DataFrame({
        "date": future_dates,
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": [75] * n_days,
        "Wait_Time": [avg_wait_time] * n_days,
        "Guest_Satisfaction_Score": [avg_satisfaction] * n_days,
        "Event_Special Event": [0] * n_days,
        "Day_of_Week_" + future_dates[0].day_name(): [1] + [0] * (n_days - 1),
        "Which part of the year did you visit USS?_Q1": [1] * n_days
    })

    # Add dummy columns for optional features if they exist
    if 'Did you purchase the Express Pass?' in df.columns:
        df_future['Did you purchase the Express Pass?_Yes'] = [0] * n_days
    if 'What was the main purpose of your visit?' in df.columns:
        df_future['What was the main purpose of your visit?_Vacation'] = [1] * n_days

    # Ensure df_future has all feature columns
    for col in feature_cols:
        if col not in df_future.columns:
            df_future[col] = 0

    future_pred = model.predict(df_future[feature_cols])
    print("7-Day Demand Forecast (Visitors per Attraction):\n", pd.DataFrame({"Date": future_dates, "Predicted Visitors": future_pred}))

    # Visualize the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_pred, marker='o')
    plt.xlabel("Date")
    plt.ylabel("Predicted Visitors")
    plt.title("7-Day Demand Forecast")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return model, future_pred, future_dates

# --- Simulate Guest Flow with Predicted Demand ---
def simulate_guest_flow(attraction, predicted_demand, service_rate=0.5, layout="single_queue", duration=1440):
    env = simpy.Environment()
    wait_times = []
    resource = simpy.Resource(env, capacity=1 if layout == "single_queue" else 2)

    def guest(env, name, resource):
        arrival_time = env.now
        with resource.request() as req:
            yield req
            wait_time = env.now - arrival_time
            wait_times.append(wait_time)
            yield env.timeout(np.random.exponential(1 / service_rate))

    for i in range(int(predicted_demand)):
        env.process(guest(env, f"Guest_{i}", resource))
        env.timeout(np.random.exponential(1440 / predicted_demand)).schedule()

    env.run(until=duration)
    return wait_times

def optimize_layout(df, model):
    if df.empty or model is None:
        print("Cannot optimize layout: Merged dataset is empty or model is not trained.")
        return

    agg_dict = {
        'temperature': 'mean',
        'rainfall': 'mean',
        'humidity': 'mean',
        'Event': 'first',
        'Wait_Time': 'mean',
        'Guest_Satisfaction_Score': 'mean',
        'Day_of_Week': 'first',
        'Which part of the year did you visit USS?': 'first',
        'Attraction': 'size'
    }
    if 'Did you purchase the Express Pass?' in df.columns:
        agg_dict['Did you purchase the Express Pass?'] = 'first'
    if 'What was the main purpose of your visit?' in df.columns:
        agg_dict['What was the main purpose of your visit?'] = 'first'

    df_agg = df.groupby(['date', 'Attraction']).agg(agg_dict).rename(columns={'Attraction': 'Attraction_Visits'}).reset_index()

    dummy_cols = ['Event', 'Day_of_Week', 'Which part of the year did you visit USS?']
    if 'Did you purchase the Express Pass?' in df_agg.columns:
        dummy_cols.append('Did you purchase the Express Pass?')
    if 'What was the main purpose of your visit?' in df_agg.columns:
        dummy_cols.append('What was the main purpose of your visit?')

    df_agg = pd.get_dummies(df_agg, columns=dummy_cols, drop_first=True)

    feature_cols = ['temperature', 'rainfall', 'humidity', 'Wait_Time', 'Guest_Satisfaction_Score'] + \
                   [col for col in df_agg.columns if col.startswith('Event_') or col.startswith('Day_of_Week_') or col.startswith('Which part of the year did you visit USS?_') or col.startswith('Did you purchase the Express Pass?_') or col.startswith('What was the main purpose of your visit?_')]
    sample_day = df_agg.iloc[-1:]
    attractions = df['Attraction'].unique()
    predicted_demands = {}

    for attraction in attractions:
        sample_day['Attraction'] = attraction
        X_sample = sample_day[feature_cols]
        predicted_demand = model.predict(X_sample)[0]
        predicted_demands[attraction] = predicted_demand

    wait_times_single = {}
    wait_times_multi = {}

    for attraction, demand in predicted_demands.items():
        wait_times_single[attraction] = simulate_guest_flow(attraction, demand, service_rate=0.5, layout="single_queue", duration=1440)
        wait_times_multi[attraction] = simulate_guest_flow(attraction, demand, service_rate=0.5, layout="multi_queue", duration=1440)

    print("\nSingle Queue - Avg Wait Times per Attraction:")
    for attraction, times in wait_times_single.items():
        print(f"{attraction}: {np.mean(times):.2f} minutes")

    print("\nMulti Queue - Avg Wait Times per Attraction:")
    for attraction, times in wait_times_multi.items():
        print(f"{attraction}: {np.mean(times):.2f} minutes")

    plt.figure(figsize=(10, 6))
    for attraction in attractions:
        plt.hist(wait_times_single[attraction], bins=20, alpha=0.3, label=f"{attraction} (Single)")
        plt.hist(wait_times_multi[attraction], bins=20, alpha=0.3, label=f"{attraction} (Multi)")
    plt.xlabel("Wait Time (minutes)")
    plt.ylabel("Frequency")
    plt.title("Wait Time Distribution by Attraction and Layout")
    plt.legend()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    df_survey = load_survey_data()
    df_weather = fetch_weather_data(df_survey)
    df_merged = merge_datasets(df_weather, df_survey)
    demand_model, future_pred, future_dates = predict_demand(df_merged)
    optimize_layout(df_merged, demand_model)