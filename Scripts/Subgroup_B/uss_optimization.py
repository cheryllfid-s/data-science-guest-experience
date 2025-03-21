import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta
import os

# --- Load Survey Data ---
def load_survey_data(file_path="survey.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please provide the survey dataset.")

    df = pd.read_csv(file_path)

    print("Columns in survey.csv:", df.columns.tolist())

    df = df.rename(columns={
        "On a scale of 1-5, how would you rate your overall experience at USS?": "Guest_Satisfaction_Score",
        "How long did you wait in line for rides on average during your visit?": "Wait_Time",
        "Timestamp": "Timestamp",
        "Which ride or attraction was your favourite?": "Attraction"
    })

    wait_time_mapping = {
        "Less than 15 mins": 10,
        "15-30 mins": 22.5,
        "30-45 mins": 37.5,
        "45 mins - 1 hr": 52.5,
        "More than 1 hr": 75
    }
    df["Wait_Time"] = df["Wait_Time"].map(wait_time_mapping).fillna(37.5)

    np.random.seed(42)
    df["Event"] = np.random.choice(['None', 'Special Event'], size=len(df), p=[0.8, 0.2])

    required_columns_for_long_wait = [
        'Attraction', 'Timestamp', 'Event', 'Wait_Time', 'Guest_Satisfaction_Score',
        'Which part of the year did you visit USS?', 'Did you purchase the Express Pass?',
        'What was the main purpose of your visit?', 'Who did you visit USS with?',
        'Which age group do you belong to?'
    ]

    if 'Did you experience any rides with longer-than-expected wait times? If yes, which ride(s)?' not in df.columns:
        print("Warning: Long wait time column missing. Skipping wait time enhancement.")
        long_wait_df = pd.DataFrame(columns=required_columns_for_long_wait)
    else:
        long_wait_rides = df['Did you experience any rides with longer-than-expected wait times? If yes, which ride(s)?'].str.split(', ', expand=True).stack().reset_index()
        long_wait_rides.columns = ['original_index', 'split_index', 'Attraction']
        long_wait_rides = long_wait_rides[long_wait_rides['Attraction'].notna()]

        queue_worth_col = 'Did you feel that overall, the queuing time was worth the experience of the attraction? '
        unpleasant_col = 'What made your experience with this ride or attraction unpleasant? '

        wait_time_adjusted = []
        for idx in long_wait_rides['original_index']:
            if queue_worth_col in df.columns:
                queue_worth = df[queue_worth_col].iloc[idx]
                base_wait = 90 if queue_worth == 'No' else 75
            else:
                base_wait = 75

            if unpleasant_col in df.columns and pd.notna(df[unpleasant_col].iloc[idx]):
                unpleasant_reason = df[unpleasant_col].iloc[idx]
                if 'long wait' in str(unpleasant_reason).lower():
                    base_wait += 15
            wait_time_adjusted.append(base_wait)

        long_wait_df = pd.DataFrame({
            'Attraction': long_wait_rides['Attraction'],
            'Timestamp': df['Timestamp'].iloc[long_wait_rides['original_index']].values,
            'Event': df['Event'].iloc[long_wait_rides['original_index']].values,
            'Wait_Time': wait_time_adjusted,
            'Guest_Satisfaction_Score': df['Guest_Satisfaction_Score'].iloc[long_wait_rides['original_index']].values,
            'Which part of the year did you visit USS?': df['Which part of the year did you visit USS?'].iloc[long_wait_rides['original_index']].values,
            'Did you purchase the Express Pass?': df['Did you purchase the Express Pass?'].iloc[long_wait_rides['original_index']].values if 'Did you purchase the Express Pass?' in df.columns else [None] * len(long_wait_rides),
            'What was the main purpose of your visit?': df['What was the main purpose of your visit?'].iloc[long_wait_rides['original_index']].values if 'What was the main purpose of your visit?' in df.columns else [None] * len(long_wait_rides),
            'Who did you visit USS with?': df['Who did you visit USS with?'].iloc[long_wait_rides['original_index']].values if 'Who did you visit USS with?' in df.columns else [None] * len(long_wait_rides),
            'Which age group do you belong to?': df['Which age group do you belong to?'].iloc[long_wait_rides['original_index']].values if 'Which age group do you belong to?' in df.columns else [None] * len(long_wait_rides)
        })

    df = pd.concat([df[['Attraction', 'Wait_Time', 'Timestamp', 'Event', 'Guest_Satisfaction_Score', 'Which part of the year did you visit USS?', 'Did you purchase the Express Pass?', 'What was the main purpose of your visit?', 'Who did you visit USS with?', 'Which age group do you belong to?']], long_wait_df], ignore_index=True)

    print("Wait Time Distribution:\n", df["Wait_Time"].value_counts())

    print("Guest Satisfaction Score Distribution (before normalization):\n", df["Guest_Satisfaction_Score"].value_counts())
    df["Guest_Satisfaction_Score"] = pd.to_numeric(df["Guest_Satisfaction_Score"], errors="coerce")
    df["Guest_Satisfaction_Score"] = (df["Guest_Satisfaction_Score"] - df["Guest_Satisfaction_Score"].min()) / (df["Guest_Satisfaction_Score"].max() - df["Guest_Satisfaction_Score"].min())

    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.date

    df['Day_of_Week'] = pd.to_datetime(df['Timestamp']).dt.day_name()
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

    valid_attractions = [
        "Revenge of the Mummy",
        "Battlestar Galactica: CYLON",
        "Transformers: The Ride",
        "Puss In Boots' Giant Journey",
        "Sesame Street Spaghetti Space Chase"
    ]

    df = df[df['Attraction'].isin(valid_attractions)]
    if df.empty:
        raise ValueError("No valid attractions found in the survey data after filtering.")

    required_columns = ["Attraction", "Wait_Time", "Timestamp", "Event", "Guest_Satisfaction_Score", "Day_of_Week", "Is_Weekend", "Which part of the year did you visit USS?"]
    if 'Did you purchase the Express Pass?' in df.columns:
        required_columns.append('Did you purchase the Express Pass?')
    if 'What was the main purpose of your visit?' in df.columns:
        required_columns.append('What was the main purpose of your visit?')
    if 'Who did you visit USS with?' in df.columns:
        required_columns.append('Who did you visit USS with?')
    if 'Which age group do you belong to?' in df.columns:
        required_columns.append('Which age group do you belong to?')

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    print("Survey Data - First 5 records:\n", df[required_columns].head())
    return df

# --- Load IoT Data ---
def load_iot_data(file_path="iot_data.csv"):
    if not os.path.exists(file_path):
        print(f"Warning: IoT data file {file_path} not found. Skipping IoT data integration.")
        return None

    df_iot = pd.read_csv(file_path)
    print("Columns in iot_data.csv:", df_iot.columns.tolist())

    required_iot_columns = ['Timestamp', 'Attraction', 'Queue_Length', 'Ride_Throughput', 'Visitor_Count']
    missing_iot_cols = [col for col in required_iot_columns if col not in df_iot.columns]
    if missing_iot_cols:
        print(f"Warning: IoT data missing required columns: {missing_iot_cols}. Skipping IoT data integration.")
        return None

    df_iot['Timestamp'] = pd.to_datetime(df_iot['Timestamp']).dt.date
    return df_iot

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
def merge_datasets(df_weather, df_survey, df_iot=None):
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date
    df_survey["date"] = pd.to_datetime(df_survey["Timestamp"]).dt.date

    df_merged = pd.merge(df_survey, df_weather, on="date", how="inner")

    if df_iot is not None:
        df_iot["date"] = pd.to_datetime(df_iot["Timestamp"]).dt.date
        df_iot = df_iot.groupby(['date', 'Attraction']).agg({
            'Queue_Length': 'mean',
            'Ride_Throughput': 'mean',
            'Visitor_Count': 'sum'
        }).reset_index()

        # Estimate Wait_Time from Queue_Length and Ride_Throughput
        df_iot['Wait_Time_IoT'] = (df_iot['Queue_Length'] / df_iot['Ride_Throughput']) * 60  # Convert to minutes

        df_merged = pd.merge(df_merged, df_iot, on=['date', 'Attraction'], how="left")

        # Update Wait_Time with IoT data where available
        df_merged['Wait_Time'] = df_merged['Wait_Time_IoT'].combine_first(df_merged['Wait_Time'])
        df_merged = df_merged.drop(columns=['Wait_Time_IoT'])

    print("Merged DataFrame columns:", df_merged.columns.tolist())
    return df_merged

# --- Predict Demand ---
def predict_demand(df):
    if df.empty:
        print("Cannot predict demand: Merged dataset is empty.")
        return None, None, None

    df['Week'] = pd.to_datetime(df['Timestamp']).dt.isocalendar().week
    agg_dict = {
        'temperature': 'mean',
        'rainfall': 'mean',
        'humidity': 'mean',
        'Event': 'first',
        'Wait_Time': 'mean',
        'Guest_Satisfaction_Score': 'mean',
        'Day_of_Week': 'first',
        'Is_Weekend': 'max',
        'Which part of the year did you visit USS?': 'first',
        'Attraction': 'size'
    }
    if 'Did you purchase the Express Pass?' in df.columns:
        agg_dict['Did you purchase the Express Pass?'] = 'first'
    if 'What was the main purpose of your visit?' in df.columns:
        agg_dict['What was the main purpose of your visit?'] = 'first'
    if 'Who did you visit USS with?' in df.columns:
        agg_dict['Who did you visit USS with?'] = 'first'
    if 'Which age group do you belong to?' in df.columns:
        agg_dict['Which age group do you belong to?'] = 'first'
    if 'Visitor_Count' in df.columns:
        agg_dict['Visitor_Count'] = 'sum'
    if 'Queue_Length' in df.columns:
        agg_dict['Queue_Length'] = 'mean'
    if 'Ride_Throughput' in df.columns:
        agg_dict['Ride_Throughput'] = 'mean'

    df_agg = df.groupby(['Week', 'Attraction']).agg(agg_dict).rename(columns={'Attraction': 'Attraction_Visits'}).reset_index()

    df_agg['Wait_Time_Satisfaction'] = df_agg['Wait_Time'] * df_agg['Guest_Satisfaction_Score']

    dummy_cols = ['Event', 'Day_of_Week', 'Which part of the year did you visit USS?']
    if 'Did you purchase the Express Pass?' in df_agg.columns:
        dummy_cols.append('Did you purchase the Express Pass?')
    if 'What was the main purpose of your visit?' in df_agg.columns:
        dummy_cols.append('What was the main purpose of your visit?')
    if 'Who did you visit USS with?' in df_agg.columns:
        dummy_cols.append('Who did you visit USS with?')
    if 'Which age group do you belong to?' in df_agg.columns:
        dummy_cols.append('Which age group do you belong to?')

    df_agg = pd.get_dummies(df_agg, columns=dummy_cols, drop_first=True)

    feature_cols = ['temperature', 'rainfall', 'humidity', 'Wait_Time', 'Guest_Satisfaction_Score', 'Wait_Time_Satisfaction', 'Is_Weekend']
    if 'Queue_Length' in df_agg.columns:
        feature_cols.append('Queue_Length')
    if 'Ride_Throughput' in df_agg.columns:
        feature_cols.append('Ride_Throughput')
    feature_cols += [col for col in df_agg.columns if col.startswith('Event_') or col.startswith('Day_of_Week_') or col.startswith('Which part of the year did you visit USS?_') or col.startswith('Did you purchase the Express Pass?_') or col.startswith('What was the main purpose of your visit?_') or col.startswith('Who did you visit USS with?_') or col.startswith('Which age group do you belong to?_')]

    low_importance_features = ['Day_of_Week_Tuesday', 'Day_of_Week_Saturday']
    feature_cols = [col for col in feature_cols if col not in low_importance_features]

    X = df_agg[feature_cols]
    y = df_agg['Visitor_Count'] if 'Visitor_Count' in df_agg.columns else df_agg['Attraction_Visits']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = XGBRegressor(n_estimators=50, min_child_weight=3, reg_lambda=1.0, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    final_pred = 0.7 * xgb_pred + 0.3 * ridge_pred

    rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    r2 = r2_score(y_test, final_pred)
    mae = mean_absolute_error(y_test, final_pred)
    print(f"Demand Prediction - RMSE: {rmse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")

    scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
    print(f"Cross-Validated R² (XGBoost): {scores.mean():.2f} ± {scores.std():.2f}")

    feature_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Feature Importance:\n", feature_importance.head(10))

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
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": [75] * n_days,
        "Wait_Time": [avg_wait_time] * n_days,
        "Guest_Satisfaction_Score": [avg_satisfaction] * n_days,
        "Wait_Time_Satisfaction": [avg_wait_time * avg_satisfaction] * n_days,
        "Is_Weekend": [1 if d.day_name() in ['Saturday', 'Sunday'] else 0 for d in future_dates],
        "Event_Special Event": [0] * n_days,
        "Day_of_Week_" + future_dates[0].day_name(): [1] + [0] * (n_days - 1),
        "Which part of the year did you visit USS?_Q1": [1] * n_days
    })

    if 'Queue_Length' in df.columns:
        df_future['Queue_Length'] = [df['Queue_Length'].mean()] * n_days
    if 'Ride_Throughput' in df.columns:
        df_future['Ride_Throughput'] = [df['Ride_Throughput'].mean()] * n_days
    if 'Did you purchase the Express Pass?' in df.columns:
        df_future['Did you purchase the Express Pass?_Yes'] = [0] * n_days
    if 'What was the main purpose of your visit?' in df.columns:
        df_future['What was the main purpose of your visit?_Vacation'] = [1] * n_days
    if 'Who did you visit USS with?' in df.columns:
        df_future['Who did you visit USS with?_Family'] = [1] * n_days
    if 'Which age group do you belong to?' in df.columns:
        df_future['Which age group do you belong to?_26-35'] = [1] * n_days

    for col in feature_cols:
        if col not in df_future.columns:
            df_future[col] = 0

    xgb_future_pred = xgb_model.predict(df_future[feature_cols])
    ridge_future_pred = ridge_model.predict(df_future[feature_cols])
    future_pred = 0.7 * xgb_future_pred + 0.3 * ridge_future_pred

    print("7-Day Demand Forecast (Visitors per Attraction):\n", pd.DataFrame({"Date": future_dates, "Predicted Visitors": future_pred}))

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_pred, marker='o')
    plt.xlabel("Date")
    plt.ylabel("Predicted Visitors")
    plt.title("7-Day Demand Forecast")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return xgb_model, future_pred, future_dates

# --- Simulate Guest Flow with Predicted Demand ---
def simulate_guest_flow(attraction, predicted_demand, service_rate=0.5, layout="single_queue", duration=1440, ride_throughput=None):
    env = simpy.Environment()
    wait_times = []
    resource = simpy.Resource(env, capacity=1 if layout == "single_queue" else 2)

    # Use IoT ride throughput if available, otherwise use default service rate
    effective_service_rate = (ride_throughput / 60) if ride_throughput is not None else service_rate  # Convert throughput (guests/hour) to guests/minute

    def guest(env, name, resource):
        arrival_time = env.now
        with resource.request() as req:
            yield req
            wait_time = env.now - arrival_time
            wait_times.append(wait_time)
            yield env.timeout(np.random.exponential(1 / effective_service_rate))

    for i in range(int(predicted_demand)):
        env.process(guest(env, f"Guest_{i}", resource))
        env.timeout(np.random.exponential(1440 / predicted_demand)).schedule()

    env.run(until=duration)
    return wait_times

def optimize_layout(df, model):
    if df.empty or model is None:
        print("Cannot optimize layout: Merged dataset is empty or model is not trained.")
        return

    df['Week'] = pd.to_datetime(df['Timestamp']).dt.isocalendar().week
    agg_dict = {
        'temperature': 'mean',
        'rainfall': 'mean',
        'humidity': 'mean',
        'Event': 'first',
        'Wait_Time': 'mean',
        'Guest_Satisfaction_Score': 'mean',
        'Day_of_Week': 'first',
        'Is_Weekend': 'max',
        'Which part of the year did you visit USS?': 'first',
        'Attraction': 'size'
    }
    if 'Did you purchase the Express Pass?' in df.columns:
        agg_dict['Did you purchase the Express Pass?'] = 'first'
    if 'What was the main purpose of your visit?' in df.columns:
        agg_dict['What was the main purpose of your visit?'] = 'first'
    if 'Who did you visit USS with?' in df.columns:
        agg_dict['Who did you visit USS with?'] = 'first'
    if 'Which age group do you belong to?' in df.columns:
        agg_dict['Which age group do you belong to?'] = 'first'
    if 'Ride_Throughput' in df.columns:
        agg_dict['Ride_Throughput'] = 'mean'

    df_agg = df.groupby(['Week', 'Attraction']).agg(agg_dict).rename(columns={'Attraction': 'Attraction_Visits'}).reset_index()

    df_agg['Wait_Time_Satisfaction'] = df_agg['Wait_Time'] * df_agg['Guest_Satisfaction_Score']

    dummy_cols = ['Event', 'Day_of_Week', 'Which part of the year did you visit USS?']
    if 'Did you purchase the Express Pass?' in df_agg.columns:
        dummy_cols.append('Did you purchase the Express Pass?')
    if 'What was the main purpose of your visit?' in df_agg.columns:
        dummy_cols.append('What was the main purpose of your visit?')
    if 'Who did you visit USS with?' in df_agg.columns:
        dummy_cols.append('Who did you visit USS with?')
    if 'Which age group do you belong to?' in df_agg.columns:
        dummy_cols.append('Which age group do you belong to?')

    df_agg = pd.get_dummies(df_agg, columns=dummy_cols, drop_first=True)

    feature_cols = ['temperature', 'rainfall', 'humidity', 'Wait_Time', 'Guest_Satisfaction_Score', 'Wait_Time_Satisfaction', 'Is_Weekend']
    if 'Queue_Length' in df_agg.columns:
        feature_cols.append('Queue_Length')
    if 'Ride_Throughput' in df_agg.columns:
        feature_cols.append('Ride_Throughput')
    feature_cols += [col for col in df_agg.columns if col.startswith('Event_') or col.startswith('Day_of_Week_') or col.startswith('Which part of the year did you visit USS?_') or col.startswith('Did you purchase the Express Pass?_') or col.startswith('What was the main purpose of your visit?_') or col.startswith('Who did you visit USS with?_') or col.startswith('Which age group do you belong to?_')]

    low_importance_features = ['Day_of_Week_Tuesday', 'Day_of_Week_Saturday']
    feature_cols = [col for col in feature_cols if col not in low_importance_features]

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
        ride_throughput = sample_day['Ride_Throughput'].iloc[0] if 'Ride_Throughput' in sample_day.columns else None
        wait_times_single[attraction] = simulate_guest_flow(attraction, demand, service_rate=0.5, layout="single_queue", duration=1440, ride_throughput=ride_throughput)
        wait_times_multi[attraction] = simulate_guest_flow(attraction, demand, service_rate=0.5, layout="multi_queue", duration=1440, ride_throughput=ride_throughput)

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
    df_iot = load_iot_data()
    df_weather = fetch_weather_data(df_survey)
    df_merged = merge_datasets(df_weather, df_survey, df_iot)
    demand_model, future_pred, future_dates = predict_demand(df_merged)
    optimize_layout(df_merged, demand_model)