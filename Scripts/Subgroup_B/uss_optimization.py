"""
USS Demand Prediction Script for Subgroup B, Question 1

This script predicts demand for attractions at Universal Studios Singapore (USS) over the next 7 days using a hybrid XGBoost + Ridge model. It incorporates survey data from survey.csv, synthetic IoT data from synthetic_theme_park_data.csv, weather data, and synthetic event data.

Dependencies:
    - pandas
    - numpy
    - requests
    - sklearn
    - xgboost
    - matplotlib
    - simpy

Install dependencies using:
    pip install pandas numpy requests sklearn xgboost matplotlib simpy
"""

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta
import os
os.chdir("C:/Users/parma/data-science-guest-experience/Scripts/Subgroup_B")

# --- Load Survey Data ---
def load_survey_data(file_path="../../data/survey.csv"):
    """
    Loads and preprocesses survey data from survey.csv for demand prediction.

    Args:
        file_path (str): Path to the survey CSV file.

    Returns:
        pd.DataFrame: Preprocessed survey data with required columns.
    """
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

    # Synthetic event data (no external event API)
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
def load_iot_data(file_path="../../data/synthetic_theme_park_data.csv"):
    """
    Loads synthetic IoT data for demand prediction.

    Args:
        file_path (str): Path to the synthetic IoT CSV file (default: synthetic_theme_park_data.csv).

    Returns:
        pd.DataFrame: Preprocessed IoT data, or None if the file is missing or invalid.
    """
    if not os.path.exists(file_path):
        print(f"Warning: IoT data file {file_path} not found. Skipping IoT data integration.")
        return None

    df_iot = pd.read_csv(file_path)
    print("Columns in synthetic_theme_park_data.csv:", df_iot.columns.tolist())

    required_iot_columns = ['Timestamp', 'Attraction', 'Queue_Length', 'Ride_Throughput', 'Visitor_Count']
    missing_iot_cols = [col for col in required_iot_columns if col not in df_iot.columns]
    if missing_iot_cols:
        print(f"Warning: IoT data missing required columns: {missing_iot_cols}. Skipping IoT data integration.")
        return None

    df_iot['Timestamp'] = pd.to_datetime(df_iot['Timestamp']).dt.date
    return df_iot

# --- Load Weather Data ---
def fetch_weather_data(df_survey, cache_file="weather_data.csv"):
    """
    Fetches weather data for survey dates, either from cache or API.

    Args:
        df_survey (pd.DataFrame): Survey data with timestamps.
        cache_file (str): Path to cache weather data.

    Returns:
        pd.DataFrame: Weather data with date, temperature, rainfall, and humidity.
    """
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
    """
    Merges survey, weather, and IoT datasets on date and attraction.

    Args:
        df_weather (pd.DataFrame): Weather data.
        df_survey (pd.DataFrame): Survey data.
        df_iot (pd.DataFrame, optional): IoT data.

    Returns:
        pd.DataFrame: Merged dataset.
    """
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

        df_iot['Wait_Time_IoT'] = (df_iot['Queue_Length'] / df_iot['Ride_Throughput']) * 60
        df_merged = pd.merge(df_merged, df_iot, on=['date', 'Attraction'], how="left")
        df_merged['Wait_Time'] = df_merged['Wait_Time_IoT'].combine_first(df_merged['Wait_Time'])
        df_merged = df_merged.drop(columns=['Wait_Time_IoT'])

    print("Merged DataFrame columns:", df_merged.columns.tolist())
    return df_merged

# --- Predict Demand ---
def predict_demand(df):
    """
    Predicts demand for attractions using a hybrid XGBoost + Ridge model.

    Args:
        df (pd.DataFrame): Merged dataset with survey, weather, and IoT data.

    Returns:
        tuple: (model, future_pred, future_dates, feature_cols, scaler, imputer) where model is the trained XGBoost model,
               future_pred is the 7-day demand forecast, future_dates are the forecast dates, feature_cols are the features used,
               scaler is the fitted scaler, and imputer is the fitted imputer.
    """
    if df.empty:
        print("Cannot predict demand: Merged dataset is empty.")
        return None, None, None, None, None, None

    # Group by date and Attraction to increase sample size
    df['date'] = pd.to_datetime(df['Timestamp']).dt.date
    agg_dict = {
        'temperature': 'mean',
        'rainfall': 'mean',
        'humidity': 'mean',
        'Event': 'first',
        'Wait_Time': 'mean',
        'Guest_Satisfaction_Score': 'mean',
        'Day_of_Week': 'first',
        'Is_Weekend': 'max',
        'Which part of the year did you visit USS?': 'first'
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

    # Add a count column to use as the target when Visitor_Count is not available
    df_agg = df.groupby(['date', 'Attraction']).agg(agg_dict).reset_index()
    df_agg['Survey_Response_Count'] = df.groupby(['date', 'Attraction']).size().values
    df_agg['Wait_Time_Satisfaction'] = df_agg['Wait_Time'] * df_agg['Guest_Satisfaction_Score']

    print("Aggregated DataFrame shape:", df_agg.shape)
    print("Aggregated DataFrame:\n", df_agg)

    if len(df_agg) < 2:
        print("Error: Aggregated dataset has fewer than 2 samples. Cannot train model.")
        return None, None, None, None, None, None

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
    y = df_agg['Visitor_Count'] if 'Visitor_Count' in df_agg.columns else df_agg['Survey_Response_Count']

    # Check for NaN values in X
    nan_columns = X.columns[X.isna().any()].tolist()
    if nan_columns:
        print(f"Columns with NaN values in X: {nan_columns}")
        for col in nan_columns:
            print(f"Number of NaN values in {col}: {X[col].isna().sum()}")
        # Drop columns that are entirely NaN
        entirely_nan_columns = X.columns[X.isna().all()].tolist()
        if entirely_nan_columns:
            print(f"Dropping columns that are entirely NaN: {entirely_nan_columns}")
            X = X.drop(columns=entirely_nan_columns)
            feature_cols = [col for col in feature_cols if col not in entirely_nan_columns]
    else:
        print("No NaN values found in X.")

    # Impute remaining NaN values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split data, ensuring there's enough data for training and testing
    if len(X_imputed) < 4:
        print("Warning: Dataset too small for train-test split. Using all data for training.")
        X_train, X_test, y_train, y_test = X_imputed, X_imputed, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Increase regularization to reduce overfitting
    xgb_model = XGBRegressor(n_estimators=50, min_child_weight=5, reg_lambda=2.0, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)

    final_pred = 0.7 * xgb_pred + 0.3 * ridge_pred

    rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    r2 = r2_score(y_test, final_pred)
    mae = mean_absolute_error(y_test, final_pred)
    print(f"Demand Prediction - RMSE: {rmse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")

    # Cross-validation for XGBoost
    if len(X_imputed) >= 3:
        cv_folds = min(3, len(X_imputed))  # Use fewer folds if dataset is small
        X_scaled = scaler.transform(X_imputed)
        scores = cross_val_score(xgb_model, X_scaled, y, cv=cv_folds, scoring='r2')
        print(f"Cross-Validated R² (XGBoost, cv={cv_folds}): {scores.mean():.2f} ± {scores.std():.2f}")
    else:
        print("Skipping cross-validation: Dataset too small (less than 3 samples).")

    # Cross-validation for the hybrid model using KFold
    if len(X_imputed) >= 3:
        cv_folds = min(3, len(X_imputed))
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        hybrid_r2_scores = []
        for train_idx, test_idx in kf.split(X_imputed):
            X_train_fold, X_test_fold = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_test_fold_scaled = scaler.transform(X_test_fold)
            xgb_model.fit(X_train_fold_scaled, y_train_fold)
            ridge_model.fit(X_train_fold_scaled, y_train_fold)
            xgb_pred_fold = xgb_model.predict(X_test_fold_scaled)
            ridge_pred_fold = ridge_model.predict(X_test_fold_scaled)
            fold_pred = 0.7 * xgb_pred_fold + 0.3 * ridge_pred_fold
            fold_r2 = r2_score(y_test_fold, fold_pred)
            hybrid_r2_scores.append(fold_r2)
        print(f"Cross-Validated R² (Hybrid Model, cv={cv_folds}): {np.mean(hybrid_r2_scores):.2f} ± {np.std(hybrid_r2_scores):.2f}")
    else:
        print("Skipping hybrid model cross-validation: Dataset too small (less than 3 samples).")

    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, final_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Visitors")
    plt.ylabel("Predicted Visitors")
    plt.title("Actual vs Predicted Visitors (Test Set)")
    plt.tight_layout()
    plt.show()

    feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importance:\n", feature_importance.head(10))

    # Fetch weather forecast for the next 7 days
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
        "Event_Special Event": [0] * n_days,  # Assuming no special events for the forecast
        "Day_of_Week_" + future_dates[0].day_name(): [1] + [0] * (n_days - 1),
        "Which part of the year did you visit USS?_Q1": [1] * n_days
    })

    if 'Queue_Length' in df.columns and 'Queue_Length' in feature_cols:
        df_future['Queue_Length'] = [df['Queue_Length'].mean()] * n_days
    if 'Ride_Throughput' in df.columns and 'Ride_Throughput' in feature_cols:
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

    # Impute NaN values in df_future and scale
    df_future_imputed = pd.DataFrame(imputer.transform(df_future[feature_cols]), columns=feature_cols)
    df_future_scaled = scaler.transform(df_future_imputed)

    xgb_future_pred = xgb_model.predict(df_future_scaled)
    ridge_future_pred = ridge_model.predict(df_future_scaled)
    future_pred = 0.7 * xgb_future_pred + 0.3 * ridge_future_pred

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Visitors": future_pred})
    print("7-Day Demand Forecast (Visitors per Attraction):\n", forecast_df)

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_pred, marker='o')
    plt.xlabel("Date")
    plt.ylabel("Predicted Visitors")
    plt.title("7-Day Demand Forecast")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return xgb_model, future_pred, future_dates, feature_cols, scaler, imputer

# --- Simulate Guest Flow with Predicted Demand ---
def simulate_guest_flow(attraction, predicted_demand, service_rate=0.5, layout="single_queue", duration=1440, ride_throughput=None):
    """
    Simulates guest flow for an attraction to estimate wait times.

    Args:
        attraction (str): Name of the attraction.
        predicted_demand (float): Predicted number of visitors.
        service_rate (float): Default service rate (guests per minute).
        layout (str): Queue layout ('single_queue' or 'multi_queue').
        duration (int): Simulation duration in minutes.
        ride_throughput (float): Ride throughput from IoT data (guests per hour).

    Returns:
        list: Wait times for each guest.
    """
    env = simpy.Environment()
    wait_times = []
    resource = simpy.Resource(env, capacity=1 if layout == "single_queue" else 2)

    effective_service_rate = (ride_throughput / 60) if ride_throughput is not None else service_rate

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

# --- Optimize Layout ---
def optimize_layout(df, model, feature_cols, scaler, imputer):
    """
    Optimizes queue layouts by simulating guest flow with predicted demand.

    Args:
        df (pd.DataFrame): Merged dataset.
        model: Trained demand prediction model.
        feature_cols (list): List of feature columns used in predict_demand.
        scaler: Fitted StandardScaler from predict_demand.
        imputer: Fitted SimpleImputer from predict_demand.
    """
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
        'Is_Weekend': 'max',
        'Which part of the year did you visit USS?': 'first'
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

    df_agg = df.groupby(['date', 'Attraction']).agg(agg_dict).reset_index()
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

    # Ensure all feature_cols are present in df_agg
    for col in feature_cols:
        if col not in df_agg.columns:
            df_agg[col] = 0

    sample_day = df_agg.iloc[-1:].copy()  # Create a copy to avoid SettingWithCopyWarning
    attractions = df['Attraction'].unique()
    predicted_demands = {}

    for attraction in attractions:
        sample_day.loc[sample_day.index, 'Attraction'] = attraction
        X_sample = sample_day[feature_cols]
        X_sample_imputed = pd.DataFrame(imputer.transform(X_sample), columns=feature_cols)
        X_sample_scaled = scaler.transform(X_sample_imputed)
        predicted_demand = model.predict(X_sample_scaled)[0]
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
    print(df_survey.head())
    df_iot = load_iot_data()
    df_weather = fetch_weather_data(df_survey)
    df_merged = merge_datasets(df_weather, df_survey, df_iot)
    demand_model, future_pred, future_dates, feature_cols, scaler, imputer = predict_demand(df_merged)
    optimize_layout(df_merged, demand_model, feature_cols, scaler, imputer)