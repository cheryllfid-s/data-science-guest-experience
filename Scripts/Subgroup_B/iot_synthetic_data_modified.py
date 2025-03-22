import os
os.chdir("C:/Users/parma/data-science-guest-experience/Scripts/Subgroup_B")
print(os.path.exists("../../data/"))

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from faker import Faker
from ctgan import CTGAN

# Initialize Faker instance
fake = Faker()

# Weather API endpoints
ENDPOINTS = {
    "temperature": "https://api.data.gov.sg/v1/environment/air-temperature",
    "rainfall": "https://api.data.gov.sg/v1/environment/rainfall",
    "humidity": "https://api.data.gov.sg/v1/environment/relative-humidity"
}

# List of attractions and their popularity
POPULAR_ATTRACTIONS = [
    "Revenge of the Mummy", "Battlestar Galactica: CYLON", "Transformers: The Ride"
]
OTHER_ATTRACTIONS = [
    "Puss In Boots' Giant Journey", "Sesame Street Spaghetti Space Chase"
]
ALL_ATTRACTIONS = POPULAR_ATTRACTIONS + OTHER_ATTRACTIONS

# Function to fetch weather data
def fetch_weather_data(date):
    params = {"date": date}
    weather_data = {}
    
    for key, url in ENDPOINTS.items():
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "items" in data and data["items"]:
                latest_reading = data["items"][0]["readings"]
                weather_data[key] = np.mean([r["value"] for r in latest_reading])
        else:
            print(f"Failed to fetch {key} data for {date}")
            weather_data[key] = None
    
    return weather_data

# Function to generate synthetic data using Faker and CTGAN
def generate_synthetic_data(start_date, end_date, num_samples=5000):
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    data = []
    np.random.seed(42)
    
    for date in dates:
        weather_info = fetch_weather_data(date.strftime("%Y-%m-%d"))
        numeric_timestamp = int(date.timestamp())
        
        for _ in range(num_samples // len(dates)):
            attraction = np.random.choice(ALL_ATTRACTIONS)
            age = np.random.choice(["Child", "Teen", "Young Adult", "Adult", "Senior"], p=[0.1, 0.3, 0.4, 0.15, 0.05])
            gender = np.random.choice(["Male", "Female"], p=[0.5, 0.5])
            loyalty_member = np.random.choice(["Yes", "No"], p=[0.2, 0.8])
            check_in_time = np.random.randint(9, 15)
            check_out_time = check_in_time + np.random.randint(3, 6)
            satisfaction = round(np.random.uniform(1, 5), 2)
            transaction_amount = np.random.randint(62, 85) if age == "Child" else np.random.randint(85, 150)
            transaction_amount += np.random.randint(15, 40) * np.random.randint(1, 3)  # Food
            transaction_amount += np.random.randint(20, 90)  # Merchandise
            
            step_count = np.random.randint(12000, 18000) if age in ["Child", "Teen"] else np.random.randint(8000, 14000)
            
            base_wait = np.random.randint(45, 90) if attraction in POPULAR_ATTRACTIONS else np.random.randint(30, 60)
            
            if date.month in [10, 12] or date.day_name() in ["Saturday", "Sunday"]:
                base_wait += 15
            
            if weather_info.get("rainfall", 0) > 5:
                base_wait = max(base_wait - 30, 15)
            
            queue_count = np.random.randint(50, 300) if attraction in POPULAR_ATTRACTIONS else np.random.randint(20, 100)
            
            data.append({
                "Timestamp": numeric_timestamp,
                "Attraction": attraction,
                "Age_Group": age,
                "Gender": gender,
                "Loyalty_Member": loyalty_member,
                "Check_In_Time": check_in_time,
                "Check_Out_Time": check_out_time,
                "Step_Count": step_count,
                "Transaction_Amount": transaction_amount,
                "Guest_Satisfaction_Score": satisfaction,
                "Average_Queue_Time": base_wait,
                "Number_of_People_in_Queue": queue_count,
                "Temperature": weather_info.get("temperature"),
                "Rainfall": weather_info.get("rainfall"),
                "Humidity": weather_info.get("humidity")
            })
    
    df = pd.DataFrame(data)
    
    # Train CTGAN model to generate more realistic synthetic data
    ctgan = CTGAN(epochs=300)
    ctgan.fit(df, discrete_columns=["Attraction", "Age_Group", "Gender", "Loyalty_Member"])
    synthetic_data = ctgan.sample(num_samples)
    
    # Convert Timestamp back to datetime
    synthetic_data["Timestamp"] = pd.to_datetime(synthetic_data["Timestamp"], unit='s')
    
    return synthetic_data

# Generate synthetic data for June to December 2024
data_df = generate_synthetic_data("2024-06-01", "2024-12-31", num_samples=5000)
print(data_df.head())

# Save to CSV
data_df.to_csv("../../data/synthetic_iot_data.csv", index=False)
