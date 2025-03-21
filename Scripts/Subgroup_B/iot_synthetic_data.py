import pandas as pd
import numpy as np
from ctgan import CTGAN
from datetime import datetime, timedelta

np.random.seed(42)
n_samples = 5000  # Increased to ensure better coverage

# Define valid attractions (same as in uss_optimization.py)
valid_attractions = [
    "Revenge of the Mummy",
    "Battlestar Galactica: CYLON",
    "Transformers: The Ride",
    "Puss In Boots' Giant Journey",
    "Sesame Street Spaghetti Space Chase"
]

# Generate timestamps that align with survey data (2025-02-15 to 2025-03-03)
start_date = datetime(2025, 2, 15)
end_date = datetime(2025, 3, 3)
date_range = (end_date - start_date).days
timestamps = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_samples)]

# Convert timestamps to Unix timestamps (numerical) for CTGAN
timestamps_unix = [(t - datetime(1970, 1, 1)).total_seconds() for t in timestamps]

# Generate synthetic IoT data
data = pd.DataFrame({
    'Timestamp': timestamps_unix,  # Numerical representation for CTGAN
    'Attraction': np.random.choice(valid_attractions, size=n_samples),
    'Queue_Length': np.random.normal(loc=50, scale=15, size=n_samples).astype(int).clip(min=0),  # Average queue length ~50 people
    'Ride_Throughput': np.random.normal(loc=30, scale=5, size=n_samples).astype(int).clip(min=10),  # Throughput ~30 guests/hour
    'Visitor_Count': np.random.poisson(lam=10, size=n_samples),  # Reduced to 10 visitors per row
    'Visitor_ID': np.arange(1, n_samples + 1),
    'Step_Count': np.random.normal(loc=12000, scale=3000, size=n_samples).astype(int),
    'Transaction_Amount': np.random.normal(loc=125, scale=30, size=n_samples).round(2),
    'Check_In_Time': np.random.uniform(9, 13, n_samples).round(2),  # Check-in between 9 AM - 1 PM
    'Check_Out_Time': np.random.uniform(17, 22, n_samples).round(2),  # Check-out between 5 PM - 10 PM
    'Loyalty_Member': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7]),  # 30% are members
    'Weather_Condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy'], size=n_samples, p=[0.6, 0.3, 0.1]),
    'Age': np.random.normal(loc=35, scale=10, size=n_samples).astype(int),  # Avg visitor age ~35
    'Gender': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.5, 0.5]),
    'Guest_Satisfaction_Score': np.random.uniform(1, 5, n_samples).round(1)  # 1-5 rating
})

# Define categorical (discrete) columns
discrete_columns = ['Attraction', 'Loyalty_Member', 'Weather_Condition', 'Gender']

# Train a CTGAN model
model = CTGAN(epochs=500)  # Increase epochs for better learning
model.fit(data, discrete_columns=discrete_columns)

# Generate synthetic data
synthetic_data = model.sample(n_samples)

# Convert Unix timestamps back to datetime
synthetic_data['Timestamp'] = pd.to_datetime(synthetic_data['Timestamp'], unit='s')

# Clip timestamps to the desired range (2025-02-15 to 2025-03-03)
start_timestamp = pd.Timestamp(start_date)
end_timestamp = pd.Timestamp(end_date)
synthetic_data['Timestamp'] = synthetic_data['Timestamp'].clip(lower=start_timestamp, upper=end_timestamp)

# Ensure Visitor_ID is unique
synthetic_data['Visitor_ID'] = np.arange(1, n_samples + 1)

# Show sample synthetic data
print(synthetic_data.head())

# Save synthetic data to CSV
synthetic_data.to_csv("synthetic_theme_park_data.csv", index=False)