import os
import pandas as pd
import numpy as np
from datetime import datetime
from faker import Faker

# Change the working directory
new_path = r"/Users/derr/Documents/DSA3101/Project/DSA3101 data-science-guest-experience/data-science-guest-experience/Scripts/Subgroup_B"
os.chdir(new_path)

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Initialize Faker
fake = Faker()

# Define theme zones and attractions
THEME_ZONES = [
    "Hollywood", "New York", "Sci-Fi City", "Ancient Egypt", "The Lost World", "Far Far Away"
]

ATTRACTIONS = {
    "Ancient Egypt": ["Revenge of the Mummy"],
    "Sci-Fi City": ["Battlestar Galactica: CYLON", "Transformers: The Ride", "Battlestar Galactica: HUMAN"],
    "New York": ["Sesame Street Spaghetti Space Chase"],
    "Hollywood": [],
    "The Lost World": ["Canopy Flyer"],
    "Far Far Away": ["Puss In Boots' Giant Journey"]
}

POPULAR_ATTRACTIONS = sum(ATTRACTIONS.values(), [])

# Generate synthetic IoT data
def generate_synthetic_data(num_samples=5000):
    data = []
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date)

    public_holidays = {
        "2024-01-01", "2024-02-10", "2024-02-11", "2024-03-29", "2024-04-10", "2024-05-01",
        "2024-06-17", "2024-08-09", "2024-10-31", "2024-12-25"
    }

    for _ in range(num_samples):
        visit_date = pd.Timestamp(np.random.choice(date_range))
        # Define the sequence order of theme zones
        theme_zone_sequence = [
            "Hollywood", "New York", "Sci-Fi City", "Ancient Egypt", "The Lost World", "Far Far Away", "Hollywood"
        ]

        # Randomly choose how many zones the visitor will actually visit
        num_zones_visited = np.random.randint(3, 6)  # Random number of zones visited
        zones_to_visit = np.random.choice(theme_zone_sequence[:6], num_zones_visited, replace=False)
        zones_to_visit = sorted(zones_to_visit, key=lambda x: theme_zone_sequence.index(x))  # Maintain order

        visitor_id = fake.uuid4()
        loyalty_member = np.random.choice(["Yes", "No"], p=[0.2, 0.8])
        age = fake.random_int(min=5, max=70)
        gender = np.random.choice(["Male", "Female"])

        current_time = float(np.random.randint(9, 15))  # Start from a random check-in time

        attraction_times = []
        total_visit_duration = 0  # Track the total time spent across all zones

        for theme_zone in zones_to_visit:
            stay_duration = np.random.randint(1, 3) if theme_zone in ["Sci-Fi City", "Ancient Egypt"] else np.random.randint(1, 2)

            # Add lunch time (if overlaps 12-2 PM)
            if current_time <= 12 and current_time + stay_duration >= 12:
                stay_duration += np.random.uniform(0.5, 0.75)  # 30-45 mins

            # Add walking time (10-20 mins)
            stay_duration += np.random.uniform(10, 21) / 60  # Convert minutes to hours
            check_out_time = min(current_time + stay_duration, 17)

            # Random attractions based on theme zone
            num_attractions_visited = np.random.randint(2, 4) if stay_duration > 2 else np.random.randint(1, 2)
            available_attractions = ATTRACTIONS.get(theme_zone, [])

            attractions_visited = np.random.choice(
                available_attractions,
                size=min(num_attractions_visited, len(available_attractions)),
                replace=False
            ).tolist()

            for attraction in attractions_visited:
                queue_time = np.random.uniform(20, 60) if attraction in POPULAR_ATTRACTIONS else np.random.uniform(10, 30)
                ride_time = np.random.uniform(5, 15)
                total_time = (queue_time + ride_time) / 60  # Convert to hours
                attraction_times.append({
                    "Attraction": attraction,
                    "Check_In": round(current_time, 2),
                    "Queue_Time": round(queue_time, 2),
                    "Check_Out": round(current_time + total_time, 2)
                })
                current_time += total_time

            total_visit_duration += stay_duration

        # Calculate average queue time across all attractions visited
        average_queue_time = round(np.mean([a['Queue_Time'] for a in attraction_times]), 2) if attraction_times else 0

        # Spending behavior
        restaurant_spending = round(np.random.uniform(10, 16), 2) if np.random.rand() < 0.6 else 0
        merchandise_spending = round(np.random.uniform(30, 51), 2) if np.random.rand() < 0.4 else 0
        total_spending = round(restaurant_spending + merchandise_spending, 2)

        # Append the visitor data
        data.append({
            "Date": visit_date.strftime("%Y-%m-%d"),
            "Visitor_ID": visitor_id,
            "Loyalty_Member": loyalty_member,
            "Age": age,
            "Gender": gender,
            "Theme_Zone_Visited": zones_to_visit,
            "Attraction_Times": attraction_times,
            "Average_Queue_Time": average_queue_time,
            "Restaurant_Spending": restaurant_spending,
            "Merchandise_Spending": merchandise_spending,
            "Total_Spending": total_spending
        })

    return pd.DataFrame(data)

# Generate synthetic data
data_df = generate_synthetic_data(5000)
print(data_df.head())

# Save to CSV with float precision
data_df.to_csv("../../data/synthetic_iot_data_v2.csv", index=False, float_format="%.2f")