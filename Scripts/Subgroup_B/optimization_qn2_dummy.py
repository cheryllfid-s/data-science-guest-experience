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
os.chdir("C:/Users/parma/data-science-guest-experience/Scripts/Subgroup_B")

# Load the datasets
df_survey = pd.read_csv("../../data/survey.csv")
df_weather = pd.read_csv("../../data/weather_data.csv")

# --- Clean Survey Data ---
df_survey = df_survey.rename(columns={
    "On a scale of 1-5, how would you rate your overall experience at USS?": "Guest_Satisfaction_Score",
    "How long did you wait in line for rides on average during your visit?": "Wait_Time",
    "Timestamp": "Timestamp"
})

wait_time_mapping = {
    "Less than 15 mins": 10,
    "15-30 mins": 22.5,
    "30-45 mins": 37.5,
    "45 mins - 1 hr": 52.5,
    "More than 1 hr": 75
}
df_survey["Wait_Time"] = df_survey["Wait_Time"].map(wait_time_mapping).fillna(37.5)
df_survey["Guest_Satisfaction_Score"] = pd.to_numeric(df_survey["Guest_Satisfaction_Score"], errors="coerce")
df_survey["Timestamp"] = pd.to_datetime(df_survey["Timestamp"]).dt.date
df_survey["date"] = df_survey["Timestamp"]

# --- Clean Weather Data ---
df_weather = df_weather.rename(columns={
    "date": "date",
    "temperature": "temperature",
    "rainfall": "rainfall",
    "humidity": "humidity"
})
df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.date

# --- Merge Datasets ---
df_merged = pd.merge(df_survey, df_weather, on="date", how="inner")
df_merged = df_merged.dropna(subset=["Guest_Satisfaction_Score", "temperature", "rainfall", "humidity"])

# --- Train Model ---
X = df_merged[["temperature", "rainfall", "humidity"]]
y = df_merged["Guest_Satisfaction_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --- Create 7-Day Forecast ---
future_dates = pd.date_range(start=datetime.now(), periods=7, freq="D").date
forecast_weather = df_weather.tail(7).copy()
forecast_weather = forecast_weather.reset_index(drop=True)
forecast_weather["date"] = future_dates
forecast_weather = forecast_weather[["date", "temperature", "rainfall", "humidity"]]

forecast_weather["predicted_satisfaction"] = model.predict(forecast_weather[["temperature", "rainfall", "humidity"]])
forecast_weather.reset_index(drop=True, inplace=True)

# --- Optimize Layouts ---
class ThemePark:
    def __init__(self, env, attractions, layout="single_queue", use_right_entrance=True):
        self.env = env
        self.layout = layout
        self.use_right_entrance = use_right_entrance  # Control entry points
        self.attractions = {name: simpy.Resource(env, capacity=(2 if layout == "single_queue" else 4)) for name in attractions.keys()}
        self.wait_times = {name: [] for name in attractions.keys()}
        self.total_times = []  # Track total experience time per guest
        self.visit_counts = {name: 0 for name in attractions.keys()}  # Track visits to each attraction

    def guest(self, env, name, attractions_to_visit):
        """Simulate a guest visiting a sequence of attractions."""
        start_time = env.now
        # Choose entry point: 50% left (0,0), 50% right (8,0) if using right entrance
        if self.use_right_entrance and random.random() < 0.5:
            current_pos = (8, 0)  # Right side entrance
        else:
            current_pos = (0, 0)  # Left side entrance

        for i, attraction_name in enumerate(attractions_to_visit):
            # Calculate walking time
            if i == 0:
                walk_time = walking_time(("Entry", current_pos), attraction_name)
            else:
                walk_time = walking_time(attractions_to_visit[i-1], attraction_name)
            yield env.timeout(walk_time)

            # Queue and ride
            arrival_time = env.now
            with self.attractions[attraction_name].request() as req:
                yield req
                wait_time = env.now - arrival_time
                self.wait_times[attraction_name].append(wait_time)
                self.visit_counts[attraction_name] += 1  # Increment visit count
                service_rate = 16 if self.layout == "single_queue" else 4  # Adjusted for realistic wait times
                ride_duration = np.random.exponential(7 / service_rate)
                yield env.timeout(ride_duration)
            current_pos = attractions_map[attraction_name]
        self.total_times.append(env.now - start_time)

    def generate_guests(self, num_guests, arrival_rates):
        """Generate guests visiting a sequence of attractions."""
        for period, (start, end, rate) in arrival_rates.items():
            while self.env.now < end:
                attractions_to_visit = random.sample(list(self.attractions.keys()), 3)  # Visit 3 attractions
                self.env.process(self.guest(self.env, f"Guest_{int(self.env.now)}", attractions_to_visit))
                yield self.env.timeout(np.random.exponential(rate))

# --- Run Simulation ---
def run_simulation(num_guests, attractions, layout, use_right_entrance=True):
    """Execute the simulation for a given layout."""
    env = simpy.Environment()
    park = ThemePark(env, attractions, layout, use_right_entrance)
    arrival_rates = {
        "Morning": (0, 120, 0.5),    # μ = 0.5 min
        "Afternoon": (120, 420, 0.1667),  # μ = 0.1667 min (high traffic)
        "Evening": (420, 540, 0.25)  # μ = 0.25 min
    }
    env.process(park.generate_guests(num_guests, arrival_rates))
    env.run(until=540)  # 9 hours
    avg_wait_times = {name: np.mean(times) if times else 0 for name, times in park.wait_times.items()}
    return avg_wait_times, park.visit_counts, park.total_times

# --- Define Attraction Layouts ---
# attractions_layout_1 = {
#     "Revenge of the Mummy": 2,
#     "Battlestar Galactica: CYLON": 3,
#     "Transformers: The Ride": 1
# }  # Single Queue
# attractions_layout_2 = {
#     "Revenge of the Mummy": 4,
#     "Battlestar Galactica: CYLON": 5,
#     "Transformers: The Ride": 2
# }  # Multi Queue

# Define attractions with coordinates
attractions_map = {
    "Revenge of the Mummy": (3, 5),
    "Battlestar Galactica: CYLON": (5, 3),
    "Transformers: The Ride": (6, 4),
    "Puss In Boots' Giant Journey": (2, 6),
    "Sesame Street Spaghetti Space Chase": (4, 2)
}

def walking_time(attraction_1, attraction_2):
    """Calculate walking time between two points."""
    if attraction_1[0] == "Entry":
        x1, y1 = attraction_1[1]
    else:
        x1, y1 = attractions_map[attraction_1]
    x2, y2 = attractions_map[attraction_2]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance * 0.5  # 0.5 min/unit

def compare_layouts():
    """Compare current vs. modified park layouts for both single and multi-queue."""
    # Reset attractions_map to original coordinates
    attractions_map.update({
        "Revenge of the Mummy": (3, 5),
        "Battlestar Galactica: CYLON": (5, 3),
        "Transformers: The Ride": (6, 4),
        "Puss In Boots' Giant Journey": (2, 6),
        "Sesame Street Spaghetti Space Chase": (4, 2)
    })

    # Realistic arrival rates
    arrival_rates = {
        "Morning": (0, 120, 0.5),    # μ = 0.5 min
        "Afternoon": (120, 420, 0.1667),  # μ = 0.1667 min (high traffic)
        "Evening": (420, 540, 0.25)  # μ = 0.25 min
    }

    # Current Layout: Two entrances (left and right)
    # Single Queue
    env1_single = simpy.Environment()
    park1_single = ThemePark(env1_single, attractions_map, layout="single_queue", use_right_entrance=True)
    env1_single.process(park1_single.generate_guests(13000, arrival_rates))
    env1_single.run(until=540)

    # Multi Queue
    env1_multi = simpy.Environment()
    park1_multi = ThemePark(env1_multi, attractions_map, layout="multi_queue", use_right_entrance=True)
    env1_multi.process(park1_multi.generate_guests(13000, arrival_rates))
    env1_multi.run(until=540)

    # Modified Layout: Single left entrance, swap Transformers and CYLON
    attractions_map["Transformers: The Ride"] = (5, 3)  # Swap with CYLON
    attractions_map["Battlestar Galactica: CYLON"] = (6, 4)  # Move to the right
    # # Single Queue
    # env2_single = simpy.Environment()
    # park2_single = ThemePark(env2_single, attractions_map, layout="single_queue", use_right_entrance=False)
    # env2_single.process(park2_single.generate_guests(13000, arrival_rates))
    # env2_single.run(until=540)

    # Multi Queue
    env2_multi = simpy.Environment()
    park2_multi = ThemePark(env2_multi, attractions_map, layout="multi_queue", use_right_entrance=False)
    env2_multi.process(park2_multi.generate_guests(13000, arrival_rates))
    env2_multi.run(until=540)

    # Compute metrics
    # Current Layout
    avg_wait_times_1_single = {name: np.mean(times) if times else 0 for name, times in park1_single.wait_times.items()}
    avg_total_time_1_single = np.mean(park1_single.total_times) if park1_single.total_times else 0
    visit_counts_1_single = park1_single.visit_counts

    avg_wait_times_1_multi = {name: np.mean(times) if times else 0 for name, times in park1_multi.wait_times.items()}
    avg_total_time_1_multi = np.mean(park1_multi.total_times) if park1_multi.total_times else 0
    visit_counts_1_multi = park1_multi.visit_counts

    # # Modified Layout
    # avg_wait_times_2_single = {name: np.mean(times) if times else 0 for name, times in park2_single.wait_times.items()}
    # avg_total_time_2_single = np.mean(park2_single.total_times) if park2_single.total_times else 0
    # visit_counts_2_single = park2_single.visit_counts

    avg_wait_times_2_multi = {name: np.mean(times) if times else 0 for name, times in park2_multi.wait_times.items()}
    avg_total_time_2_multi = np.mean(park2_multi.total_times) if park2_multi.total_times else 0
    visit_counts_2_multi = park2_multi.visit_counts

    # Print results
    # print("\nCurrent USS Layout (Two Entrances) - Single Queue:")
    # for attraction, time in avg_wait_times_1_single.items():
    #     print(f"{attraction}: {time:.2f} min")
    # print(f"Average Total Time (Wait + Walk): {avg_total_time_1_single:.2f} min")
    # print("Visit Counts:", visit_counts_1_single)

    print("\nCurrent USS Layout (Two Entrances) - Multi Queue:")
    for attraction, time in avg_wait_times_1_multi.items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Total Time (Wait + Walk): {avg_total_time_1_multi:.2f} min")
    print("Visit Counts:", visit_counts_1_multi)

    # print("\nModified USS Layout (Left Entrance Only, Swapped Transformers and CYLON) - Single Queue:")
    # for attraction, time in avg_wait_times_2_single.items():
    #     print(f"{attraction}: {time:.2f} min")
    # print(f"Average Total Time (Wait + Walk): {avg_total_time_2_single:.2f} min")
    # print("Visit Counts:", visit_counts_2_single)

    print("\nModified USS Layout (Left Entrance Only, Swapped Transformers and CYLON) - Multi Queue:")
    for attraction, time in avg_wait_times_2_multi.items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Total Time (Wait + Walk): {avg_total_time_2_multi:.2f} min")
    print("Visit Counts:", visit_counts_2_multi)

    # Justification
    print("\nJustification for Modified Layout:")
    print("- Removed the right-side entrance to force a left-to-right traversal, reducing congestion at central attractions.")
    print("- Swapped 'Transformers: The Ride' and 'Battlestar Galactica: CYLON' to move a high-traffic ride to the right, balancing guest flow.")
    print("- Result: Lower total experience time due to optimized walking distances and better distribution of guests.")

if __name__ == "__main__":
    # Satisfaction forecast already generated above
    print("\n 7-Day Forecast of Guest Satisfaction:")
    print(forecast_weather[["date", "temperature", "rainfall", "humidity", "predicted_satisfaction"]])


    # Run simulations for single and multi-queue
    single_queue_wait_times, single_queue_visits, single_queue_total_times = run_simulation(13000, attractions_map, "single_queue", use_right_entrance=True)
    multi_queue_wait_times, multi_queue_visits, multi_queue_total_times = run_simulation(13000, attractions_map, "multi_queue", use_right_entrance=True)

    # Display baseline results
    # print("\nBaseline Single-Queue Layout - Average Wait Times (minutes):")
    # for attraction, wait_time in single_queue_wait_times.items():
    #     print(f" - {attraction}: {wait_time:.2f} min")
    # print("Visit Counts:", single_queue_visits)
    # print(f"Average Total Time (Wait + Walk): {np.mean(single_queue_total_times):.2f} min")

    print("\nBaseline Multi-Queue Layout - Average Wait Times (minutes):")
    for attraction, wait_time in multi_queue_wait_times.items():
        print(f" - {attraction}: {wait_time:.2f} min")
    print("Visit Counts:", multi_queue_visits)
    print(f"Average Total Time (Wait + Walk): {np.mean(multi_queue_total_times):.2f} min")

    # Compare layouts
    compare_layouts()

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(single_queue_wait_times.keys(), single_queue_wait_times.values(), alpha=0.6, label="Single Queue")
    plt.bar(multi_queue_wait_times.keys(), multi_queue_wait_times.values(), alpha=0.6, label="Multi Queue")
    plt.xlabel("Attractions")
    plt.ylabel("Average Wait Time (minutes)")
    plt.title("Comparison of Wait Times: Single vs. Multi-Queue (Baseline Layout)")
    plt.legend()
    plt.show()