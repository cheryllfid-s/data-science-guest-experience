import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import PercentFormatter
import matplotlib
matplotlib.use('Qt5Agg')  
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from pathlib import Path

# Get the directory of the current script (guest_satisfaction.py)
script_dir = Path(__file__).parent

# Navigate project root (Git/) and then into data/
project_root = script_dir.parent.parent  # Adjust based on your structure
csv_path = project_root / "data" / "survey.csv"

# Verify  path
print(f"Absolute CSV path: {csv_path}")
print(f"File exists? {csv_path.exists()}")

# Load the CSV
df = pd.read_csv(csv_path)


# Setting up visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)



# Calculate NPS
def calculate_nps(recommend_series):
    promoters = (recommend_series == 'Yes').sum()  
    detractors = (recommend_series == 'No').sum()
    total = recommend_series.count()
    return ((promoters - detractors) / total) * 100

nps = calculate_nps(df['Would you recommend USS to others?'])
print(f"Net Promoter Score: {nps:.1f}")
#NPS is 48.6 

# Analyse Rides & Attractions 
# %%
def analyze_rides(df):# Ride wait times vs satisfaction
    plt.figure()
    sns.boxplot(x='How long did you wait in line for rides on average during your visit?',
                y='On a scale of 1-5, how would you rate your overall experience at USS?',
                data=df,
                order=['Less than 15 minutes', '15 to 30 minutes', '31 to 45 minutes',
                       '46 to 60 minutes', '61 to 90 minutes', '90+ minutes'])
    plt.title('Ride Wait Times vs Overall Satisfaction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rides_wait_time_vs_satisfaction.png')
    plt.show()

    # Most popular rides
    ride_counts = df['Which ride or attraction was your favourite?'].value_counts().head(5)
    plt.figure()
    ride_counts.plot(kind='barh')
    plt.title('Top 5 Most Popular Rides/Attractions')
    plt.xlabel('Number of Votes')
    plt.tight_layout()
    plt.savefig('top_rides.png')
    plt.show()

analyze_rides(df)


# 2. Food and Beverages Analysis 

def analyze_food(df):
    # Food quality 
    plt.figure()
    df['How would you rate the food quality and service? '].value_counts(
        normalize=True).sort_index().plot(kind='bar')
    plt.title('Food Quality Ratings Distribution')
    plt.ylabel('Percentage of Responses')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.savefig('food_quality_distribution.png')
    plt.close()

    # Food variety 
    food_variety = df['Did you find a good variety of food options?  '].value_counts(normalize=True)
    plt.figure()
    food_variety.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Perception of Food Variety')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('food_variety.png')
    plt.close()

analyze_food(df)

# 3. Services Analysis

def analyze_staff(df):
    # Staff friendliness 
    plt.figure()
    df['Were the park staff at USS friendly and helpful? Rate on a scale from 1-5.'].value_counts(
        normalize=True).sort_index().plot(kind='bar')