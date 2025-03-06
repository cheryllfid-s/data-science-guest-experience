import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import PercentFormatter

# Setting up visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the CSV file
csv_path = "../data/survey.csv"  # Relative path from Scripts/Subgroup_A/ to data/
df = pd.read_csv(csv_path)

# Calculate NPS
def calculate_nps(recommend_series):
    promoters = (recommend_series == 'Yes').sum()
    detractors = (recommend_series == 'No').sum()
    total = recommend_series.count()
    return ((promoters - detractors) / total) * 100

nps = calculate_nps(df['Would you recommend USS to others?'])
print(f"Net Promoter Score: {nps:.1f}")

# Analyse Rides & Attractions 
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
    plt.close()

    # Most popular rides
    ride_counts = df['Which ride or attraction was your favourite?'].value_counts().head(5)
    plt.figure()
    ride_counts.plot(kind='barh')
    plt.title('Top 5 Most Popular Rides/Attractions')
    plt.xlabel('Number of Votes')
    plt.tight_layout()
    plt.savefig('top_rides.png')
    plt.close()

analyze_rides(df)

