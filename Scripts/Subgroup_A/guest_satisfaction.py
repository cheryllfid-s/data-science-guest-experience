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

