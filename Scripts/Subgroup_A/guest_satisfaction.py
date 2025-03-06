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
data = pd.read_csv(csv_path)

