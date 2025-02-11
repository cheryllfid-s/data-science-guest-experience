import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Disney Dataset (via Hugging Face)
file_path_disney = ""  # Replace with the correct file path
df_disney = kagglehub.load_dataset(KaggleDatasetAdapter.HUGGING_FACE, "ayushtankha/hackathon", file_path_disney)

# Load NOAA Weather Data (replace with actual file path)
file_path_noaa = ""  # Replace with NOAA dataset file path
df_noaa = pd.read_csv(file_path_noaa)

# Merge datasets on DATE
df = pd.merge(df_disney, df_noaa, on='DATE', how='left')

# Feature Engineering
df.fillna(df.mean(), inplace=True)

# Define features and target
features = ['TEMP', 'PRCP', 'WDSP', 'VISIB']  # Weather-related features
target = 'WAIT_TIME'

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
