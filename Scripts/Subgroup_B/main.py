import pandas as pd
from load_data import load_disney_data, load_noaa_data, load_tripadvisor_data, load_yelp_data
from preprocess import preprocess_disney_data, preprocess_weather_data, preprocess_reviews_data
from model import train_demand_model, train_complaint_model, evaluate_model

# Load the data
file_path_disney = ""  # Replace with the correct file path
file_path_noaa = ""  # Replace with NOAA dataset file path
file_path_tripadvisor = ""  # Replace with the correct file path
file_path_yelp = ""  # Replace with the correct file path

df_disney = load_disney_data(file_path_disney)
df_noaa = load_noaa_data(file_path_noaa)
df_tripadvisor = load_tripadvisor_data(file_path_tripadvisor)
df_yelp = load_yelp_data(file_path_yelp)

# Preprocess the data
df_disney = preprocess_disney_data(df_disney)
df_noaa = preprocess_weather_data(df_noaa)
df_tripadvisor = preprocess_reviews_data(df_tripadvisor)
df_yelp = preprocess_reviews_data(df_yelp)

# Merge datasets (for demand prediction)
df = pd.merge(df_disney, df_noaa, on='DATE', how='left')

# Feature Engineering for demand prediction
features = ['TEMP', 'PRCP', 'WDSP', 'VISIB']
target = 'WAIT_TIME'
X = df[features]
y = df[target]

# Train-test split for demand prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the demand model
demand_model = train_demand_model(X_train, y_train)
demand_mae = evaluate_model(demand_model, X_test, y_test)
print(f"Demand Prediction MAE: {demand_mae}")

# Text data for complaint prediction
df_reviews = pd.concat([df_tripadvisor[['Review', 'Rating']], df_yelp[['text', 'stars']]], ignore_index=True)
df_reviews.columns = ['Review', 'Rating']

# Text vectorization for complaint prediction
vectorizer = CountVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(df_reviews['Review'])
y_text = df_reviews['Complaint']

# Train-test split for complaint prediction
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

# Train the complaint model
complaint_model = train_complaint_model(X_train_text, y_train_text)
complaint_accuracy = evaluate_model(complaint_model, X_test_text, y_test_text)
print(f"Complaint Prediction Accuracy: {complaint_accuracy}")
