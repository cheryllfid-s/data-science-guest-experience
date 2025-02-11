import pandas as pd

def preprocess_disney_data(df):
    # Data cleaning and feature engineering for Disney dataset
    df.fillna(df.mean(), inplace=True)
    return df

def preprocess_weather_data(df):
    # Data cleaning for NOAA weather data
    df.fillna(df.mean(), inplace=True)
    return df

def preprocess_reviews_data(df):
    # Clean and prepare the reviews data
    df['Complaint'] = df['Rating'].apply(lambda x: 1 if x <= 3 else 0)
    return df
