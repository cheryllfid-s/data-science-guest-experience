import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    tripadvisor_data = pd.read_csv('../data/tripadvisor_reviews.csv')
    weather_data = pd.read_csv('../data/noaa_weather.csv')
    yelp_data = pd.read_csv('../data/yelp_dataset.csv')

    tripadvisor_data.fillna(method='ffill', inplace=True)
    scaler = StandardScaler()
    tripadvisor_data[['rating', 'wait_time']] = scaler.fit_transform(tripadvisor_data[['rating', 'wait_time']])

    return tripadvisor_data, weather_data, yelp_data

if __name__ == "__main__":
    preprocess_data()