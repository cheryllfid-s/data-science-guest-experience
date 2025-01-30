import pandas as pd
from pyod.models.knn import KNN

def detect_anomalies(tripadvisor_data):
    clf = KNN()
    clf.fit(tripadvisor_data[['age', 'spending', 'satisfaction']])
    tripadvisor_data['anomaly'] = clf.predict(tripadvisor_data[['age', 'spending', 'satisfaction']])
    return tripadvisor_data

if __name__ == "__main__":
    tripadvisor_data = pd.read_csv('../data/tripadvisor_reviews.csv')
    anomaly_data = detect_anomalies(tripadvisor_data)
    anomaly_data.to_csv('../data/anomaly_data.csv', index=False)