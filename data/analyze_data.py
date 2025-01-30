import pandas as pd
from sklearn.cluster import KMeans
from textblob import TextBlob

def analyze_guest_journey(tripadvisor_data):
    tripadvisor_data['sentiment'] = tripadvisor_data['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    kmeans = KMeans(n_clusters=4)
    tripadvisor_data['segment'] = kmeans.fit_predict(tripadvisor_data[['age', 'spending', 'satisfaction']])
    return tripadvisor_data

if __name__ == "__main__":
    tripadvisor_data = pd.read_csv('../data/tripadvisor_reviews.csv')
    analyzed_data = analyze_guest_journey(tripadvisor_data)
    analyzed_data.to_csv('../data/analyzed_tripadvisor_data.csv', index=False)