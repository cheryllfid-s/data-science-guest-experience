import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_disney_data(file_path):
    return kagglehub.load_dataset(KaggleDatasetAdapter.HUGGING_FACE, "ayushtankha/hackathon", file_path)

def load_noaa_data(file_path):
    return pd.read_csv(file_path)

def load_tripadvisor_data(file_path):
    return kagglehub.load_dataset(KaggleDatasetAdapter.HUGGING_FACE, "joebeachcapital/hotel-reviews", file_path)

def load_yelp_data(file_path):
    return kagglehub.load_dataset(KaggleDatasetAdapter.HUGGING_FACE, "yelp-dataset/yelp-dataset", file_path)
