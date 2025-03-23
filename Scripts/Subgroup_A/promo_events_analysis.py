## ----- Set up ----- ##
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from emoji import demojize
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import corpora

from gensim.models import LdaModel
from matplotlib.pyplot import thetagrids
from scipy.stats import stats

# ----- Required downloads -----
# nltk.download('stopwords')
# nltk.download('punkt_tab')

## ----- Import review and promotion events datasets ----- ##
path = os.getcwd()
# df_reviews = pd.read_csv(path + '/data/universal_studio_branches.csv')
df_events = pd.read_csv(path + '/data/uss_promo_events.csv')

## ----- Data cleaning ----- ##
# (1) Check for missing values
def handle_missing_values(df, drop=True, fill_value=None):
    missing_counts = df.isnull().sum()
    if missing_counts.sum() == 0:
        print("No missing values found.")
        return df
    print("Missing values per column:\n", missing_counts)
    if drop:
        df = df.dropna()
        print("Dropped rows with missing values.")
    elif fill_value is not None:
        df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}.")

    return df

df_reviews = handle_missing_values(df_reviews)
df_events = handle_missing_values(df_events)

# (2) Convert dates to datetime type
df_reviews["written_date"] = pd.to_datetime(df_reviews["written_date"], errors='coerce')
df_events["start"] = pd.to_datetime(df_events["start"], format='%b %d, %Y', errors='coerce')
df_events["end"] = pd.to_datetime(df_events["end"], format='%b %d, %Y', errors='coerce')

df_events = df_events.sort_values("start")  # order events by start date

# (3) Handle duplicates
def remove_duplicates(df):
    # check if there are duplicated rows
    dup_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {dup_count}")

    # drop duplicates if any
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        print("Duplicates removed.")
    else:
        print("No duplicates found.")
    return df

df_reviews = remove_duplicates(df_reviews)
df_events = remove_duplicates(df_events)

## --- Sentiment analysis --- ##
"""
Sentiment analysis: df_reviews 
-------------------
Using TextBlob to compute the polarity scores 
We will be analysing the 'combined_text' = review title + text
"""

# Combine the text columns
df_reviews['combined_text'] = df_reviews['title'] + " " + df_reviews['review_text']
df_reviews = df_reviews.drop(columns=['title', 'review_text'])


# (1) Text preprocessing: convert emojis
df_reviews["combined_text"] = df_reviews["combined_text"].apply(lambda text: demojize(text))

# (2) TextBlob: compute polarity scores
from textblob import TextBlob

def analyse_sentiment(df, text_column):
    df["polarity"] = df[text_column].apply(lambda text: TextBlob(text).sentiment.polarity)
    return df

df_reviews = analyse_sentiment(df_reviews, 'combined_text')

## --- Data exploration --- ##    
"""
Data exploration
-------------------
Since 'df_events' contains information on past promotion events of Universal Studios Singapore (USS),
we will filter the 'df_reviews' data to focus our analysis on USS reviews.
"""
# (1) Compute event duration
df_events["duration"] = (df_events["end"] - df_events["start"]).dt.days

# (2) Filter for reviews written during event period
def filter_reviews_during_events(reviews, events, branch = 'Universal Studios Singapore'):
    event_reviews_dict = {}

    for _, event in events.iterrows():
        event_key = f"{event['event']} ({event['start'].date()})"  # event as unique key
        start = event["start"]
        end = event["end"]

        filtered_reviews = reviews[
            (reviews["written_date"] >= start) &
            (reviews["written_date"] <= end) &
            (reviews["branch"] == branch)
        ]

        event_reviews_dict[event_key] = filtered_reviews

    return event_reviews_dict

# (3) Filter for reviews written before event period
def filter_reviews_before_events(reviews, events, branch = 'Universal Studios Singapore'):
    event_reviews_dict = {}
    for _, event in events.iterrows():
        event_key = f"{event['event']} ({event['start'].date()})"  # event as unique key

        # set the period before event to have the same duration for fair comparison
        start = event["start"] - timedelta(days=event["duration"])
        end = event["start"] - timedelta(days=1)

        filtered_reviews = reviews[
            (reviews["written_date"] >= start) &
            (reviews["written_date"] <= end) &
            (reviews["branch"] == branch)
        ]

        event_reviews_dict[event_key] = filtered_reviews

    return event_reviews_dict

# Call both functions to create 2 filtered dataframes
# Structure of filtered df: {event_key: [df of filtered reviews]}
reviews_during_events = filter_reviews_during_events(df_reviews, df_events)
reviews_before_events = filter_reviews_before_events(df_reviews, df_events)

# Drop unnecessary column
df_events = df_events.drop(columns=["duration"])

## --- Data Analysis --- ##
"""
Data Analysis: Change in polarity scores, review rating, review volume; statistical analysis
-------------------
Calculate the change in polarity scores before vs during event:
 - A positive value indicates higher satisfaction during event period
 - A negative value indicates lower satisfaction during event period
"""

# (1) Compute difference in key metrics before event vs during event
def compute_change_in_reviews(reviews_before_event, reviews_during_event):
    changes = {}
    for event, reviews_before in reviews_before_event.items():
        # Get reviews during the event from the second dictionary
        reviews_during = reviews_during_event.get(event)
        if reviews_during is not None:
            # Ensure the reviews are DataFrames before performing calculations
            reviews_before = pd.DataFrame(reviews_before)
            reviews_during = pd.DataFrame(reviews_during)

            # Calculate average polarity score, review rating, and review volume before and during the event
            before_avg = reviews_before[['polarity', 'rating']].mean()
            during_avg = reviews_during[['polarity', 'rating']].mean()
            before_volume = len(reviews_before)
            during_volume = len(reviews_during)

            # Calculate change in polarity score, review rating, and review volume
            change = {
                'review_polarity_change': during_avg['polarity'] - before_avg['polarity'],
                'review_rating_change': during_avg['rating'] - before_avg['rating'],
                'review_volume_change': during_volume - before_volume
            }
            changes[event] = change

    return pd.DataFrame.from_dict(changes, orient='index')


df_change_data = compute_change_in_reviews(reviews_before_events, reviews_during_events)
