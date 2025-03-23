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

## --- Data exploration --- ##    # edit: added data exploration section
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
