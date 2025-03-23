## Set up
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

## ----- Import review and promotion events datasets -----
path = os.getcwd()
# df_reviews = pd.read_csv(path + '/data/universal_studio_branches.csv')
df_events = pd.read_csv(path + '/data/uss_promo_events.csv')

## ----- Data cleaning -----
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
