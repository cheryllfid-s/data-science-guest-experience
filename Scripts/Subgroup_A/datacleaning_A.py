import kagglehub
import pandas as pd
import os
from emoji import demojize
from textblob import TextBlob

# (Q3) data journey analysis 
def prepare_tivoli_data():
    # Download latest version of disney dataset
    path = kagglehub.dataset_download("ayushtankha/hackathon")
    print("Path to dataset files:", path)

    waitingtimes_path = os.path.join(path, "waiting_times.csv")
    waiting_times_df = pd.read_csv(waitingtimes_path)

    parkattr_path = os.path.join(path, "link_attraction_park.csv")
    parkattr_df = pd.read_csv(parkattr_path)

    attendace_path = os.path.join(path, "attendance.csv")
    attendance_df = pd.read_csv(attendace_path)

    # FORMATTING THE DATA
    # (1) Turn the deb_time into datetime objects and remove seconds (they are all 0)
    waiting_times_df['DEB_TIME'] = pd.to_datetime(waiting_times_df['DEB_TIME'])
    waiting_times_df['DEB_TIME'] = waiting_times_df['DEB_TIME'].dt.strftime('%H:%M')

    # (2) Turn guest_carried into integers + round off capacity and adjust capacity to their nearest integers
    waiting_times_df['GUEST_CARRIED'] = waiting_times_df['GUEST_CARRIED'].astype(int)
    waiting_times_df['CAPACITY'] = waiting_times_df['CAPACITY'].round().astype(int)
    waiting_times_df['ADJUST_CAPACITY'] = waiting_times_df['ADJUST_CAPACITY'].round().astype(int)

    # (3) Sorting based on the date and time of debarkation
    waiting_times_df = waiting_times_df.sort_values(['WORK_DATE', 'DEB_TIME'])

    # Filter out data where rides were not operational
    waitingtimes_oper = waiting_times_df[
        (waiting_times_df['GUEST_CARRIED'] != 0) & 
        (waiting_times_df['CAPACITY'] != 0)
    ]

    waitingtimes_oper = waitingtimes_oper.sort_values(['WORK_DATE', 'DEB_TIME', 'WAIT_TIME_MAX'], ascending=[True, True, False])
    waitingtimes_oper = waitingtimes_oper.drop(columns=['DEB_TIME_HOUR', 'FIN_TIME', 'NB_UNITS', 'OPEN_TIME', 'UP_TIME', 
                                                        'NB_MAX_UNIT', 'DOWNTIME', 'GUEST_CARRIED', 'CAPACITY', 'ADJUST_CAPACITY'])

    # Merge with link_attraction_park.csv
    parkattr_df[['ATTRACTION', 'PARK']] = parkattr_df['ATTRACTION;PARK'].str.split(";", expand=True)
    parkattr_df = parkattr_df.drop(columns=['ATTRACTION;PARK'])
    merged_df = pd.merge(waitingtimes_oper, parkattr_df, left_on='ENTITY_DESCRIPTION_SHORT', right_on='ATTRACTION', how='inner')
    merged_df = merged_df.drop(columns=['ENTITY_DESCRIPTION_SHORT'])

    # Filter for Tivoli Gardens
    tivoli_g = merged_df[merged_df['PARK'] == 'Tivoli Gardens']
    tivoli_g['WORK_DATE'] = pd.to_datetime(tivoli_g['WORK_DATE'])
    tivoli_g['TIMESTAMP'] = pd.to_datetime(tivoli_g['WORK_DATE'].astype(str) + ' ' + tivoli_g['DEB_TIME'])

    # Finding COVID dates
    attendance_df['USAGE_DATE'] = pd.to_datetime(attendance_df['USAGE_DATE'])
    covid = attendance_df[(attendance_df['USAGE_DATE'] >= '2020-03-01') & (attendance_df['USAGE_DATE'] <= '2021-08-01')]

    negative_att = attendance_df[attendance_df['attendance'] < 0]

    return tivoli_g, attendance_df, covid, negative_att


# (4) analysis of past promotion events
def q4_handle_missing_values(df, drop=True, fill_value=None):
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

def q4_remove_duplicates(df):
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

def q4_analyse_sentiment(df, text_column):
    df["polarity"] = df[text_column].apply(lambda text: TextBlob(text).sentiment.polarity)
    return df
    
def q4_prepare_reviews_data():
    # (1) Download and import universal_studio_branches.csv
    kaggle_download_path = kagglehub.dataset_download("dwiknrd/reviewuniversalstudio")
    print("Path to dataset files:", kaggle_download_path)
    
    reviews_path = os.path.join(kaggle_download_path, "universal_studio_branches.csv")
    df_reviews = pd.read_csv(reviews_path)

    # (2) Check for missing values
    df_reviews = handle_missing_values(df_reviews)

    # (3) Convert to datetime object
    df_reviews["written_date"] = pd.to_datetime(df_reviews["written_date"], errors='coerce')

    # (4) Remove duplicates
    df_reviews = remove_duplicates(df_reviews)

    # (5) Combine the text columns
    df_reviews['combined_text'] = df_reviews['title'] + " " + df_reviews['review_text']
    df_reviews = df_reviews.drop(columns=['title', 'review_text'])

    # (6) Convert emojis
    df_reviews["combined_text"] = df_reviews["combined_text"].apply(lambda text: demojize(text))

    # (7) Compute polarity scores
    df_reviews = analyse_sentiment(df_reviews, 'combined_text')

    return df_reviews

def q4_prepare_events_data():
    # (1) Import uss_promo_events.csv
    events_path = os.path.join("../../data/uss_promo_events.csv")
    print(events_path)
    df_events = pd.read_csv(events_path)

    # (2) Check for missing values
    df_events = handle_missing_values(df_events)

    # (3) Convert to datetime object
    df_events["start"] = pd.to_datetime(df_events["start"], format='%b %d, %Y', errors='coerce')
    df_events["end"] = pd.to_datetime(df_events["end"], format='%b %d, %Y', errors='coerce')
    df_events = df_events.sort_values("start")  # order events by start date

    # (4) Remove duplicates
    df_events = remove_duplicates(df_events)

    # (5) Compute event duration
    df_events["duration"] = (df_events["end"] - df_events["start"]).dt.days

    return df_events
