import kagglehub
import pandas as pd
import os
from emoji import demojize
from textblob import TextBlob
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


# (Q2) Guest Segment Data
def cleaning_q2():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    csv_path = project_root / "data" / "survey.csv"

    survey = pd.read_csv(csv_path)
    survey = survey.rename(columns={
        'Which age group do you belong to?': 'age_group',
        'What is your employment status?': 'employment',
        'Who did you visit USS with?': 'group_type',
        'What was the main purpose of your visit?': 'visit_purpose',
        'On a scale of 1-5, how would you rate your overall experience at USS?': 'experience_rating',
        'Did you purchase the Express Pass?': 'express_pass',
        'How long did you wait in line for rides on average during your visit?': 'avg_wait_time',
        'Would you choose to revisit USS?': 'revisit',
        'Would you recommend USS to others?': 'recommend',
        'How did you first hear about Universal Studios Singapore?': 'awareness',
        'Have you seen any recent advertisements or promotions for USS?': 'response_to_ads',
        'What type of promotions or discounts would encourage you to visit USS?': 'preferred_promotion'
    })

    selected_cols = [
        'age_group', 'group_type', 'visit_purpose', 'express_pass', 'experience_rating',
        'awareness', 'response_to_ads', 'preferred_promotion'
    ]
    survey_clean = survey[selected_cols].dropna().reset_index(drop=True)

    # ==== SYNTHETIC DATA GENERATION ====
    np.random.seed(42)
    n_samples = 400
    group_type_syn = np.random.choice(['Friends', 'Family (including children)'], size=n_samples, p=[0.5, 0.5])

    age_group_syn = ['18 - 24 years old' if gt == 'Friends' else '25 - 34 years old' for gt in group_type_syn]
    visit_purpose_syn = ['Social gathering' if gt == 'Friends' else 'Family outing' for gt in group_type_syn]
    express_pass_syn = [
        np.random.choice(['Yes', 'No'], p=[0.25, 0.75]) if gt == 'Friends' else
        np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
        for gt in group_type_syn
    ]
    experience_rating_syn = [
        round(np.random.normal(4.2, 0.3), 1) if ep == 'Yes' else round(np.random.normal(3.5, 0.5), 1)
        for ep in express_pass_syn
    ]
    awareness_syn = [np.random.choice(
        ['Word of mouth', 'Social media', 'Online ads', 'Travel agencies/tour packages', 'News'],
        p=[0.6, 0.3, 0.05, 0.025, 0.025]) for _ in range(n_samples)]
    response_to_ads_syn = [np.random.choice(
        ['Yes, but they did not influence my decision', 'Yes and they influenced my decision', "No, I haven't seen any ads"],
        p=[0.7, 0.1, 0.2]) if gt == 'Friends' else np.random.choice(
        ["No, I haven't seen any ads", 'Yes, but they did not influence my decision', 'Yes and they influenced my decision'],
        p=[0.7, 0.2, 0.1]) for gt in group_type_syn]
    preferred_promotion_syn = [np.random.choice(
        ['Discounted tickets', 'Family/group discounts', 'Seasonal event promotions', 'Bundle deals (hotel + ticket packages)'],
        p=[0.5, 0.25, 0.15, 0.1]) for _ in range(n_samples)]

    synthetic = pd.DataFrame({
        'age_group': age_group_syn,
        'group_type': group_type_syn,
        'visit_purpose': visit_purpose_syn,
        'express_pass': express_pass_syn,
        'experience_rating': experience_rating_syn,
        'awareness': awareness_syn,
        'response_to_ads': response_to_ads_syn,
        'preferred_promotion': preferred_promotion_syn
    })

    df_combined = pd.concat([survey_clean, synthetic], ignore_index=True)
    df_labeled = df_combined.copy()

    for col in df_combined.select_dtypes(include='object').columns:
        df_combined[col] = LabelEncoder().fit_transform(df_combined[col])

    scaled = StandardScaler().fit_transform(df_combined)
    pca = PCA(n_components=2).fit_transform(scaled)

    return df_combined, df_labeled, scaled, pca


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
    print("\nImporting reviews data...")
    kaggle_download_path = kagglehub.dataset_download("dwiknrd/reviewuniversalstudio")
    print("Path to dataset:", kaggle_download_path)

    reviews_path = os.path.join(kaggle_download_path, "universal_studio_branches.csv")
    df_reviews = pd.read_csv(reviews_path)

    # (2) Check for missing values
    print("Cleaning reviews data...")
    df_reviews = q4_handle_missing_values(df_reviews)

    # (3) Convert to datetime object
    df_reviews["written_date"] = pd.to_datetime(df_reviews["written_date"], errors='coerce')

    # (4) Remove duplicates
    df_reviews = q4_remove_duplicates(df_reviews)

    # (5) Combine the text columns
    df_reviews['combined_text'] = df_reviews['title'] + " " + df_reviews['review_text']
    df_reviews = df_reviews.drop(columns=['title', 'review_text'])

    # (6) Convert emojis
    df_reviews["combined_text"] = df_reviews["combined_text"].apply(lambda text: demojize(text))

    # (7) Compute polarity scores
    print("Computing review polarity scores...")
    df_reviews = q4_analyse_sentiment(df_reviews, 'combined_text')
    print("df_reviews is ready for analysis.\n")

    return df_reviews


def q4_prepare_events_data():
    # (1) Import uss_promo_events.csv
    print("Importing promo events data...")
    events_path = os.path.join("../../Data/Raw Data/uss_promo_events.csv")
    print("Path to dataset:", events_path)
    df_events = pd.read_csv(events_path)

    # (2) Check for missing values
    print("Cleaning events data...")
    df_events = q4_handle_missing_values(df_events)

    # (3) Convert to datetime object
    df_events["start"] = pd.to_datetime(df_events["start"], format='%b %d, %Y', errors='coerce')
    df_events["end"] = pd.to_datetime(df_events["end"], format='%b %d, %Y', errors='coerce')
    df_events = df_events.sort_values("start")  # order events by start date

    # (4) Remove duplicates
    df_events = q4_remove_duplicates(df_events)

    # (5) Compute event duration
    df_events["duration"] = (df_events["end"] - df_events["start"]).dt.days
    print("df_events is ready for analysis.\n")

    return df_events


# (5) External factors
def q5_clean_data():

    csv_path_reviews = Path("../../data/usstripadvisor.csv").resolve()
    print(f"Absolute CSV path: {csv_path_reviews}")
    print(f"File exists? {csv_path_reviews.exists()}")
    df_reviews = pd.read_csv(csv_path_reviews)
    # Drop any rows with missing review text or title
    df_reviews = df_reviews.dropna(subset=['title', 'review_text'])

    # Remove unnecessary whitespaces in text columns
    df_reviews['title'] = df_reviews['title'].str.strip()
    df_reviews['review_text'] = df_reviews['review_text'].str.strip()

    # Segment assignment function
    def assign_segment(row):
        # Combine title and review text into one string for analysis
        full_text = f"{row['title']} {row['review_text']}".lower()

        # Keywords for each category (define your own as necessary)
        keywords = {
            "youth": ["friends", "bros", "squad", "group", "hangout", "party", "social", "fun", "young", "teen", "college", "students"],
            "family": ["family", "kids", "children", "parents", "mom", "dad", "toddler", "baby", "grandparents"],
            "express_pass": ["express pass", "fast lane", "skip queue", "VIP", "no wait", "priority", "worth the money", "premium"],
            "budget": ["expensive", "cost", "too pricey", "overpriced", "save money", "cheap", "budget"]
        }

        # Score each category based on keyword matches
        scores = {category: sum(full_text.count(word) for word in words) for category, words in keywords.items()}

        # Determine the segment based on dominant categories
        age_group = "youth" if scores["youth"] > scores["family"] else "family"
        spending_type = "express_pass" if scores["express_pass"] > scores["budget"] else "budget"

        if age_group == "youth" and spending_type == "express_pass":
            return 0  # Social-Driven Youths
        elif age_group == "family" and spending_type == "budget":
            return 1  # Value-Conscious Families
        elif age_group == "youth" and spending_type == "budget":
            return 2  # Budget-Conscious Youths
        elif age_group == "family" and spending_type == "express_pass":
            return 3  # Premium Spenders
        else:
            return -1  # Unclassified

    # Apply the segmentation logic
    df_reviews['segment'] = df_reviews.apply(assign_segment, axis=1)
    # for rain
    def label_rain_reviews(df):
        weather_keywords = ["rain", "wet", "storm", "downpour", "drizzle", "thunder", "shower"]
        df["mentions_rain"] = df["review_text"].str.contains("|".join(weather_keywords), case=False, na=False)
        return df

    # Apply the rain labeling function
    df_reviews = label_rain_reviews(df_reviews)

    #month
    def categorize_by_month(df_reviews):
        df_reviews['written_date'] = pd.to_datetime(df_reviews['written_date'], errors='coerce')
        df_reviews['month'] = df_reviews['written_date'].dt.month_name()
        return df_reviews
    
    df_reviews = categorize_by_month(df_reviews)

    def categorize_season(month):
        if month in [5, 6, 7]:
            return 'Summer Holidays'
        elif month in [11, 12, 1]:
            return 'Winter Holidays'
        elif month in [2, 3, 4]:
            return 'Feb - Apr'
        else:
            return 'Aug - Oct'

    # Apply season categories
    df_reviews['season'] = df_reviews['written_date'].dt.month.apply(categorize_season)


    # Return cleaned data with segments
    return df_reviews
