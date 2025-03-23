## ----- Set up ----- ##
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from emoji import demojize
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats


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

# (2) Statistical analysis - understand whether event periods bring about statistically significant changes

# Metrics to analyse
metrics = {
    "Review Rating Change": "review_rating_change",
    "Review Polarity Change": "review_polarity_change",
    "Review Volume Change": "review_volume_change"
}

for metric_name, col in metrics.items():
    changes = df_change_data[col]

    # Check normality using Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(changes)
    print(f"Shapiro-Wilk Test for {metric_name}: p-value = {shapiro_p:.4f}")

    # Choose appropriate test based on normality
    if shapiro_p >= 0.05:
        # Normally distributed: use one-sample t-test
        test_stat, p_value = stats.ttest_1samp(changes, 0)
        test_type = "T-test"
    else:
        # Not normally distributed: Use Wilcoxon signed-rank test
        test_stat, p_value = stats.wilcoxon(changes)
        test_type = "Wilcoxon Test"

    # Print test results
    print(f"{metric_name} ({test_type}): Test-statistic = {test_stat:.4f}, P-value = {p_value:.4f}")

    # Print interpretation
    if p_value < 0.05:
        print(f"Significant change detected in {metric_name} before vs. during the event.\n")
    else:
        print(f"No significant change in {metric_name} before vs. during the event.\n")


"""
Analysis:
-------------------
There is no significant change in review volume, rating and polarity before vs during promotional events.

"""
##
"""
Data Visualisation: Plot the change data; Correlation matrix
-------------------
"""
# ------ Visualise correlation matrix using heatmap ------
correlation_matrix = df_change_data[['review_polarity_change', 'review_rating_change', 'review_volume_change']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, fmt='.2f', cbar=True)
plt.title('Correlation Matrix of Review Polarity Change, Rating Change, and Volume Change')
plt.tight_layout()
plt.show()

# ------ Visualise change using diverging bar plot ------

def plot_review_change(ax, change_data, diff_col):
    event_names = change_data.index.values
    avg_change = change_data[diff_col]

    df = pd.DataFrame({'event': event_names, 'difference': avg_change})
    df['color'] = df['difference'].apply(lambda x: 'red' if x < 0 else 'blue')

    ax.hlines(y=df.index, xmin=0, xmax=df['difference'], color=df['color'], alpha=0.7, linewidth=10)
    ax.set_yticks(df.index)
    ax.set_yticklabels(df['event'])
    ax.set_xlabel(diff_col)
    ax.set_title(f"{diff_col} before vs. during event periods")

# fig 1. Individual plot for review_volume_change
fig, ax = plt.subplots(figsize=(12, 8))
plot_review_change(ax, df_change_data, "review_volume_change")
plt.tight_layout()
plt.show()

# fig 2. Side-by-Side plot comparing review_polarity_change and review_rating_change
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
plot_review_change(axes[0], df_change_data, "review_rating_change")
plot_review_change(axes[1], df_change_data, "review_polarity_change")
plt.tight_layout()
plt.show()

"""
Analysis: 
-------------
Insights from correlation matrix:
1. Review Volume and Sentiment, Review Volume and Ratings:
    - Negligible negative correlation, which suggest that more reviews does not reflect guest satisfaction.
    - While an increase in review volume typically signals higher engagement and interest in the campaign, it is not 
    sufficient on its own to gauge guest satisfaction.
    - To assess a campaign’s true impact, other factors, such as review sentiment and ratings, should be considered.
    
2.  Ratings and Sentiment
    - Positive correlation, which indicates that higher ratings reflect better guest experience.
    - To strengthen brand perception during event periods, campaigns should aim to create memorable experiences that not 
    only generate reviews but also result in high ratings.
 

Insights from visualisations:
1. Plot of review_volume_change before vs during events:
    - Comparing to the plots on review_rating_change and review_polarity_change, there is no visible trend. This 
    aligns with our insights from the correlation matrix.

2. Side-by-Side Plot of review_rating_change and review_polarity_change:
    - Based on the plot, we can see that review_rating_change and review_polarity_change follow the same trend. This 
    aligns with our insights from the correlation matrix. 
"""

## --- Conclusion --- ##
"""
Conclusion: 
-------------
Key findings:
 - No significant change detected in review volume, rating and sentiment (polarity) before vs during promotional events
 periods at USS.
 - This suggests that promotional events by USS did not significantly affect review volume, ratings, or sentiment, 
 indicating that current marketing strategies may not be effectively influencing guest satisfaction during events.

In summary, for a campaign to improve guest satisfaction, USS should focus on fostering positive guest experiences rather
than traditional campaign success metrics like engagement rate etc.
"""

