"""
Subgroup A
Business Question 4: Impact of Marketing Strategies on Guest Behaviour
(i) Analyze past campaign data to study changes in guest satisfaction
(ii) Recommend tailored marketing strategies for specific segments
"""

## ----- Set up: import required packages ----- ##
import pandas as pd
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def compute_change_in_reviews(reviews_before_event, reviews_during_event):
    """
    Compute the difference in key review metrics before and during an event.

    Parameters:
    - reviews_before_event (dict): key = event name, value = DataFrame containing review data before the event
    - reviews_during_event (dict): key = event name, value = DataFrame containing review data during the event

    Return:
    A DataFrame with each event as a row, with the following columns:
    - 'review_polarity_change': Change in average polarity score
    - 'review_rating_change': Change in average rating
    - 'review_volume_change': Change in the number of reviews
    """
    changes = {}
    print("Computing changes in review polarity, ratings and volume before vs during event...\n")

    # Iterate through each event, get corresponding reviews before event
    for event, reviews_before in reviews_before_event.items():
        reviews_during = reviews_during_event.get(event)       # Get corresponding reviews during event
        if reviews_during is not None:

            # Compute average polarity and rating before and during the event
            before_avg = reviews_before[['polarity', 'rating']].mean()
            during_avg = reviews_during[['polarity', 'rating']].mean()

            # Compute review volume (number of reviews) before and during the event
            before_volume = len(reviews_before)
            during_volume = len(reviews_during)

            # Compute the changes in key metrics
            change = {
                'review_polarity_change': during_avg['polarity'] - before_avg['polarity'],
                'review_rating_change': during_avg['rating'] - before_avg['rating'],
                'review_volume_change': during_volume - before_volume
            }
            # Store the computed changes for the current event in a dict
            changes[event] = change

    # Convert dict into dataframe for easier analysis
    changes_df = pd.DataFrame.from_dict(changes, orient='index')
    print("Dataframe containing changes in review polarity, ratings and volume:")
    print(changes_df)
    return changes_df


def visualize_review_changes(df_change_data):
    """
    Visualizes changes in review metrics (polarity, rating, and volume) before and during events.

    Parameters:
    - df_change_data (pd.DataFrame): A DataFrame with each event as a row, with the following columns:
        - 'review_polarity_change': Change in average polarity score
        - 'review_rating_change': Change in average rating
        - 'review_volume_change': Change in the number of reviews

    The function generates:
    1. A correlation matrix of the review metric changes
    2. A diverging bar plot for review volume change
    3. Side-by-side diverging bar plots to compare review polarity change and review rating change
    """

    # (1) Correlation matrix visualization
    correlation_matrix = df_change_data[
        ['review_polarity_change', 'review_rating_change', 'review_volume_change']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, fmt='.2f', cbar=True)
    plt.title('Correlation Matrix of Review Polarity Change, Rating Change, and Volume Change')
    plt.tight_layout()
    plt.show()

    # (2) Function to create diverging bar plots for review changes
    def plot_review_change(ax, df_change_data, diff_col):
        """
       Creates a horizontal diverging bar plot showing the difference in review metrics before and during an event.

       Parameters:
       - ax (matplotlib.axes.Axes): The subplot axis to draw the plot on
       - df_change_data (pd.DataFrame): Same as above, contains review change data
       - diff_col (str): The column name representing the review change metric
       """
        event_names = df_change_data.index.values  # Extract event names
        avg_change = df_change_data[diff_col]      # Extract the metric to be plotted

        # Create a dataframe for visualization
        df = pd.DataFrame({'event': event_names, 'difference': avg_change})

        # Assign colors: red for negative changes, blue for positive changes
        df['color'] = df['difference'].apply(lambda x: 'red' if x < 0 else 'blue')

        # Create horizontal bar lines to represent differences
        ax.hlines(y=df.index, xmin=0, xmax=df['difference'], color=df['color'], alpha=0.7, linewidth=10)
        ax.set_yticks(df.index)
        ax.set_yticklabels(df['event'])
        ax.set_xlabel(diff_col)
        ax.set_title(f"{diff_col} before vs. during event periods")

    # (3) Plot for review_volume_change
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_review_change(ax, df_change_data, "review_volume_change")
    plt.tight_layout()
    plt.show()

    # (4) Side-by-side plots for review_rating_change and review_polarity_change
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    plot_review_change(axes[0], df_change_data, "review_rating_change")
    plot_review_change(axes[1], df_change_data, "review_polarity_change")
    plt.tight_layout()
    plt.show()


def metrics_analysis(df_change_data):
    """
    Performs statistical analysis on review metric changes to determine if there is a
    significant difference before vs. during the event.

    Parameters:
    - df_change_data (pd.DataFrame): A DataFrame containing the following columns:
        - 'review_rating_change': Change in average rating
        - 'review_polarity_change': Change in average polarity score
        - 'review_volume_change': Change in the number of reviews

    The function:
    1. Tests normality of each metric using the Shapiro-Wilk test
    2. Performs an appropriate statistical test:
        - If normal: Uses a one-sample t-test (compares changes against 0).
        - If not normal: Uses a Wilcoxon signed-rank test (non-parametric alternative).
    """
    # Define the metrics to analyze (column names)
    metrics = {
        "Review Rating Change": "review_rating_change",
        "Review Polarity Change": "review_polarity_change",
        "Review Volume Change": "review_volume_change"
    }
    print("\nStatistical analysis of review metric changes:")
    # Loop through each metric and perform statistical tests
    for metric_name, col in metrics.items():
        changes = df_change_data[col]
        # (1) Check normality using Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(changes)
        print(f"Shapiro-Wilk Test for {metric_name}: p-value = {shapiro_p:.4f}")

        # (2) Choose appropriate test based on normality
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


class USSPromoAnalysis:
    """
    Analyzes the impact of promotional events on guest satisfaction at **Universal Studios Singapore (USS)**
    by examining changes in guest reviews before and during each event.
    Focusing on USS reviews since the events data contains USS-specific promotional events only.

    Parameters:
    - df_reviews (pd.DataFrame): DataFrame containing reviews with columns:
        - 'written_date' (datetime): Date the review was written
        - 'branch' (str): Universal Studios branch (Florida, Japan, **Singapore**)
        - 'rating' (float): Guest rating of the experience
        - 'polarity' (float): Sentiment polarity score of the review
        - other review-related columns (e.g., username, review text)

    - df_events (pd.DataFrame): DataFrame containing event details with columns:
        - 'event' (str): Event name
        - 'start' (datetime): Event start date
        - 'end' (datetime): Event end date
        - 'duration' (int): Event duration in days
        - other columns containing details about the promotion (e.g., type, specific discounts)

    Methods:
    - filter_reviews_during_events(): Filters reviews written during each event period
    - filter_reviews_before_events(): Filters reviews written before each event period
    - run_promo_events_analysis(): Runs the full promotional event analysis pipeline, including:
        1. Filtering reviews before and during events
        2. Computing changes in review metrics
        3. Performing statistical analysis
        4. Visualizing review changes
    """
    def __init__(self, df_reviews, df_events):
        """
        Initializes the analysis with review and event data
        """
        self.df_reviews = df_reviews
        self.df_events = df_events

    # --- Filter for reviews written during event period --- #
    def filter_reviews_during_events(self, branch='Universal Studios Singapore'):
        """
        Filters reviews written DURING each event period at the specified branch.

        Parameters:
        - branch (str): The Universal Studios branch to filter reviews for (default: USS)

        Returns:
        - dict: key = event name (appended with start date),
                value = DataFrames containing reviews written DURING the event
        """
        print("Filtering for reviews written during each event period...")
        event_reviews_dict = {}
        for _, event in self.df_events.iterrows():
            event_key = f"{event['event']} ({event['start'].date()})"  # event as unique key
            start = event["start"]
            end = event["end"]

            # Filter reviews within the event period
            filtered_reviews = self.df_reviews[
                (self.df_reviews["written_date"] >= start) &
                (self.df_reviews["written_date"] <= end) &
                (self.df_reviews["branch"] == branch)
                ]
            event_reviews_dict[event_key] = filtered_reviews
        return event_reviews_dict

    # --- Filter for reviews written before event period --- #
    def filter_reviews_before_events(self, branch='Universal Studios Singapore'):
        """
        Filters reviews written BEFORE each event period at the specified branch.

        Parameters:
        - branch (str): The Universal Studios branch to filter reviews for (default: USS)

        Returns:
        - dict: key = event name (appended with start date),
                value = DataFrames containing reviews written BEFORE the event
        """
        print("Filtering for reviews written before each event period...")
        event_reviews_dict = {}
        for _, event in self.df_events.iterrows():
            event_key = f"{event['event']} ({event['start'].date()})"   # event as unique key
            start = event["start"] - timedelta(days=event["duration"])  # before event period start
            end = event["start"] - timedelta(days=1)                    # one day before event starts

            # Filter reviews within the pre-event period
            filtered_reviews = self.df_reviews[
                (self.df_reviews["written_date"] >= start) &
                (self.df_reviews["written_date"] <= end) &
                (self.df_reviews["branch"] == branch)
            ]

            event_reviews_dict[event_key] = filtered_reviews

        return event_reviews_dict


    def run_promo_events_analysis(self):
        """
         Runs the full promotional event analysis pipeline.

         Steps:
         1. Filters reviews written before and during each event.
         2. Computes changes in review polarity, volume, and ratings.
         3. Performs statistical analysis to check significance.
         4. Visualizes the changes in review metrics.

         The results will be printed and visualized.
         """
        print("Analysis of USS' past promotional events data to study changes in guest satisfaction:\n")
        # (1) Filter for reviews written before and during events
        reviews_during_event = self.filter_reviews_during_events()
        reviews_before_event = self.filter_reviews_before_events()

        # (2) Compute changes in review polarity, volume and ratings 
        df_change_data = compute_change_in_reviews(reviews_before_event, reviews_during_event)

        # (3) Perform statistical analysis
        metrics_analysis(df_change_data)

        # (4) Visualize the changes in review polarity, volume and ratings
        visualize_review_changes(df_change_data)


if __name__ == "__main__":
    analysis = USSPromoAnalysis()
    analysis.run_promo_events_analysis()

