## ----- Set up: import required packages ----- ##
import pandas as pd
import os
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


class USSPromoAnalysis:
    def __init__(self, df_reviews, df_events):
        self.df_reviews = df_reviews
        self.df_events = df_events

    # --- Filter for reviews written during event period --- #
    def filter_reviews_during_events(self, branch='Universal Studios Singapore'):
        event_reviews_dict = {}
        for _, event in self.df_events.iterrows():
            event_key = f"{event['event']} ({event['start'].date()})"  # event as unique key
            start = event["start"]
            end = event["end"]
            filtered_reviews = self.df_reviews[
                (self.df_reviews["written_date"] >= start) &
                (self.df_reviews["written_date"] <= end) &
                (self.df_reviews["branch"] == branch)
                ]
            event_reviews_dict[event_key] = filtered_reviews
        return event_reviews_dict

    # --- Filter for reviews written before event period --- #
    def filter_reviews_before_events(self, branch='Universal Studios Singapore'):
        event_reviews_dict = {}
        for _, event in self.df_events.iterrows():
            event_key = f"{event['event']} ({event['start'].date()})"
            start = event["start"] - timedelta(days=event["duration"])
            end = event["start"] - timedelta(days=1)

            filtered_reviews = self.df_reviews[
                (self.df_reviews["written_date"] >= start) &
                (self.df_reviews["written_date"] <= end) &
                (self.df_reviews["branch"] == branch)
            ]

            event_reviews_dict[event_key] = filtered_reviews

        return event_reviews_dict

    # --- Compute difference in key metrics before event vs during event --- #
    def compute_change_in_reviews(self, reviews_before_event, reviews_during_event):
        changes = {}
        for event, reviews_before in reviews_before_event.items():
            reviews_during = reviews_during_event.get(event)
            if reviews_during is not None:
                # Ensure the reviews are DataFrames before performing calculations
                reviews_before = pd.DataFrame(reviews_before)
                reviews_during = pd.DataFrame(reviews_during)

                before_avg = reviews_before[['polarity', 'rating']].mean()
                during_avg = reviews_during[['polarity', 'rating']].mean()
                before_volume = len(reviews_before)
                during_volume = len(reviews_during)

                change = {
                    'review_polarity_change': during_avg['polarity'] - before_avg['polarity'],
                    'review_rating_change': during_avg['rating'] - before_avg['rating'],
                    'review_volume_change': during_volume - before_volume
                }
                changes[event] = change
        return pd.DataFrame.from_dict(changes, orient='index')

    # --- Statistical analysis - understand whether event periods bring about statistically significant changes --- #
    def metrics_analysis(self, df_change_data):
        # Metrics to analyze
        metrics = {
            "Review Rating Change": "review_rating_change",
            "Review Polarity Change": "review_polarity_change",
            "Review Volume Change": "review_volume_change"
        }

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

    # --- Data visualization --- #
    def visualize_review_changes(self, df_change_data):
        # (1) Correlation matrix 
        correlation_matrix = df_change_data[['review_polarity_change', 'review_rating_change', 'review_volume_change']].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, fmt='.2f', cbar=True)
        plt.title('Correlation Matrix of Review Polarity Change, Rating Change, and Volume Change')
        plt.tight_layout()
        plt.show()

        # (2) Diverging bar plot
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



    def run_promo_events_analysis(self):
        # (1) Filter for reviews written before and during events
        reviews_during_event = self.filter_reviews_during_events()
        reviews_before_event = self.filter_reviews_before_events()

        # (2) Compute changes in review polarity, volume and ratings 
        df_change_data = self.compute_change_in_reviews(reviews_before_event, reviews_during_event)

        # (3) Perform statistical analysis
        self.metrics_analysis(df_change_data)

        # (4) Visualize the changes in review polarity, volume and ratings
        self.visualize_review_changes(df_change_data)


if __name__ == "__main__":
    analysis = USSPromoAnalysis()
    analysis.run_promo_events_analysis()

