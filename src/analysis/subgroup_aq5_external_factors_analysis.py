# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# %%
def q5_analyse(df_reviews_cleaned):
    # Bar plot of segment distribution
    def plotdistsegment(df_reviews_cleaned):
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df_reviews_cleaned["segment"])
        plt.xticks([0, 1, 2, 3], ["Social-Driven Youths", "Value-Conscious Families","Budget-Conscious Youths", "Premium Spenders" ], rotation=45, ha='right')
        plt.xlabel("Segment")
        plt.ylabel("Number of Reviews")
        plt.title("TripAdvisor Review Segments")
        plt.show()
    plotdistsegment(df_reviews_cleaned)


    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    df_reviews_cleaned["sentiment"] = df_reviews_cleaned["review_text"].apply(get_sentiment)

     # Boxplot of sentiment scores by segment
    def sentibyseg(df_reviews_cleaned):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df_reviews_cleaned["segment"], y=df_reviews_cleaned["sentiment"])
        plt.xticks([0, 1, 2, 3], ["Social-Driven Youths", "Value-Conscious Families","Budget-Conscious Youths", "Premium Spenders" ], rotation=45, ha='right')
        plt.xlabel("Segment")
        plt.ylabel("Sentiment Score")
        plt.title("Sentiment Analysis of TripAdvisor Reviews by Segment")
        plt.show()
    sentibyseg(df_reviews_cleaned)

    rain_reviews = df_reviews_cleaned[df_reviews_cleaned['mentions_rain'] == True]
    no_rain_reviews = df_reviews_cleaned[df_reviews_cleaned['mentions_rain'] == False]

    # Group by the 'segment' and calculate the average sentiment for reviews mentioning rain
    rain_segment_sentiment = rain_reviews.groupby('segment')['sentiment'].mean().reset_index()

    # Group by the 'segment' and calculate the average rating for reviews not mentioning rain
    no_rain_segment_sentiment = no_rain_reviews.groupby('segment')['sentiment'].mean().reset_index()

    # Merging the sentiment results for comparison
    sentiment_comparison = pd.merge(rain_segment_sentiment, no_rain_segment_sentiment, on='segment', suffixes=('_rain', '_no_rain'))
    print(sentiment_comparison)

    def sentimentrain(sentiment_comparison):
        # Melt the sentiment_comparison DataFrame to plot
        sentiment_melted = sentiment_comparison.melt(id_vars='segment', value_vars=['sentiment_rain', 'sentiment_no_rain'], 
                                                    var_name='weather_condition', value_name='sentiment')

        plt.figure(figsize=(10, 6))
        # Create a bar plot with 'weather_condition' as the hue to differentiate rain vs no rain
        sns.barplot(x='segment', y='sentiment', hue='weather_condition', data=sentiment_melted, palette=['blue', 'orange'])

        # Add labels and title
        plt.xticks([0, 1, 2, 3],["Social-Driven Youths", "Value-Conscious Families","Budget-Conscious Youths", "Premium Spenders" ])
        plt.xlabel('Segment')
        plt.ylabel('Average Sentiment Score')
        plt.title('Average Sentiment Scores by Segment: Rain vs No Rain')
        plt.legend(title='Weather Condition')
        plt.show()
    sentimentrain(sentiment_comparison)

    # Calculate the difference in sentiment between rain and no rain reviews for each segment
    sentiment_diff = sentiment_comparison.groupby('segment').apply(
        lambda x: x['sentiment_rain'].mean() - x['sentiment_no_rain'].mean()).reset_index(name='sentiment_diff')

    # Rename columns for clarity
    sentiment_diff.columns = ['segment', 'sentiment_diff']

    # Display the sentiment differences for each segment
    print(sentiment_diff)
    def sentimentdiffrain(sentiment_diff):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='segment', y='sentiment_diff', data=sentiment_diff)

        # Add labels and title
        plt.xticks([0, 1, 2, 3], ["Social-Driven Youths", "Value-Conscious Families","Budget-Conscious Youths", "Premium Spenders" ])

        plt.xlabel('Segment')
        plt.ylabel('Sentiment Difference (Rain - No Rain)')
        plt.title('Sentiment Difference by Segment (Rain vs No Rain)')
        plt.show()
    sentimentdiffrain(sentiment_diff)
    # Analysis:
    # Guests across all segments exhibited lower sentiment when keywords related to rain were mentioned in their reviews. This suggests that unfavorable weather conditions negatively impact the overall guest experience. Additionally, Segment 1 "Value-Conscious Families" and Segment 2 "Budget-Conscious Youths" showed the strongest decline in sentiment, indicating that weather disruptions may disproportionately affect guests who are more price-sensitive in their visits. The decline in sentiment could be attributed to fewer outdoor activities or discomfort caused by wet conditions.
    # 
    # Value-conscious families with children are the most affected by rain likely due to:
    # 1. The disruption to outdoor attractions and play areas that children enjoy.
    # 2. Comfort and convenience concerns for both children and parents.
    # 3. Limited access to family-friendly amenities and attractions in the rain.
    # 4. The higher expectations of value for money and potential dissatisfaction when the park experience doesn’t meet those expectations.
    # 
    # This may be due to the rain disrupting outdoor play areas and attractions, making them less enjoyable or inaccessible. For families with children, the rain might limit the park experience, especially when children are not able to engage in outdoor activities they find exciting. Parents might be more cautious about outdoor rides, especially if they are wet or slippery. Parents of young children might decide to avoid certain activities if the weather poses any safety risks, even if there are indoor alternatives.
    # 
    # Furthermore, parents with young children often bring strollers and in rainy conditions, strollers become harder to maneuver or may get wet, creating additional inconvenience. The need for rain gear for the entire family (umbrellas, ponchos) can add to the stress, especially if the park does not provide adequate shelter or rain protection. Moreover, for these value-conscious families, they might feel the cost of poncho for each family too much to spend on top of the admission tickets.
    # 
    # As "value-conscious" guests, these families likely expect a lot of value for their money. Rain can lead to less value because outdoor attractions they were excited about might be inaccessible or less enjoyable, leading to dissatisfaction with the cost of the experience.
    
    def sentiment_analysis_by_segment_month(df_reviews_cleaned):
        # Apply sentiment analysis to the review text for each row
        df_reviews_cleaned['sentiment'] = df_reviews_cleaned['review_text'].apply(get_sentiment)
        # Group by 'segment' and 'month' to get average sentiment per group
        sentiment_by_segment_month = df_reviews_cleaned.groupby(['segment', 'month'])['sentiment'].mean().reset_index()

        return sentiment_by_segment_month
    sentiment_by_segment_month = sentiment_analysis_by_segment_month(df_reviews_cleaned)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']

    # Convert the 'month' column to a categorical type with the correct order
    sentiment_by_segment_month['month'] = pd.Categorical(sentiment_by_segment_month['month'], categories=month_order, ordered=True)
    # Check the result
    print(sentiment_by_segment_month.head())

    # Function to plot sentiment by segment and month
    def plot_sentiment_by_segment_month(sentiment_by_segment_month):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='month', y='sentiment', hue='segment', data=sentiment_by_segment_month, marker='o')

        # Add labels and title
        plt.xlabel('Month')
        plt.ylabel('Average Sentiment Score')
        plt.title('Average Sentiment by Segment and Month')

        # Add legend
        plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Plot the sentiment data
    plot_sentiment_by_segment_month(sentiment_by_segment_month)

    # Function to plot sentiment by segment and month heatmpa
    def sentimonthheat(df_reviews_cleaned):
        # Pivot table for sentiment by segment and month
        pivot_sentiment = df_reviews_cleaned.pivot_table(values='sentiment', 
                                                index='segment', 
                                                columns='month', 
                                                aggfunc='mean')

        # Heatmap for sentiment
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_sentiment, annot=True, cmap='YlGnBu', fmt='.2f', cbar=True, 
                    xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        plt.title('Seasonal Sentiment by Segment')
        plt.xlabel('Month')
        plt.ylabel('Segment')
        plt.show()
    sentimonthheat(df_reviews_cleaned)

    # Group sentiment by season and segment
    seasonal_sentiment = df_reviews_cleaned.groupby(['season', 'segment'])['sentiment'].mean().reset_index()

    # Define the correct season order
    season_order = ['Feb - Apr', 'Summer Holidays', 'Aug - Oct', 'Winter Holidays']
    seasonal_sentiment['season'] = pd.Categorical(seasonal_sentiment['season'], categories=season_order, ordered=True)

    # Sort by season order
    seasonal_sentiment = seasonal_sentiment.sort_values('season')

    # Plot sentiment by season(line)
    def sentiseasonline(seasonal_sentiment):
        # Plot line graph
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='season', y='sentiment', hue='segment', data=seasonal_sentiment, marker='o')

        # Customize labels and title
        plt.xlabel('Season')
        plt.ylabel('Average Sentiment Score')
        plt.title('Sentiment Trends Across Seasons by Segment')
        plt.legend(title='Segment')
        plt.grid(True)

        plt.show()
    sentiseasonline(seasonal_sentiment)

    # Function to plot sentiment by segment for seasons
    def sentiseasonheat(df_reviews_cleaned):
        # Pivot table for sentiment by segment and season
        pivot_sentiment = df_reviews_cleaned.pivot_table(values='sentiment', 
                                                index='segment', 
                                                columns='season', 
                                                aggfunc='mean')
        season_order = ['Feb - Apr', 'Summer Holidays', 'Aug - Oct', 'Winter Holidays']
        pivot_sentiment = pivot_sentiment[season_order]
        # Heatmap for sentiment by season
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_sentiment, annot=True, cmap='YlGnBu', fmt='.2f', cbar=True)

        plt.title('Seasonal Sentiment by Segment')
        plt.xlabel('Season')
        plt.ylabel('Segment')
        plt.show()
    sentiseasonheat(df_reviews_cleaned)

    guest_volume = df_reviews_cleaned.groupby(['segment', 'month']).size().reset_index(name='guest_count')
    def guestvolmonth(guest_volume):
        # Plot the line graph
        plt.figure(figsize=(12, 6))

        # Create a line plot
        sns.lineplot(data=guest_volume, x='month', y='guest_count', hue='segment', marker='o', palette='tab10')

        # Titles and labels
        plt.title('Guest Volume by Segment and Month')
        plt.xlabel('Month')
        plt.ylabel('Guest Volume (Number of Reviews)')
        plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.legend(title='Segment', loc='upper left')

        # Show the plot
        plt.show()
    guestvolmonth(guest_volume)
    print(guest_volume.head())
    


    def guestvolseason(guest_volume):
        season_mapping = {
        'February': 'Feb-Apr', 'March': 'Feb-Apr', 'April': 'Feb-Apr',
        'May': 'Summer Holidays', 'June': 'Summer Holidays', 'July': 'Summer Holidays',
        'August': 'Aug-Oct', 'September': 'Aug-Oct', 'October': 'Aug-Oct',
        'November': 'Winter Holidays', 'December': 'Winter Holidays', 'January': 'Winter Holidays'
        }

        # Map months to seasons
        guest_volume['season'] = guest_volume['month'].map(season_mapping)

        # Create a duplicated dataframe where Feb-Apr appears again on the right
        guest_volume_dup = guest_volume[guest_volume['season'] == 'Feb-Apr'].copy()
        guest_volume_dup['season'] = 'Feb-Apr (2)'

        # Combine original with duplicated Feb-Apr
        guest_volume_extended = pd.concat([guest_volume, guest_volume_dup])

        # Maintain correct order
        season_order = ['Feb-Apr', 'Summer Holidays', 'Aug-Oct', 'Winter Holidays', 'Feb-Apr (2)']
        guest_volume_extended['season'] = pd.Categorical(guest_volume_extended['season'], categories=season_order, ordered=True)

        # Aggregate guest count by segment and season
        seasonal_guest_volume = guest_volume_extended.groupby(['segment', 'season'], observed=True)['guest_count'].sum().reset_index()
        return seasonal_guest_volume
    seasonal_guest_volume = guestvolseason(guest_volume)
    def plot_seasonal_volume(seasonal_guest_volume):
        # Plot the line graph
        plt.figure(figsize=(12, 6))

        # Create a line plot with reordered x-axis
        sns.lineplot(data=seasonal_guest_volume, x='season', y='guest_count', hue='segment', marker='o', 
                    palette=['blue', 'red', 'green', 'purple'])

        # Titles and labels
        plt.title('Guest Volume by Segment and Season (with Feb-Apr Comparison)')
        plt.xlabel('Season')
        plt.ylabel('Guest Volume (Number of Reviews)')
        plt.legend(title='Segment', loc='upper left')
        plt.grid(True)

        # Show the plot
        plt.show()
    plot_seasonal_volume(seasonal_guest_volume)
    # Analysis:
    # The trend of guest count across Budget-Conscious Youths,Premium Spenders, and Social-Driven Youths segments during the different periods remained fairly consistent. For Value-Conscious Families segment, the volume of guests in February to April and August to September periods were lower when compared to the higher volume seen in the summer (May to July) and winter (November to January) periods. 
    # 
    # This pattern is likely influenced by several factors, including seasonal availability, as families, particularly those with children, tend to visit more during school holidays in the summer and winter. Furthermore, the winter holiday period (November to January) aligns with family vacations and year-end celebrations, contributing to higher guest volume. 
    # 
    # While there are some fluctuations in the exact number of visitors across segments, the overall trend shows a clear drop in guest count in the low-season months (February to April and August to September). This reinforces the idea that external factors such as timing, weather, and events play a crucial role in driving guest attendance and satisfaction.
    # 
    # Overall, these trends suggest that Universal Studios experiences a cyclical pattern of higher and lower guest volumes for Value-Conscious Families segment based on the calendar year and external conditions, which could influence decisions related to staffing, promotions, and overall resource management.
    # Operational adjustments:
    # 1. Organize special events, festivals, or limited-time shows during off-peak periods to maintain high guest engagement. For example, consider hosting smaller-scale events, exclusive promotions, or theme nights tailored to specific segments.
    # 2. During months with lower guest volume, reduce staffing levels to maintain cost-efficiency. Conversely, during periods with peak attendance, ensure enough staff are available to handle the increased guest flow.
    # 3. For segments like Premium Spenders and Value-Conscious Families, offer loyalty programs or re-engagement incentives (e.g., discounts or priority access) to encourage repeat visits during quieter months.
    # 4. Send targeted emails with special offers or event notifications to guests who visited during peak times but might not typically visit during off-peak periods. These incentives could include discounts or exclusive offers to lure them back during quieter months.
