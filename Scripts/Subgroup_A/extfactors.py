# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

# %%
#Import data
csv_path_segment = Path("../../data/survey_with_segments.csv").resolve()
print(f"Absolute CSV path: {csv_path_segment}")
print(f"File exists? {csv_path_segment.exists()}")
df_survey = pd.read_csv(csv_path_segment)

csv_path_reviews = Path("../../data/usstripadvisor.csv").resolve()
print(f"Absolute CSV path: {csv_path_reviews}")
print(f"File exists? {csv_path_reviews.exists()}")
df_reviews = pd.read_csv(csv_path_reviews)

csv_path_rain = Path("../../data/rainfall-monthly-total.csv").resolve()
print(f"Absolute CSV path: {csv_path_reviews}")
print(f"File exists? {csv_path_reviews.exists()}")
df_rain = pd.read_csv(csv_path_rain)

# %%
# Format date for reviews
def preprocess_reviews(df_reviews):
    df_reviews['written_date'] = pd.to_datetime(df_reviews['written_date'])
    df_reviews['visit_period'] = df_reviews['written_date'].dt.to_period('Q')
    return df_reviews

# Format date into generic quarters for reviews
def adjust_review_periods(df_reviews):
    df_reviews['generic_quarter'] = df_reviews['visit_period'].dt.quarter.map({
        1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"
    })
    return df_reviews

# Format date into visit_period(quarters) for survey
def map_survey_periods(df_survey):
    period_mapping = {
        "January - March": "Q1",
        "April - June": "Q2",
        "July - September": "Q3",
        "October - December": "Q4"
    }
    df_survey['visit_period'] = df_survey['Which part of the year did you visit USS?'].map(period_mapping)
    df_survey = df_survey.dropna(subset=['visit_period'])
    
    return df_survey

# Format date for rainfall
def preprocess_rainfall(df_rain):
    """Convert month to datetime and map it to quarters."""
    df_rain['month'] = pd.to_datetime(df_rain['month'])

    df_rain = df_rain[(df_rain['month'] >= '2009-12-01') & (df_rain['month'] <= '2019-12-31')]
    df_rain['quarter'] = df_rain['month'].dt.to_period('Q')
    
    # Convert period to string format (Q1, Q2, etc.)
    df_rain['quarter'] = df_rain['quarter'].dt.quarter.map({
        1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"
    })
    # Aggregate rainfall data by quarter
    quarterly_rainfall = df_rain.groupby('quarter')['total_rainfall'].mean().reset_index()
    return quarterly_rainfall


# %%
# Preprocess data
df_reviews = preprocess_reviews(df_reviews)
#print(df_reviews.head())
df_reviews = adjust_review_periods(df_reviews)
#print(df_reviews.head())
df_survey = map_survey_periods(df_survey)
# print(df_survey.head())
quarterly_rainfall = preprocess_rainfall(df_rain)
# print(quarterly_rainfall.head())

# %%
# Compare ratings with segments
def compare_ratings_with_segments(df_reviews, df_survey):
    # Get average reviews ratings by quarter
    reviews_avg = df_reviews.groupby('generic_quarter')['rating'].mean()
    
    # Get average survey ratings by quarter
    survey_avg = df_survey.groupby('visit_period')['experience_rating'].mean()

    # Segment-wise experience ratings (per quarter)
    segment_quarter_avg = df_survey.groupby(['visit_period', 'segment'])['experience_rating'].mean().unstack()

    # Combine reviews_avg and survey_avg into a comparison DataFrame
    comparison = pd.DataFrame({
        'Reviews_Avg': reviews_avg,
        'Survey_Avg': survey_avg
    })
    
    print("\nQuarterly Comparison of Reviews and Survey Ratings:")
    print(comparison)

    print("\nSegment-wise Average Ratings (by Quarter):")
    print(segment_quarter_avg)

    return comparison, segment_quarter_avg

# Perform
comparison, segment_quarter_avg = compare_ratings_with_segments(df_reviews, df_survey)

# %%
reviews_avg = comparison["Reviews_Avg"]
survey_avg = comparison["Survey_Avg"]

# Calculating Pearson correlation for significance
correlation, p_value = pearsonr(reviews_avg, survey_avg)

print(f"Pearson Correlation: {correlation:.2f}")
print(f"P-value: {p_value:.4f}")

# %% [markdown]
# Since the Pearson Correlation is negative, we cannot segment the tripadvisor reviews dataset with the survey dataset using quarters of the year.
# 
# Therefore, we only use the survey dataset for more analysis.

# %%
# Heatmap for segment-wise survey ratings by quarter
def plot_segment_quarter_comparison(segment_quarter_avg):
    plt.figure(figsize=(8, 6))
    sns.heatmap(segment_quarter_avg, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Average Rating'})
    plt.title('Segment-wise Average Ratings by Quarter')
    plt.xlabel('Segment')
    plt.ylabel('Quarter')
    plt.show()

# Call the plotting function
plot_segment_quarter_comparison(segment_quarter_avg)

# %%
# Merge rainfall data with survey data
print(df_survey)
def merge_rainfall_with_survey(df_survey, quarterly_rainfall):
    df_survey = df_survey.merge(quarterly_rainfall, left_on='visit_period', right_on='quarter', how='left')
    df_survey.drop(columns=['quarter'], inplace=True)  # Remove duplicate column
    
    return df_survey

# %%
# Call the function to merge rainfall data with survey data
df_survey = merge_rainfall_with_survey(df_survey, quarterly_rainfall)
print(df_survey.head())


# %%
# Double line graph for segment 0 with total rainfall for the four quarters
# Filter for segment 0
df_segment_0 = df_survey[df_survey['segment'] == 0]

# Group by visit_period and calculate the average for total_rainfall and experience_rating
df_segment_0_avg = df_segment_0.groupby('visit_period')[['total_rainfall', 'experience_rating']].mean().reset_index()

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line for total rainfall (left y-axis)
sns.lineplot(data=df_segment_0_avg, x='visit_period', y='total_rainfall', label='Total Rainfall', marker='o', color='blue', ax=ax1)

# Set labels and title for left y-axis
ax1.set_xlabel('Visit Period (Q1 to Q4)', fontsize=14)
ax1.set_ylabel('Total Rainfall (mm)', fontsize=14, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for experience rating (right y-axis)
ax2 = ax1.twinx()
ax2.set_ylim(0, 5)
sns.lineplot(data=df_segment_0_avg, x='visit_period', y='experience_rating', label='Experience Rating', marker='o', color='green', ax=ax2)

# Set labels for right y-axis
ax2.set_ylabel('Experience Rating', fontsize=14, color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Set plot title and adjust for readability
plt.title('Total Rainfall and Experience Rating for Segment 0 Guests Across Quarters', fontsize=16)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

# %%
# Double line graph for segment 1 with total rainfall for the four quarters
# Filter for segment 1
df_segment_1 = df_survey[df_survey['segment'] == 1]

# Group by visit_period and calculate the average for total_rainfall and experience_rating
df_segment_1_avg = df_segment_1.groupby('visit_period')[['total_rainfall', 'experience_rating']].mean().reset_index()

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line for total rainfall (left y-axis)
sns.lineplot(data=df_segment_1_avg, x='visit_period', y='total_rainfall', label='Total Rainfall', marker='o', color='blue', ax=ax1)

# Set labels and title for left y-axis
ax1.set_xlabel('Visit Period (Q1 to Q4)', fontsize=14)
ax1.set_ylabel('Total Rainfall (mm)', fontsize=14, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')


# Create a second y-axis for experience rating (right y-axis)
ax2 = ax1.twinx()
ax2.set_ylim(0, 5)
sns.lineplot(data=df_segment_1_avg, x='visit_period', y='experience_rating', label='Experience Rating', marker='o', color='green', ax=ax2)

# Set labels for right y-axis
ax2.set_ylabel('Experience Rating', fontsize=14, color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Set plot title and adjust for readability
plt.title('Total Rainfall and Experience Rating for Segment 1 Guests Across Quarters', fontsize=16)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

# %%
# Double line graph for segment 1 with total rainfall for the four quarters
# Filter for segment 2
df_segment_2 = df_survey[df_survey['segment'] == 2]

# Group by visit_period and calculate the average for total_rainfall and experience_rating
df_segment_2_avg = df_segment_2.groupby('visit_period')[['total_rainfall', 'experience_rating']].mean().reset_index()

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line for total rainfall (left y-axis)
sns.lineplot(data=df_segment_2_avg, x='visit_period', y='total_rainfall', label='Total Rainfall', marker='o', color='blue', ax=ax1)

# Set labels and title for left y-axis
ax1.set_xlabel('Visit Period (Q1 to Q4)', fontsize=14)
ax1.set_ylabel('Total Rainfall (mm)', fontsize=14, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for experience rating (right y-axis)
ax2 = ax1.twinx()
sns.lineplot(data=df_segment_2_avg, x='visit_period', y='experience_rating', label='Experience Rating', marker='o', color='green', ax=ax2)

# Set labels and limits for right y-axis
ax2.set_ylabel('Experience Rating', fontsize=14, color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, 5)  # Ensure rating axis is fixed from 0 to 5

# Set plot title and adjust for readability
plt.title('Total Rainfall and Experience Rating for Segment 2 Guests Across Quarters', fontsize=16)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

# %%
# Double line graph for segment 3 with total rainfall for the four quarters
# Filter for segment 3
df_segment_3 = df_survey[df_survey['segment'] == 3]

# Group by visit_period and calculate the average for total_rainfall and experience_rating
df_segment_3_avg = df_segment_3.groupby('visit_period')[['total_rainfall', 'experience_rating']].mean().reset_index()

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line for total rainfall (left y-axis)
sns.lineplot(data=df_segment_3_avg, x='visit_period', y='total_rainfall', label='Total Rainfall', marker='o', color='blue', ax=ax1)

# Set labels and title for left y-axis
ax1.set_xlabel('Visit Period (Q1 to Q4)', fontsize=14)
ax1.set_ylabel('Total Rainfall (mm)', fontsize=14, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for experience rating (right y-axis)
ax2 = ax1.twinx()
sns.lineplot(data=df_segment_3_avg, x='visit_period', y='experience_rating', label='Experience Rating', marker='o', color='green', ax=ax2)

# Set labels and limits for right y-axis
ax2.set_ylabel('Experience Rating', fontsize=14, color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, 5)  # Ensure rating axis is fixed from 0 to 5

# Set plot title and adjust for readability
plt.title('Total Rainfall and Experience Rating for Segment 3 Guests Across Quarters', fontsize=16)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

# %%

plt.figure(figsize=(10, 6))

# Scatter plot with different colors for each segment
sns.scatterplot(data=df_avg, x="total_rainfall", y="experience_rating", hue="segment", palette="Set1", style="segment")

# Regression lines for each segment
colors = ['red', 'blue', 'green', 'purple']
for i, segment in enumerate([0, 1, 2, 3]):
    df_segment = df_avg[df_avg["segment"] == segment]
    sns.regplot(data=df_segment, x="total_rainfall", y="experience_rating", scatter=False, color=colors[i], label=f'Segment {segment} Trend')

# Labels & title
plt.xlabel("Total Rainfall (mm)")
plt.ylabel("Experience Rating")
plt.title("Impact of Rainfall on Experience Rating Across Segments")

plt.legend(title="Segment & Trend", bbox_to_anchor=(1.05, 1), loc="upper left")  # Adjust legend placement
plt.show()

# %%
colors = ['red', 'blue', 'green', 'purple']

# Loop over each segment and plot
for i, segment in enumerate([0, 1, 2, 3]):
    # Filter the dataframe for the current segment
    df_segment = df_avg[df_avg["segment"] == segment]
    
    # Create a new figure for each segment
    plt.figure(figsize=(10, 6))

    # Scatter plot for the current segment
    sns.scatterplot(data=df_segment, x="total_rainfall", y="experience_rating", color=colors[i], label=f'Segment {segment}')
    
    # Regression line for the current segment
    sns.regplot(data=df_segment, x="total_rainfall", y="experience_rating", scatter=False, color=colors[i], label=f'Segment {segment} Trend')

    # Labels & title
    plt.xlabel("Total Rainfall (mm)")
    plt.ylabel("Experience Rating")
    plt.title(f"Impact of Rainfall on Experience Rating for Segment {segment}")
    
    # Add legend
    plt.legend(title="Trend", bbox_to_anchor=(1.05, 1), loc="upper left")  # Adjust legend placement
    
    # Show plot
    plt.show()

# %%
results = {}

# Loop through each segment and fit a linear regression model
for segment in [0, 1, 2, 3]:
    df_segment = df_avg[df_avg["segment"] == segment]
    
    # Define independent (X) and dependent (Y) variables
    X = df_segment["total_rainfall"]  # Predictor
    Y = df_segment["experience_rating"]  # Response
    
    # Add a constant to the model (intercept)
    X = sm.add_constant(X)
    
    ## Fit the linear regression model
    model = sm.OLS(Y, X).fit()
    
    # Store results
    results[segment] = model

    # Print summary
    print(f"\n--- Segment {segment} ---")
    print(model.summary())

# %% [markdown]
# Since the p-values are not significant, it suggests that rainfall does not have a strong enough statistical effect on experience ratings for these segments. However, the R-squared values indicate that rainfall still explains some variation in ratings, particularly negatively for Segment 2 (33.1%).
# 
# This shows that "Low Satisfaction Budget Guests" are the most affected by rain. 
# 
# Value-conscious families with children are likely the most affected by rain due to:
# 1. The disruption to outdoor attractions and play areas that children enjoy.
# 2. Comfort and convenience concerns for both children and parents.
# 3. Limited access to family-friendly amenities and attractions in the rain.
# 4. The higher expectations of value for money and potential dissatisfaction when the park experience doesn’t meet those expectations.
# 
# This may be due to the rain disrupting outdoor play areas and attractions, making them less enjoyable or inaccessible. For families with children, the rain might limit the park experience, especially when children are not able to engage in outdoor activities they find exciting. Parents might be more cautious about outdoor rides, especially if they are wet or slippery. Parents of young children might decide to avoid certain activities if the weather poses any safety risks, even if there are indoor alternatives.
# 
# 
# Poncho and Strollers: Parents with young children often bring strollers. In rainy conditions, strollers become harder to maneuver or may get wet, creating additional inconvenience. The need for rain gear for the entire family (umbrellas, ponchos) can add to the stress, especially if the park does not provide adequate shelter or rain protection. Moreover, for these value-conscious families, they might feel the cost of poncho for each family too much to spend on top of the admission tickets.
# 
# Higher Expectations for Value: As "value-conscious" guests, these families likely expect a lot of value for their money. Rain can lead to less value because outdoor attractions they were excited about might be inaccessible or less enjoyable, leading to dissatisfaction with the cost of the experience.

# %% [markdown]
# Possible operational adjustments:
# 1. Rainy Day Passes: Introducing a rainy-day pass or a flexible ticket that allows guests to return on a different day or provides discounts on future visits if their experience was negatively affected by weather. This would offer reassurance to value-conscious families that their visit can still be worthwhile even in bad weather.
# 2. Covered Walkways and Sheltered Areas: Ensure there are ample sheltered areas throughout the park where families can wait or relax without getting wet, particularly near popular attractions or dining areas. Indoor seating areas where families can rest and regroup would be a valuable addition.
# 3. Temporary Covered Playgrounds: Install temporary covered play structures or outdoor tents with engaging activities that can shield families from rain but still allow children to play and  explore.  


