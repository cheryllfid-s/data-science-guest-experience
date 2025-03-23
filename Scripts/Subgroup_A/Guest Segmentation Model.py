
!pip install seaborn



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns






# Load data
survey = pd.read_csv("C:/Users/Admin/Downloads/survey.csv")  


# Step 1: Clean & Preprocess


# Rename columns for easier handling
survey = survey.rename(columns={
    'Which age group do you belong to?': 'age_group',
    'What is your employment status?': 'employment',
    'Who did you visit USS with?': 'group_type',
    'What was the main purpose of your visit?': 'visit_purpose',
    'On a scale of 1-5, how would you rate your overall experience at USS?': 'experience_rating',
    'Did you purchase the Express Pass?': 'express_pass',
    'How long did you wait in line for rides on average during your visit?': 'avg_wait_time',
    'Would you choose to revisit USS?': 'revisit',
    'Would you recommend USS to others?': 'recommend'
})

# Select relevant columns
selected_columns = [
    'age_group', 'employment', 'group_type', 'visit_purpose',
    'experience_rating', 'express_pass', 'avg_wait_time', 'revisit', 'recommend'
]
df_segment = survey[selected_columns].copy()

# Drop missing values
df_segment = df_segment.dropna()

# Encode categorical features
label_encoders = {}
for col in df_segment.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_segment[col] = le.fit_transform(df_segment[col])
    label_encoders[col] = le

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_segment)

# %%
# Determine Optimal Clusters (k)


# Elbow Method
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), sse, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()


##Compare Clustering Techniques


from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import pandas as pd

# Use normalized features from Step 3A
X = scaled_features

#  KMeans (again, for comparison)
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

#  Hierarchical Clustering (Ward)
linkage_matrix = linkage(X, method='ward')
hier_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')
hier_silhouette = silhouette_score(X, hier_labels)

#  DBSCAN
dbscan_model = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan_model.fit_predict(X)
dbscan_silhouette = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

# Compile results
comparison_df = pd.DataFrame({
    'Model': ['KMeans', 'Hierarchical (Ward)', 'DBSCAN'],
    'Silhouette Score': [kmeans_silhouette, hier_silhouette, dbscan_silhouette],
    'Clusters Found': [len(set(kmeans_labels)), len(set(hier_labels)), len(set(dbscan_labels))]
})

print(" Clustering Technique Comparison:")
display(comparison_df)





# markdown
# We evaluated three clustering techniques to segment guests based on survey responses:
# 
# 1. **KMeans** – Centroid-based algorithm ideal for structured, normalized data.
# 2. **Hierarchical Clustering (Ward)** – Linkage-based model that builds a nested tree of clusters.
# 3. **DBSCAN** – Density-based clustering that can find non-linear clusters and handle noise.
# 
# ####  Model Performance Comparison:
# 
# | Model               | Silhouette Score | Clusters Found |
# |---------------------|------------------|----------------|
# | KMeans              | 0.228            | 4              |
# | Hierarchical (Ward) | 0.229            | 4              |
# | DBSCAN              | 0.056            | 4              |
# 
# ---
# 
# ### Why We Chose KMeans:
# 
# - **KMeans** achieved a **high silhouette score** very close to that of Hierarchical, indicating strong internal cohesion.
# - It produced **balanced and interpretable clusters** that aligned well with our business objective of defining guest personas.
# - **Hierarchical Clustering**, although slightly stronger in score, lacks scalability for larger datasets and is harder to visualize in multi-dimensional space.
# - **DBSCAN** performed poorly with our data, producing weak separation between clusters.
# 
# ---
# 
# **Conclusion**: We chose **KMeans** for its combination of performance, scalability, and compatibility with downstream analysis like persona generation and marketing strategy development.
# 
# 


# Fit Final KMeans


# Based on elbow chart, let's choose k = 4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_segment['cluster'] = kmeans.fit_predict(scaled_features)


#Analyze and Summarize


# Add cluster labels back to original survey data
survey_with_segments = survey.loc[df_segment.index].copy()
survey_with_segments['segment'] = df_segment['cluster']

# Cluster feature summary
cluster_summary = df_segment.groupby('cluster').mean()

# Heatmap visualization
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_summary, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Guest Segmentation Cluster Summary")
plt.ylabel("Cluster")
plt.xlabel("Feature")
plt.show()


#Generate Guest Personas (Labels)


persona_definitions = {}

for cluster_id in range(optimal_k):
    segment_data = survey_with_segments[survey_with_segments['segment'] == cluster_id]
    persona_definitions[cluster_id] = {
        'Most Common Age Group': segment_data['age_group'].value_counts().idxmax(),
        'Most Common Group Type': segment_data['group_type'].value_counts().idxmax(),
        'Common Purpose': segment_data['visit_purpose'].value_counts().idxmax(),
        'Avg Experience Rating': round(segment_data['experience_rating'].mean(), 2),
        'Express Pass Usage %': round((segment_data['express_pass'] == 'Yes').mean() * 100, 1),
        'Cluster Size': len(segment_data)
    }

# Display personas
for cluster_id, persona in persona_definitions.items():
    print(f"\nCluster {cluster_id} – Persona:")
    for k, v in persona.items():
        print(f"{k}: {v}")


# Export Segmented Data to CSV


survey_with_segments.to_csv("survey_with_segments.csv", index=False)
print("\n Segmented survey data saved as 'survey_with_segments.csv'")


#  Visualize Segments (Personas)


# Pie Chart: Segment Sizes
plt.figure(figsize=(6, 6))
survey_with_segments['segment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='tab20')
plt.title("Segment Distribution (Pie Chart)")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Bar Chart: Age Group Distribution by Segment
plt.figure(figsize=(10, 6))
sns.countplot(data=survey_with_segments, x='age_group', hue='segment', palette='tab10')
plt.title("Age Group Distribution by Segment")
plt.xlabel("Age Group")
plt.ylabel("Number of Guests")
plt.xticks(rotation=45)
plt.legend(title='Segment')
plt.tight_layout()
plt.show()

# Express Pass Usage by Segment
plt.figure(figsize=(8, 6))
sns.countplot(data=survey_with_segments, x='express_pass', hue='segment', palette='Set2')
plt.title("Express Pass Usage by Segment")
plt.xlabel("Used Express Pass?")
plt.ylabel("Number of Guests")
plt.legend(title='Segment')
plt.tight_layout()
plt.show()

# Revisit Intention Heatmap
revisit_heatmap = pd.crosstab(survey_with_segments['segment'], survey_with_segments['revisit'], normalize='index')
plt.figure(figsize=(8, 4))
sns.heatmap(revisit_heatmap, annot=True, cmap='YlGnBu', fmt='.1%')
plt.title("Revisit Intention by Segment")
plt.xlabel("Would Revisit")
plt.ylabel("Segment")
plt.tight_layout()
plt.show()




# Generate Business Strategy Summary per Segment


# Recalculate persona definitions 
persona_definitions = {}

for cluster_id in range(optimal_k):
    segment_data = survey_with_segments[survey_with_segments['segment'] == cluster_id]
    persona_definitions[cluster_id] = {
        'Most Common Age Group': segment_data['age_group'].value_counts().idxmax(),
        'Most Common Group Type': segment_data['group_type'].value_counts().idxmax(),
        'Common Purpose': segment_data['visit_purpose'].value_counts().idxmax(),
        'Avg Experience Rating': round(segment_data['experience_rating'].mean(), 2),
        'Express Pass Usage %': round((segment_data['express_pass'] == 'Yes').mean() * 100, 1),
        'Cluster Size': len(segment_data)
    }

# Define marketing strategies based on persona traits
segment_recommendations = {}

for cluster_id, persona in persona_definitions.items():
    age_group = persona['Most Common Age Group']
    group_type = persona['Most Common Group Type']
    purpose = persona['Common Purpose']
    rating = persona['Avg Experience Rating']
    express_pct = persona['Express Pass Usage %']
    size = persona['Cluster Size']
    
    # Decide on segment label and marketing strategy
    if express_pct >= 90:
        tier = "Premium Spenders"
        strategy = "Upsell Express/VIP packages, offer loyalty perks for families, enhance high-end food options."
    elif express_pct <= 10 and rating < 3.5:
        tier = "Low Satisfaction Budget Guests"
        strategy = "Improve queue experience, offer re-engagement vouchers, enhance communication and ride information."
    elif "family" in group_type.lower():
        tier = "Value-Conscious Families"
        strategy = "Offer family meal combos, child-friendly attraction bundles, and off-peak promotions."
    else:
        tier = "Young Social Visitors"
        strategy = "Create group ticket bundles, influencer-driven social campaigns, and off-peak youth discounts."

    # Store strategy in dictionary
    segment_recommendations[cluster_id] = {
        "Segment Name": tier,
        "Age Group": age_group,
        "Group Type": group_type,
        "Visit Purpose": purpose,
        "Avg Experience Rating": rating,
        "Express Pass Usage %": express_pct,
        "Segment Size": size,
        "Marketing Strategy": strategy
    }

# Convert to DataFrame
segment_strategy_df = pd.DataFrame.from_dict(segment_recommendations, orient='index')

# Display the strategy table
print("\n Business Strategy Recommendations by Segment:")
display(segment_strategy_df)

## --- Business Question 4: Impact on Marketing Strategies on Guest Behaviour --- ##
"""
Recommend tailored marketing strategies for specific segments
-------------------
We will be analysing the responses of 3 key questions from the survey.
Q1. 'How did you first hear about Universal Studios Singapore?'
Q2. 'Have you seen any recent advertisements or promotions for USS?'
Q3. 'What type of promotions or discounts would encourage you to visit USS?'
"""

# (1) Generate tailored marketing strategies based on Q1

q1_col = 'How did you first hear about Universal Studios Singapore?'

# Define marketing strategies based on how guests first heard about USS
q1_segment_recommendations = {}

for cluster_id, persona in persona_definitions.items():
    age_group = persona['Most Common Age Group']
    group_type = persona['Most Common Group Type']
    purpose = persona['Common Purpose']
    rating = persona['Avg Experience Rating']
    express_pct = persona['Express Pass Usage %']
    size = persona['Cluster Size']

    # Get most common Q1 response for the segment
    segment_data = survey_with_segments[survey_with_segments['segment'] == cluster_id]
    primary_response = segment_data[q1_col].value_counts().idxmax()

    # Decide on segment label based on express pass usage
    if express_pct >= 90:
        tier = "Premium Spenders"
    elif express_pct <= 10 and rating < 3.5:
        tier = "Low Satisfaction Budget Guests"
    elif "family" in group_type.lower():
        tier = "Value-Conscious Families"
    else:
        tier = "Young Social Visitors"

    # Decide on marketing strategy based on Q1 response and segment traits
    if primary_response == 'Online ads':
        if express_pct >= 90:
            strategy = "Target premium digital ads, offer exclusive VIP packages online."
        elif "family" in group_type.lower():
            strategy = "Use online family-targeted ads with discounts for kids' attractions."
        else:
            strategy = "Social media influencer ads, exclusive digital coupons for young adults."

    elif primary_response == 'Word of mouth':
        if express_pct >= 90:
            strategy = "Leverage word-of-mouth for VIP and referral discounts, create loyalty programs."
        else:
            strategy = "Engage customers through referral programs and positive review sharing."

    elif primary_response == 'News':
        if express_pct >= 90:
            strategy = "Feature premium packages in high-end media outlets and news channels."
        else:
            strategy = "Use local news and media to engage families with promotional offers."

    elif primary_response == 'Social media':
        if express_pct >= 90:
            strategy = "Partner with influencers for VIP promotions and exclusive offers on social platforms."
        else:
            strategy = "Offer limited-time promotions through social media contests and influencer events."

    elif primary_response == 'Travel agencies/tour packages':
        if express_pct >= 90:
            strategy = "Collaborate with travel agencies to offer bundled premium packages."
        else:
            strategy = "Promote group tours and cost-effective travel packages through agencies."

    else:
        strategy = "Create tailored strategies based on the specific needs of the segment."

    # Store strategy and segment info in dictionary
    q1_segment_recommendations[cluster_id] = {
        "Segment Name": tier,
        "Age Group": age_group,
        "Group Type": group_type,
        "Visit Purpose": purpose,
        "Avg Experience Rating": rating,
        "Express Pass Usage %": express_pct,
        "Segment Size": size,
        "Primary Response": primary_response,
        "Marketing Strategy": strategy
    }

# Convert to dataframe
q1_segment_strategy_df = pd.DataFrame.from_dict(q1_segment_recommendations, orient='index')

# Display the strategy table for Q1 responses
print("\n Marketing Strategy Recommendations by Segment (Q1 Responses):")
display(q1_segment_strategy_df)


# (2) Generate tailored marketing strategies based on Q2

q2_col = 'Have you seen any recent advertisements or promotions for USS?'

# Define marketing strategies based on guests' exposure to USS ads
q2_segment_recommendations = {}

for cluster_id, persona in persona_definitions.items():
    age_group = persona['Most Common Age Group']
    group_type = persona['Most Common Group Type']
    purpose = persona['Common Purpose']
    rating = persona['Avg Experience Rating']
    express_pct = persona['Express Pass Usage %']
    size = persona['Cluster Size']

    # Get the most common primary Q2 response for the segment
    segment_data = q2_by_segment[q2_by_segment['segment'] == cluster_id]
    primary_response = segment_data[q2_col].value_counts().idxmax()

    # Decide on segment label based on express pass usage
    if express_pct >= 90:
        tier = "Premium Spenders"
    elif express_pct <= 10 and rating < 3.5:
        tier = "Low Satisfaction Budget Guests"
    elif "family" in group_type.lower():
        tier = "Value-Conscious Families"
    else:
        tier = "Young Social Visitors"

    # Tailor marketing strategies based on both tier and primary response
    if tier == "Premium Spenders":
        if primary_response == 'Yes, but they did not influence my decision':
            strategy = "Increase visibility through targeted promotions, reinforce brand presence."
        elif primary_response == 'Yes and they influenced my decision to visit':
            strategy = "Leverage this influence to promote upsell opportunities, loyalty programs."
        elif primary_response == "No, I haven't seen any ads":
            strategy = "Boost ad visibility and engagement across various channels, focus on awareness campaigns."

    elif tier == "Low Satisfaction Budget Guests":
        if primary_response == 'Yes, but they did not influence my decision':
            strategy = "Enhance advertisement engagement, focusing on the value proposition."
        elif primary_response == 'Yes and they influenced my decision to visit':
            strategy = "Emphasize offers and discounts, capitalize on ad-driven decisions."
        elif primary_response == "No, I haven't seen any ads":
            strategy = "Increase awareness, particularly focusing on affordable options."

    elif tier == "Value-Conscious Families":
        if primary_response == 'Yes, but they did not influence my decision':
            strategy = "Increase targeted family promotions in advertisements."
        elif primary_response == 'Yes and they influenced my decision to visit':
            strategy = "Create tailored family packages based on ad-driven decisions."
        elif primary_response == "No, I haven't seen any ads":
            strategy = "Boost family-oriented advertisements, increase ad presence in relevant channels."

    elif tier == "Young Social Visitors":
        if primary_response == 'Yes, but they did not influence my decision':
            strategy = "Increase ad presence and awareness, focusing on interactive content for young visitors."
        elif primary_response == 'Yes and they influenced my decision to visit':
            strategy = "Leverage social media-driven campaigns for youth engagement and promotions."
        elif primary_response == "No, I haven't seen any ads":
            strategy = "Focus on influencer marketing and social media engagement to drive awareness."

    # Store strategy and segment info in dictionary
    q2_segment_recommendations[cluster_id] = {
        "Segment Name": tier,
        "Age Group": age_group,
        "Group Type": group_type,
        "Visit Purpose": purpose,
        "Avg Experience Rating": rating,
        "Express Pass Usage %": express_pct,
        "Segment Size": size,
        "Primary Response": primary_response,
        "Marketing Strategy": strategy
    }

# Convert to DataFrame
q2_segment_strategy_df = pd.DataFrame.from_dict(q2_segment_recommendations, orient='index')

# Display the strategy table for Q2 responses
print("\n Marketing Strategy Recommendations by Segment (Q2 Responses):")
display(q2_segment_strategy_df)

# (3) Generate tailored marketing strategies based on Q3

q3_col = 'What type of promotions or discounts would encourage you to visit USS?'

# Define marketing strategies based on guests' preferred promotion type
q3_segment_recommendations = {}

for cluster_id, persona in persona_definitions.items():
    age_group = persona['Most Common Age Group']
    group_type = persona['Most Common Group Type']
    purpose = persona['Common Purpose']
    rating = persona['Avg Experience Rating']
    express_pct = persona['Express Pass Usage %']
    size = persona['Cluster Size']

    # Get most common primary Q3 response for the segment
    segment_data = q3_exploded[q3_exploded['segment'] == cluster_id]
    primary_response = segment_data[q3_col].value_counts().idxmax()

    # Decide on segment label based on express pass usage
    if express_pct >= 90:
        tier = "Premium Spenders"
    elif express_pct <= 10 and rating < 3.5:
        tier = "Low Satisfaction Budget Guests"
    elif "family" in group_type.lower():
        tier = "Value-Conscious Families"
    else:
        tier = "Young Social Visitors"

    # Tailor marketing strategies based on both tier and primary response
    if tier == "Premium Spenders":
        if primary_response == 'Discounted tickets':
            strategy = "Offer exclusive high-end discounts or access to VIP experiences."
        elif primary_response == 'Bundle deals (hotel + ticket packages etc.)':
            strategy = "Promote luxury bundle deals including premium accommodations and special perks."
        elif primary_response == 'Family/group discounts':
            strategy = "Provide premium family group packages with VIP options."
        elif primary_response == 'Seasonal event promotions':
            strategy = "Offer VIP access to seasonal events and exclusive premium seating."

    elif tier == "Low Satisfaction Budget Guests":
        if primary_response == 'Discounted tickets':
            strategy = "Offer affordable ticket discounts and emphasize value for money."
        elif primary_response == 'Bundle deals (hotel + ticket packages etc.)':
            strategy = "Promote cost-effective bundle deals with hotel stays for budget-conscious guests."
        elif primary_response == 'Family/group discounts':
            strategy = "Provide budget-friendly family group tickets with additional value-added services."
        elif primary_response == 'Seasonal event promotions':
            strategy = "Offer discounted seasonal events to attract budget-conscious visitors."

    elif tier == "Value-Conscious Families":
        if primary_response == 'Discounted tickets':
            strategy = "Offer family-friendly ticket discounts and emphasize convenience."
        elif primary_response == 'Bundle deals (hotel + ticket packages etc.)':
            strategy = "Promote family-friendly bundles, offering hotel and ticket combos for more savings."
        elif primary_response == 'Family/group discounts':
            strategy = "Provide exclusive family discounts for larger groups and multi-day visits."
        elif primary_response == 'Seasonal event promotions':
            strategy = "Create family-centric seasonal events with bundled deals."

    else:  # Young Social Visitors
        if primary_response == 'Discounted tickets':
            strategy = "Offer group discounts and emphasize social media perks."
        elif primary_response == 'Bundle deals (hotel + ticket packages etc.)':
            strategy = "Promote bundles that are ideal for young social visitors and friends."
        elif primary_response == 'Family/group discounts':
            strategy = "Offer group discounts for young visitors coming with friends."
        elif primary_response == 'Seasonal event promotions':
            strategy = "Leverage seasonal events for social media promotions and group discounts."

    # Store strategy and segment info in dictionary
    q3_segment_recommendations[cluster_id] = {
        "Segment Name": tier,
        "Age Group": age_group,
        "Group Type": group_type,
        "Visit Purpose": purpose,
        "Avg Experience Rating": rating,
        "Express Pass Usage %": express_pct,
        "Segment Size": size,
        "Primary Response": primary_response,
        "Marketing Strategy": strategy
    }

# Convert to DataFrame
q3_segment_strategy_df = pd.DataFrame.from_dict(q3_segment_recommendations, orient='index')

# Display the strategy table for Q3 responses
print("\n Marketing Strategy Recommendations by Segment (Q3 Responses):")
display(q3_segment_strategy_df)

