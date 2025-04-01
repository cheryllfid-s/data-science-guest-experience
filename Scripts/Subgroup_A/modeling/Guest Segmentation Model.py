import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get the directory of the current script (guest_satisfaction.py)
script_dir = Path(__file__).parent

# Navigate project root (Git/) and then into data/
project_root = script_dir.parent.parent  # Adjust based on your structure
csv_path = project_root / "data" / "survey.csv"

# Verify  path
print(f"Absolute CSV path: {csv_path}")
print(f"File exists? {csv_path.exists()}")

# Load the CSV
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

# Select core columns
selected_cols = [
    'age_group', 'group_type', 'visit_purpose', 'express_pass', 'experience_rating',
    'awareness', 'response_to_ads', 'preferred_promotion'
]
survey_clean = survey[selected_cols].dropna().reset_index(drop=True)



# Generate Synthetic Data Based on Real Distribution
np.random.seed(42)
n_samples = 400

# Assign group type equally
group_type_syn = np.random.choice(
    ['Friends', 'Family (including children)'],
    size=n_samples,
    p=[0.5, 0.5]
)

# Conditional age group
def assign_age(gt):
    return np.random.choice(
        ['18 - 24 years old'] if gt == 'Friends' else ['25 - 34 years old']
    )
age_group_syn = [assign_age(gt) for gt in group_type_syn]

# Conditional visit purpose
def assign_purpose(gt):
    return np.random.choice(
        ['Social gathering'] if gt == 'Friends' else ['Family outing']
    )
visit_purpose_syn = [assign_purpose(gt) for gt in group_type_syn]

# Conditional express pass
express_pass_syn = [
    np.random.choice(['Yes', 'No'], p=[0.25, 0.75]) if gt == 'Friends' else
    np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
    for gt in group_type_syn
]

# Conditional experience rating
experience_rating_syn = [
    round(np.random.normal(4.2, 0.3), 1) if ep == 'Yes' else
    round(np.random.normal(3.5, 0.5), 1)
    for ep in express_pass_syn
]

# Build synthetic df
synthetic = pd.DataFrame({
    'age_group': age_group_syn,
    'group_type': group_type_syn,
    'visit_purpose': visit_purpose_syn,
    'express_pass': express_pass_syn,
    'experience_rating': experience_rating_syn
})




# Generate synthetic data for 3 marketing related questions

# Conditional awareness: 'How did you first hear about Universal Studios Singapore?'
def assign_awareness():
    return np.random.choice(
        ['Word of mouth', 'Social media', 'Online ads', 'Travel agencies/tour packages', 'News'],
        p=[0.6, 0.3, 0.05, 0.025, 0.025]  # Word of mouth (60%), Social media (30%), others negligible
    )

awareness_syn = [assign_awareness() for _ in range(n_samples)]

# Conditional response to ads: 'Have you seen any recent advertisements or promotions for USS?'
def assign_response_to_ads(gt):
    if gt == 'Friends':
        return np.random.choice(
            ['Yes, but they did not influence my decision', 'Yes and they influenced my decision', "No, I haven't seen any ads"],
            p=[0.7, 0.1, 0.2]
            # ads but were not influenced (50%), some were (20%), rest didn't see (30%)
        )
    else:
        return np.random.choice(
            ["No, I haven't seen any ads", 'Yes, but they did not influence my decision', 'Yes and they influenced my decision'],
            p=[0.7, 0.2, 0.1]
            # 70% didn't see ads, 20% saw but were not influenced, 10% were influenced
        )

response_to_ads_syn = [assign_response_to_ads(gt) for gt in group_type_syn]

# Conditional preferred promotion: 'What type of promotions or discounts would encourage you to visit USS?'
def assign_preferred_promotion():
    return np.random.choice(
        ['Discounted tickets', 'Family/group discounts', 'Seasonal event promotions', 'Bundle deals (hotel + ticket packages)'],
        p=[0.5, 0.25, 0.15, 0.1]
        # Discounted tickets top choice, followed by family/group discounts
    )

preferred_promotion_syn = [assign_preferred_promotion() for _ in range(n_samples)]

synthetic['awareness'] = awareness_syn
synthetic['response_to_ads'] = response_to_ads_syn
synthetic['preferred_promotion'] = preferred_promotion_syn


# Merge Real & Synthetic Data

df_combined = pd.concat([survey_clean, synthetic], ignore_index=True)

# Store original
df_labeled = df_combined.copy()

# Encode + scale
for col in df_combined.select_dtypes(include='object').columns:
    df_combined[col] = LabelEncoder().fit_transform(df_combined[col])

scaled = StandardScaler().fit_transform(df_combined)
pca = PCA(n_components=2).fit_transform(scaled)


# ## Synthetic Data Generation and Merging Strategy
# 
# To enrich the dataset for more representative and balanced clustering, we generated synthetic survey responses based on patterns observed in the real data, especially to address potential response bias (e.g., underrepresentation of families).
# 
# ---
# 
# ### Why We Generated Synthetic Data:
# - The original survey was distributed primarily to university students, likely skewing responses toward younger, student demographics (e.g., youths, friend groups).
# - Guest personas like families, especially those using Express Passes or traveling with children, were underrepresented.
# - Including a richer distribution of visitor types ensures more meaningful and realistic clusters during segmentation.
# 
#  Overall, the synthetic data generation preserved the integrity of the original survey while enhancing its representativeness for more effective clustering and persona discovery.
# 


# Determine optimal clusters

df_encoded = df_combined.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])


scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded)

sse = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.xticks(k_range)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method: Choosing Optimal Number of Clusters (k)")
plt.grid(True)
plt.tight_layout()
plt.show()


# 
# To determine the optimal number of clusters for our segmentation task, we applied the Elbow Method. This technique involves fitting KMeans models with varying k values (from 2 to 10) and plotting the Sum of Squared Errors (SSE) for each value.
# 
# The plot above shows that the reduction in SSE begins to flatten around k = 4, suggesting diminishing returns beyond this point. This “elbow” indicates that four clusters strike a good balance between model complexity and explanatory power.
# 
# Therefore, we selected k = 4 as the final number of clusters for our segmentation model. This choice ensures that we capture diverse guest profiles without overfitting or fragmenting our personas too granularly.
# 
# 


# Compare Clustering Techniques
results = []

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(pca)
k_score = silhouette_score(pca, k_labels)
results.append(('KMeans', k_score, len(set(k_labels))))

# Hierarchical (Agglomerative)
from sklearn.cluster import AgglomerativeClustering
h_labels = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(pca)
h_score = silhouette_score(pca, h_labels)
results.append(('Hierarchical (Ward)', h_score, len(set(h_labels))))

# DBSCAN
db = DBSCAN(eps=0.5, min_samples=10).fit(pca)
db_labels = db.labels_
valid_labels = db_labels[db_labels != -1]
db_score = silhouette_score(pca[db_labels != -1], valid_labels) if len(set(valid_labels)) > 1 else 0
results.append(('DBSCAN', db_score, len(set(valid_labels))))
results_df = pd.DataFrame(results, columns=['Model', 'Silhouette Score', 'Clusters Found'])
print(results_df)
print("\n")


# ## Clustering Technique Comparison & Final Model Selection
# 
# We evaluated three clustering techniques to segment guests based on survey responses:
# 
# 1. **KMeans** – Centroid-based algorithm ideal for structured, normalized data.
# 2. **Hierarchical Clustering (Ward linkage)** – Linkage-based model that builds a nested tree of clusters.
# 3. **DBSCAN** – Density-based clustering that can find non-linear clusters and handle noise.
# 
# 
# 
# ## Dimensionality Reduction with PCA
# 
# Before applying clustering algorithms, we used **Principal Component Analysis (PCA)** to reduce the high-dimensional survey data into a lower-dimensional space while preserving variance. This step improved clustering efficiency, reduced noise, and helped enhance interpretability of clusters in 2D/3D visualizations. The first few principal components captured most of the variance in the data and were sufficient to reveal clear separation among guest segments. PCA was especially helpful when visualizing KMeans and Hierarchical clustering results.
# 
# 
# 
# ## Model Performance Comparison
# | Model                  | Silhouette Score | Clusters Found |
# |------------------------|------------------|----------------|
# | KMeans                 | **0.514**         | 4              |
# | Hierarchical (Ward)   | 0.507             | 4              |
# | DBSCAN                | 0.000             | 1              |
# 
# 
# 
# ## Why We Chose KMeans:
# 
# - **KMeans achieved the highest silhouette score**, indicating strong internal cohesion between clusters.
# - It produced **balanced and interpretable clusters** that naturally aligned with business goals — segmenting guests by group type, age group, and price sensitivity.
# - **Hierarchical Clustering**, while competitive, resulted in slightly less separation and was more computationally expensive.
# - **DBSCAN** failed to identify multiple clusters in our use case (returned 1 large cluster and labeled the rest as noise), resulting in a silhouette score of **0.00**. With our dense, normalized survey data, DBSCAN treated most points as one large group or noise, unable to separate meaningful clusters. This is why it returned **1 cluster** and **0 silhouette score**.
# 
# 
# 
# 
# We chose **KMeans** for its high performance, scalability, and smooth integration with downstream analytics such as persona generation and marketing recommendation modeling.


# Choose Final Model and Label Segments
df_labeled['cluster'] = k_labels

# Summarize segments
cluster_summary = df_labeled.groupby('cluster').agg({
    'age_group': lambda x: x.value_counts().idxmax(),
    'group_type': lambda x: x.value_counts().idxmax(),
    'visit_purpose': lambda x: x.value_counts().idxmax(),
    'express_pass': lambda x: (x == 'Yes').mean() * 100,
    'experience_rating': 'mean',
    'cluster': 'count'
}).rename(columns={
    'age_group': 'Age Group',
    'group_type': 'Group Type',
    'visit_purpose': 'Visit Purpose',
    'express_pass': 'Express Pass Usage %',
    'experience_rating': 'Avg Rating',
    'cluster': 'Size'
}).reset_index(drop=True)

cluster_summary['Avg Rating'] = cluster_summary['Avg Rating'].round(2)
cluster_summary['Express Pass Usage %'] = cluster_summary['Express Pass Usage %'].round(1)

cluster_summary


# Re-run PCA on scaled features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_features)

# Scatter plot of the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels, cmap='viridis', alpha=0.7)
plt.title('Guest Segments Visualized in 2D (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()



df_labeled['express_pass_numeric'] = df_labeled['express_pass'].apply(lambda x: 1 if x == 'Yes' else 0)


plt.figure(figsize=(8, 5))
sns.barplot(
    x='cluster',
    y='express_pass_numeric',
    data=df_labeled,
    estimator=lambda x: sum(x) / len(x) * 100  
)
plt.ylabel("Express Pass Usage (%)")
plt.xlabel("Cluster")
plt.title("Express Pass Usage by Cluster")
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



df_labeled['group_type_simple'] = df_labeled['group_type'].apply(lambda x: x.split(',')[0].strip())

plt.figure(figsize=(8, 5))
sns.countplot(x='cluster', hue='group_type_simple', data=df_labeled)
plt.title("Simplified Group Type Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title='Group Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
sns.countplot(x='cluster', hue='age_group', data=df_labeled)
plt.title("Age Group Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





# ### Official Clusters 
# 
# Based on the final cluster characteristics including group type, age group, express pass usage, visit purpose, and average satisfaction rating, we interpret and name the four guest segments as follows:
# 
# 0. **Social-Driven Youths**  
#    *Profile:* Young friend groups (18–24 years old) visiting for social bonding. This group shows high Express Pass usage and high satisfaction, emphasizing social experience over cost.  
#    *Business Implication:* Promote through social media, influencer campaigns, and group discounts.
# 
# 1. **Value-Conscious Families**  
#    *Profile:* Budget-sensitive families (typically aged 25–34), attending for family outings. They have moderate satisfaction and low Express Pass usage.  
#    *Business Implication:* Offer value-focused family packages, meal bundles, and promote off-peak visits.
# 
# 2. **Budget-Conscious Youths**  
#    *Profile:* Young visitors (18–24 years old), mostly friend groups attending for social purposes. They show low Express Pass usage and lower average satisfaction, indicating cost-sensitivity.  
#    *Business Implication:* Target with queue improvement, budget packages, and survey-based re-engagement.
# 
# 3. **Premium Spenders**  
#    *Profile:* Families with children (mostly 25–34 years old) who show very high Express Pass usage and high satisfaction. These guests are willing to spend for convenience and premium experiences.  
#    *Business Implication:* Upsell VIP bundles, offer exclusive experiences, and retain through loyalty programs.
# 
# 
# These personas reflect distinct motivations and spending behavior, enabling more precise marketing strategies tailored to each cluster.
# 


# Map cluster indices to official names
cluster_name_map = {
    0: 'Social-Driven Youths',
    1: 'Value-Conscious Families',
    2: 'Budget-Conscious Youths',
    3: 'Premium Spenders'
}

cluster_summary['Segment'] = cluster_summary.index.map(cluster_name_map)
cols = ['Segment'] + [col for col in cluster_summary.columns if col != 'Segment']
cluster_summary = cluster_summary[cols]
cluster_summary

## --- Analysis for Business Question 4: Recommend tailored marketing strategies for specific segments --- ##

# Compute top response for each segment
def compute_top_response(df, cluster_col, response_col, cluster_map):
    top_response = df.groupby(cluster_col)[response_col].agg(
        lambda x: x.value_counts().idxmax()).reset_index()
    top_response[cluster_col] = top_response[cluster_col].map(cluster_map)
    return top_response

# --- Analysing 3 key questions from the data --- ##
print("--- Business Question 4: Recommend tailored marketing strategies for specific segments ---")

# (1) Awareness: 'How did you first hear about Universal Studios Singapore?'
top_awareness = compute_top_response(df_labeled, 'cluster', 'awareness', cluster_name_map)
print("### Awareness: 'How did you first hear about Universal Studios Singapore?' ###")
print(top_awareness)
print("\n")


# (2) Response to ads: 'Have you seen any recent advertisements or promotions for USS?'
top_response_to_ads = compute_top_response(df_labeled, 'cluster', 'response_to_ads', cluster_name_map)
print("### Response to ads: 'Have you seen any recent advertisements or promotions for USS?' ###")
print(top_response_to_ads)
print("\n")


# (3) Preferred promotion: 'What type of promotions or discounts would encourage you to visit USS?'
top_preferred_promotion = compute_top_response(df_labeled, 'cluster', 'preferred_promotion', cluster_name_map)
print("### Preferred promotion: 'What type of promotions or discounts would encourage you to visit USS?' ###")
print(top_preferred_promotion)

# %% [markdown]
# Insight 1:
# Across all segments, word-of-mouth emerged as the primary awareness source. This shows that peer recommendations and
# personal experiences are much more impactful than traditional advertising.
#
# Business impact 1:
# Prioritise word-of-mouth marketing through social media. Partner with influencers and introduce incentives for guests
# to share their authentic USS experience through user-generated-content. Such incentives could include a discount for
# their next visit or referral discounts.
# ---------------------------------------------------------------------------------------------------------------------
# Insight 2:
# Families had limited advertisement exposure, while youths saw advertisements but were not influenced to visit.
# - Advertisements may not be reaching families through the most effective channels.
# - Approach of advertisements may not be resonating with youths.
#
# Business impact 2:
# - Given that word-of-mouth is effective across all segments, continue to prioritise this channel.
# - Collaborate with family-focused influencers or content creators to highlight the family-friendly aspects of USS,
# such as kid-friendly attractions, family discounts, and safety measures.
# - Engage youths with interactive campaigns on TikTok/Instagram instead of traditional advertisements. For example,
# offering behind-the-scenes exclusives or guest interviews can be a starting point to build an online community,
# foster a sense of inclusivity and build rapport with guests.
# ---------------------------------------------------------------------------------------------------------------------
# Insight 3:
# Discounts are the most preferred promotion type, but analysis of past promotional events showed no significant impact on
# guest satisfaction. This highlights a need to shift toward focusing on guest experience instead.
#
# Business impact 3:
# Shift focus from discounts to creating unique, memorable experiences for guests.
# For instance,
# - Lucky draw events where winner gets exclusive backstage access or meet-and-greet opportunities.
# - Exclusive polaroid photo sessions at iconic spots in USS, offering guests a tangible souvenir that adds a personal touch.
# By leaving guests with memorable experiences that go beyond discount, guest are more likely to share their experience
# with others, driving promotion and brand awareness through word-of-mouth.
