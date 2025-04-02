import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class guest_segmentation_model:
    """
    A class to perform guest segmentation using clustering techniques (KMeans, Hierarchical, DBSCAN).
    It handles evaluating clustering performance, summarizing cluster characteristics,
    and visualizing results.
    """

    def __init__(self, df_combined, df_labeled, scaled, pca):
        """
        Initialize the segmentation model with data.

        Parameters:
        - df_combined: Encoded and scaled dataframe used for clustering
        - df_labeled: Original labeled dataframe before encoding
        - scaled: Scaled numerical version of df_combined
        - pca: PCA-transformed features for dimensionality reduction
        """
        self.df_combined = df_combined
        self.df_labeled = df_labeled
        self.scaled = scaled
        self.pca = pca

    def determine_optimal_clusters(self, df_combined):
        """
        Use the Elbow Method to determine the optimal number of clusters based on SSE.
        """
        df_encoded = df_combined.copy()

        # Encode categorical columns
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

        # Standardize features
        scaled_features = StandardScaler().fit_transform(df_encoded)

        sse = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            sse.append(kmeans.inertia_)

        # Plot SSE to visualize the elbow
        plt.figure(figsize=(8, 4))
        plt.plot(range(2, 11), sse, marker='o')
        plt.title("Elbow Method: Choosing Optimal Number of Clusters (k)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return 4  # manually chosen from elbow curve

    def compare_models(self, pca):
        """
        Compare KMeans, Hierarchical, and DBSCAN clustering using silhouette scores.
        """
        results = []

        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        k_labels = kmeans.fit_predict(pca)
        k_score = silhouette_score(pca, k_labels)
        results.append(('KMeans', k_score, len(set(k_labels))))

        # Hierarchical Clustering
        h_labels = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(pca)
        h_score = silhouette_score(pca, h_labels)
        results.append(('Hierarchical (Ward)', h_score, len(set(h_labels))))

        # DBSCAN
        db = DBSCAN(eps=0.5, min_samples=10).fit(pca)
        db_labels = db.labels_
        valid_labels = db_labels[db_labels != -1]
        db_score = silhouette_score(pca[db_labels != -1], valid_labels) if len(set(valid_labels)) > 1 else 0
        results.append(('DBSCAN', db_score, len(set(valid_labels))))

        # Print comparison table
        print(pd.DataFrame(results, columns=["Model", "Silhouette Score", "Clusters Found"]))
        return k_labels

    def summarize_clusters(self, df_labeled, k_labels):
        """
        Summarize key characteristics of each cluster including majority attributes.
        """
        df_labeled['cluster'] = k_labels

        # Aggregate summary for each cluster
        summary = df_labeled.groupby('cluster').agg({
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

        summary['Avg Rating'] = summary['Avg Rating'].round(2)
        summary['Express Pass Usage %'] = summary['Express Pass Usage %'].round(1)

        # Rename clusters to meaningful names
        cluster_name_map = {
            0: 'Social-Driven Youths',
            1: 'Value-Conscious Families',
            2: 'Budget-Conscious Youths',
            3: 'Premium Spenders'
        }
        summary['Segment'] = summary.index.map(cluster_name_map)
        cols = ['Segment'] + [col for col in summary.columns if col != 'Segment']
        return summary[cols], df_labeled


    def visualize_clusters(self, df_labeled, scaled_features, k_labels):
        """
        Plot cluster results using PCA projection and cluster-specific distributions.
        """
        X_pca = PCA(n_components=2).fit_transform(scaled_features)

        # 2D PCA scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels, cmap='viridis', alpha=0.7)
        plt.title("Guest Segments Visualized in 2D (PCA)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Bar plot: Express Pass usage per cluster
        df_labeled['express_pass_numeric'] = df_labeled['express_pass'].map({'Yes': 1, 'No': 0})
        sns.barplot(x='cluster', y='express_pass_numeric', data=df_labeled, estimator=lambda x: sum(x)/len(x)*100)
        plt.ylabel("Express Pass Usage (%)")
        plt.title("Express Pass Usage by Cluster")
        plt.tight_layout()
        plt.show()

        # Count plots by group type and age
        df_labeled['group_type_simple'] = df_labeled['group_type'].str.split(',').str[0]
        sns.countplot(x='cluster', hue='group_type_simple', data=df_labeled)
        plt.title("Group Type by Cluster")
        plt.tight_layout()
        plt.show()

        ax = sns.countplot(x='cluster', hue='age_group', data=df_labeled)
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l != 'Option 2']
        filtered_handles, filtered_labels = zip(*filtered)
        ax.legend(filtered_handles, filtered_labels, title='age_group')
        plt.title("Age Group by Cluster")
        plt.tight_layout()
        plt.show()


    def run_pipeline(self):
        """
        Full pipeline: determines clusters, summarizes them, analyzes strategies, and visualizes.
        """
        optimal_k = self.determine_optimal_clusters(self.df_combined)
        k_labels = self.compare_models(self.pca)
        summary, df_labeled = self.summarize_clusters(self.df_labeled, k_labels)
        self.df_labeled = df_labeled
        self.visualize_clusters(df_labeled, self.scaled, k_labels)
        return summary, df_labeled

    # --- For Business Question 4: Impact of Marketing Strategies on Guest Behaviour --- #
    def analyze_marketing_strategies(self, df_labeled):
        """
        Identify top awareness source, ad response, and promotions for each segment.
        """
        cluster_name_map = {
            0: 'Social-Driven Youths',
            1: 'Value-Conscious Families',
            2: 'Budget-Conscious Youths',
            3: 'Premium Spenders'
        }

        def compute_top_response(df_labeled, cluster_col, response_col, cluster_map):
            top = df_labeled.groupby(cluster_col)[response_col].agg(lambda x: x.value_counts().idxmax()).reset_index()
            top[cluster_col] = top[cluster_col].map(cluster_map)
            return top

        print("--- Question 4, Part II: Recommend tailored marketing strategies for specific segments ---")
        # (1) Awareness: 'How did you first hear about Universal Studios Singapore?'
        top_awareness = compute_top_response(self.df_labeled, 'cluster', 'awareness', cluster_name_map)
        print("(1) Awareness: 'How did you first hear about Universal Studios Singapore?'")
        print(top_awareness)
        print("\n")

        # (2) Response to ads: 'Have you seen any recent advertisements or promotions for USS?'
        top_response_to_ads = compute_top_response(self.df_labeled, 'cluster', 'response_to_ads', cluster_name_map)
        print("(2) Response to ads: 'Have you seen any recent advertisements or promotions for USS?'")
        print(top_response_to_ads)
        print("\n")

        # (3) Preferred promotion: 'What type of promotions or discounts would encourage you to visit USS?'
        top_preferred_promotion = compute_top_response(self.df_labeled, 'cluster', 'preferred_promotion', cluster_name_map)
        print("(3) Preferred promotion: 'What type of promotions or discounts would encourage you to visit USS?'")
        print(top_preferred_promotion)

    def run_marketing_analysis(self):
        self.analyze_marketing_strategies(self.df_labeled)
