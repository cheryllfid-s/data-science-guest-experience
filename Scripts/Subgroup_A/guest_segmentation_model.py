import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class guest_segmentation_model:
    """
    Guest Segmentation Model Pipeline

    This class encapsulates the end-to-end process for preparing survey data, generating synthetic guests,
    running PCA and clustering algorithms, and providing segmentation insights and tailored marketing strategy analysis.
    """

    def __init__(self):
        """Initialize project paths and locate the CSV file."""
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent
        self.csv_path = self.project_root / "data" / "survey.csv"
        print(f"Absolute CSV path: {self.csv_path}")
        print(f"File exists? {self.csv_path.exists()}")

    def load_and_clean_data(self):
        """Load the survey CSV and clean/rename columns relevant for clustering."""
        survey = pd.read_csv(self.csv_path)
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
        return survey_clean

    def generate_synthetic_data(self, n_samples=400):
        """Generate a balanced synthetic dataset of guest profiles for clustering."""
        np.random.seed(42)
        group_type_syn = np.random.choice(
            ['Friends', 'Family (including children)'],
            size=n_samples,
            p=[0.5, 0.5]
        )

        def assign_age(gt):
            return np.random.choice(
                ['18 - 24 years old'] if gt == 'Friends' else ['25 - 34 years old']
            )
        age_group_syn = [assign_age(gt) for gt in group_type_syn]

        def assign_purpose(gt):
            return np.random.choice(
                ['Social gathering'] if gt == 'Friends' else ['Family outing']
            )
        visit_purpose_syn = [assign_purpose(gt) for gt in group_type_syn]

        express_pass_syn = [
            np.random.choice(['Yes', 'No'], p=[0.25, 0.75]) if gt == 'Friends' else
            np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
            for gt in group_type_syn
        ]

        experience_rating_syn = [
            round(np.random.normal(4.2, 0.3), 1) if ep == 'Yes' else
            round(np.random.normal(3.5, 0.5), 1)
            for ep in express_pass_syn
        ]

        def assign_awareness():
            return np.random.choice(
                ['Word of mouth', 'Social media', 'Online ads', 'Travel agencies/tour packages', 'News'],
                p=[0.6, 0.3, 0.05, 0.025, 0.025]
            )
        awareness_syn = [assign_awareness() for _ in range(n_samples)]

        def assign_response_to_ads(gt):
            if gt == 'Friends':
                return np.random.choice(
                    ['Yes, but they did not influence my decision', 'Yes and they influenced my decision', "No, I haven't seen any ads"],
                    p=[0.7, 0.1, 0.2]
                )
            else:
                return np.random.choice(
                    ["No, I haven't seen any ads", 'Yes, but they did not influence my decision', 'Yes and they influenced my decision'],
                    p=[0.7, 0.2, 0.1]
                )
        response_to_ads_syn = [assign_response_to_ads(gt) for gt in group_type_syn]

        def assign_preferred_promotion():
            return np.random.choice(
                ['Discounted tickets', 'Family/group discounts', 'Seasonal event promotions', 'Bundle deals (hotel + ticket packages)'],
                p=[0.5, 0.25, 0.15, 0.1]
            )
        preferred_promotion_syn = [assign_preferred_promotion() for _ in range(n_samples)]

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

        return synthetic

    def merge_and_encode(self, survey_clean, synthetic):
        """Merge real and synthetic data, encode categorical fields, and scale features for PCA."""
        df_combined = pd.concat([survey_clean, synthetic], ignore_index=True)
        df_labeled = df_combined.copy()

        for col in df_combined.select_dtypes(include='object').columns:
            df_combined[col] = LabelEncoder().fit_transform(df_combined[col])

        scaled = StandardScaler().fit_transform(df_combined)
        pca = PCA(n_components=2).fit_transform(scaled)

        return df_combined, df_labeled, scaled, pca

    def determine_optimal_clusters(self, df_combined):
        """Use the Elbow Method to visually determine the optimal number of KMeans clusters."""
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

        plt.figure(figsize=(8, 4))
        plt.plot(k_range, sse, marker='o')
        plt.xticks(k_range)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.title("Elbow Method: Choosing Optimal Number of Clusters (k)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return 4

    def compare_models(self, pca):
        """Compare clustering models (KMeans, Hierarchical, DBSCAN) using silhouette scores."""
        results = []

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        k_labels = kmeans.fit_predict(pca)
        k_score = silhouette_score(pca, k_labels)
        results.append(('KMeans', k_score, len(set(k_labels))))

        h_labels = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(pca)
        h_score = silhouette_score(pca, h_labels)
        results.append(('Hierarchical (Ward)', h_score, len(set(h_labels))))

        db = DBSCAN(eps=0.5, min_samples=10).fit(pca)
        db_labels = db.labels_
        valid_labels = db_labels[db_labels != -1]
        db_score = silhouette_score(pca[db_labels != -1], valid_labels) if len(set(valid_labels)) > 1 else 0
        results.append(('DBSCAN', db_score, len(set(valid_labels))))

        results_df = pd.DataFrame(results, columns=['Model', 'Silhouette Score', 'Clusters Found'])
        print(results_df)
        return k_labels

    def summarize_clusters(self, df_labeled, k_labels):
        """Generate descriptive summary of each cluster based on majority attributes."""
        df_labeled['cluster'] = k_labels

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

        cluster_name_map = {
            0: 'Social-Driven Youths',
            1: 'Value-Conscious Families',
            2: 'Budget-Conscious Youths',
            3: 'Premium Spenders'
        }

        summary['Segment'] = summary.index.map(cluster_name_map)
        cols = ['Segment'] + [col for col in summary.columns if col != 'Segment']
        summary = summary[cols]

        return summary, df_labeled

    def analyze_marketing_strategies(self, df_labeled):
        """Analyze top awareness sources, ad response, and preferred promotions by cluster."""
        cluster_name_map = {
            0: 'Social-Driven Youths',
            1: 'Value-Conscious Families',
            2: 'Budget-Conscious Youths',
            3: 'Premium Spenders'
        }

        def compute_top_response(df, cluster_col, response_col, cluster_map):
            top_response = df.groupby(cluster_col)[response_col].agg(
                lambda x: x.value_counts().idxmax()).reset_index()
            top_response[cluster_col] = top_response[cluster_col].map(cluster_map)
            return top_response

        print("--- Business Question 4: Recommend tailored marketing strategies for specific segments ---")

        top_awareness = compute_top_response(df_labeled, 'cluster', 'awareness', cluster_name_map)
        print("### Awareness: 'How did you first hear about Universal Studios Singapore?' ###")
        print(top_awareness)
        print("\n")

        top_response_to_ads = compute_top_response(df_labeled, 'cluster', 'response_to_ads', cluster_name_map)
        print("### Response to ads: 'Have you seen any recent advertisements or promotions for USS?' ###")
        print(top_response_to_ads)
        print("\n")

        top_preferred_promotion = compute_top_response(df_labeled, 'cluster', 'preferred_promotion', cluster_name_map)
        print("### Preferred promotion: 'What type of promotions or discounts would encourage you to visit USS?' ###")
        print(top_preferred_promotion)

    def visualize_clusters(self, df_labeled, scaled_features, k_labels):
        """Generate PCA scatter plot and distribution plots by cluster."""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(scaled_features)

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

    def run_pipeline(self):
        """Execute the full segmentation pipeline and print marketing analysis."""
        survey_clean = self.load_and_clean_data()
        synthetic = self.generate_synthetic_data()
        df_combined, df_labeled, scaled, pca = self.merge_and_encode(survey_clean, synthetic)
        optimal_k = self.determine_optimal_clusters(df_combined)
        k_labels = self.compare_models(pca)
        cluster_summary, df_labeled = self.summarize_clusters(df_labeled, k_labels)
        self.analyze_marketing_strategies(df_labeled)
        self.visualize_clusters(df_labeled, scaled, k_labels)
        return cluster_summary, df_labeled

if __name__ == "__main__":
    analysis = guest_segmentation_model()
    analysis.run_pipeline()
