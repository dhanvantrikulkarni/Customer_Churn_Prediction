import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class CustomerSegmentation:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.kmeans_model = None
        self.dbscan_model = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_segmentation_features(self, df):
        feature_cols = [
            'CreditScore', 'Age', 'Tenure', 'Account Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score',
            'Point Earned', 'Engagement_Score'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].copy()
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=available_cols)
        
        print(f"Segmentation features prepared. Shape: {X_scaled.shape}")
        return X_scaled
    
    def find_optimal_k(self, X, max_k=10):
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        # Use a sample for silhouette score to avoid memory issues
        sample_size = min(5000, len(X))
        X_sample = X.sample(sample_size, random_state=42) if hasattr(X, 'sample') else X[:sample_size]
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            # Use sample for silhouette score calculation
            labels_sample = kmeans.labels_[:sample_size] if len(kmeans.labels_) > sample_size else kmeans.labels_
            X_for_silhouette = X_sample if len(X) > sample_size else X
            silhouette_scores.append(silhouette_score(X_for_silhouette, labels_sample))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'optimal_k_analysis.png'))
        plt.close()
        print("Optimal k analysis plot saved.")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k, inertias, silhouette_scores
    
    def train_kmeans(self, X, n_clusters=4):
        print(f"Training K-Means with {n_clusters} clusters...")
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, self.cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        self.save_model('kmeans', self.kmeans_model)
        return self.cluster_labels, silhouette_avg
    
    def train_dbscan(self, X, eps=0.5, min_samples=5):
        print("Training DBSCAN...")
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = self.dbscan_model.fit_predict(X)
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        if n_clusters > 1:
            valid_labels = self.cluster_labels != -1
            if valid_labels.sum() > 0:
                silhouette_avg = silhouette_score(X[valid_labels], self.cluster_labels[valid_labels])
                print(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
            else:
                silhouette_avg = 0
        else:
            silhouette_avg = 0
        
        self.save_model('dbscan', self.dbscan_model)
        return self.cluster_labels, silhouette_avg
    
    def visualize_clusters(self, X, method='kmeans'):
        if self.cluster_labels is None:
            print("No cluster labels available. Train a model first.")
            return
        
        df_vis = X.copy()
        df_vis['Cluster'] = self.cluster_labels
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        sns.scatterplot(
            data=df_vis, x='CreditScore', y='Account Balance',
            hue='Cluster', palette='viridis', ax=axes[0, 0]
        )
        axes[0, 0].set_title('Credit Score vs Account Balance')
        
        sns.scatterplot(
            data=df_vis, x='Age', y='EstimatedSalary',
            hue='Cluster', palette='viridis', ax=axes[0, 1]
        )
        axes[0, 1].set_title('Age vs Estimated Salary')
        
        sns.scatterplot(
            data=df_vis, x='Engagement_Score', y='Satisfaction Score',
            hue='Cluster', palette='viridis', ax=axes[1, 0]
        )
        axes[1, 0].set_title('Engagement Score vs Satisfaction Score')
        
        sns.scatterplot(
            data=df_vis, x='Tenure', y='NumOfProducts',
            hue='Cluster', palette='viridis', ax=axes[1, 1]
        )
        axes[1, 1].set_title('Tenure vs Number of Products')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'{method}_clusters_visualization.png'))
        plt.close()
        print(f"Cluster visualization saved for {method}.")
    
    def analyze_clusters(self, df, method='kmeans'):
        if self.cluster_labels is None:
            print("No cluster labels available.")
            return None
        
        df_analysis = df.copy()
        df_analysis['Cluster'] = self.cluster_labels
        
        cluster_summary = df_analysis.groupby('Cluster').agg({
            'CreditScore': ['mean', 'std'],
            'Age': ['mean', 'std'],
            'Tenure': ['mean', 'std'],
            'Account Balance': ['mean', 'std'],
            'NumOfProducts': ['mean', 'std'],
            'EstimatedSalary': ['mean', 'std'],
            'Satisfaction Score': ['mean', 'std'],
            'Point Earned': ['mean', 'std'],
            'Exited': ['mean', 'count']
        })
        
        cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
        cluster_summary.to_csv(os.path.join(self.model_dir, f'{method}_cluster_analysis.csv'))
        print(f"Cluster analysis saved for {method}.")
        
        return cluster_summary
    
    def save_model(self, model_name, model):
        joblib.dump(model, os.path.join(self.model_dir, f'{model_name}_model.pkl'))
        print(f"Model saved: {model_name}_model.pkl")
    
    def load_model(self, model_name):
        model = joblib.load(os.path.join(self.model_dir, f'{model_name}_model.pkl'))
        print(f"Model loaded: {model_name}_model.pkl")
        return model
    
    def segment_customers(self, df, method='kmeans', **kwargs):
        X = self.prepare_segmentation_features(df)
        
        if method == 'kmeans':
            if 'n_clusters' not in kwargs:
                optimal_k, _, _ = self.find_optimal_k(X)
                kwargs['n_clusters'] = optimal_k
            labels, silhouette = self.train_kmeans(X, **kwargs)
        elif method == 'dbscan':
            labels, silhouette = self.train_dbscan(X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.visualize_clusters(X, method)
        cluster_summary = self.analyze_clusters(df, method)
        
        return labels, silhouette, cluster_summary


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor('Customer-Churn_Records.csv')
    preprocessor.load_data()
    preprocessor.handle_missing_values()
    preprocessor.encode_categorical_variables()
    preprocessor.feature_engineering()
    
    segmentation = CustomerSegmentation()
    
    print("\n=== K-Means Clustering ===")
    kmeans_labels, kmeans_silhouette, kmeans_summary = segmentation.segment_customers(
        preprocessor.df, method='kmeans'
    )
    
    print("\n=== DBSCAN Clustering ===")
    dbscan_labels, dbscan_silhouette, dbscan_summary = segmentation.segment_customers(
        preprocessor.df, method='dbscan', eps=0.5, min_samples=5
    )
    
    print("\nCustomer segmentation completed successfully!")
