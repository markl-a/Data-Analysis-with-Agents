"""
User Behavior Clustering Analysis
Clustering users based on their interaction patterns, session duration, and engagement metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


class UserBehaviorClustering:
    """Cluster users based on their behavior patterns"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        np.random.seed(random_state)

    def generate_user_data(self, n_users=1000):
        """
        Generate realistic user behavior data
        Features: sessions_per_week, avg_session_duration, pages_per_session,
                 bounce_rate, conversion_rate, time_on_site
        """
        # Create different user segments
        # Segment 1: Power Users (high engagement)
        n_power = n_users // 4
        power_users = pd.DataFrame({
            'sessions_per_week': np.random.normal(15, 3, n_power).clip(10, 25),
            'avg_session_duration': np.random.normal(600, 100, n_power).clip(400, 900),
            'pages_per_session': np.random.normal(12, 2, n_power).clip(8, 20),
            'bounce_rate': np.random.normal(0.2, 0.05, n_power).clip(0.05, 0.35),
            'conversion_rate': np.random.normal(0.15, 0.03, n_power).clip(0.08, 0.25),
            'time_on_site': np.random.normal(450, 50, n_power).clip(350, 600)
        })

        # Segment 2: Regular Users (moderate engagement)
        n_regular = n_users // 3
        regular_users = pd.DataFrame({
            'sessions_per_week': np.random.normal(5, 1.5, n_regular).clip(2, 10),
            'avg_session_duration': np.random.normal(300, 60, n_regular).clip(180, 450),
            'pages_per_session': np.random.normal(5, 1.5, n_regular).clip(3, 8),
            'bounce_rate': np.random.normal(0.45, 0.08, n_regular).clip(0.25, 0.65),
            'conversion_rate': np.random.normal(0.05, 0.02, n_regular).clip(0.01, 0.10),
            'time_on_site': np.random.normal(200, 40, n_regular).clip(120, 300)
        })

        # Segment 3: Occasional Users (low engagement)
        n_occasional = n_users // 4
        occasional_users = pd.DataFrame({
            'sessions_per_week': np.random.normal(1.5, 0.5, n_occasional).clip(0.5, 3),
            'avg_session_duration': np.random.normal(120, 30, n_occasional).clip(60, 200),
            'pages_per_session': np.random.normal(2, 0.8, n_occasional).clip(1, 4),
            'bounce_rate': np.random.normal(0.7, 0.1, n_occasional).clip(0.5, 0.9),
            'conversion_rate': np.random.normal(0.01, 0.005, n_occasional).clip(0, 0.03),
            'time_on_site': np.random.normal(80, 20, n_occasional).clip(40, 150)
        })

        # Segment 4: Churned/At-Risk Users (very low engagement)
        n_churned = n_users - n_power - n_regular - n_occasional
        churned_users = pd.DataFrame({
            'sessions_per_week': np.random.normal(0.5, 0.2, n_churned).clip(0.1, 1),
            'avg_session_duration': np.random.normal(50, 15, n_churned).clip(20, 100),
            'pages_per_session': np.random.normal(1.2, 0.3, n_churned).clip(1, 2),
            'bounce_rate': np.random.normal(0.85, 0.08, n_churned).clip(0.7, 0.98),
            'conversion_rate': np.random.normal(0.002, 0.001, n_churned).clip(0, 0.01),
            'time_on_site': np.random.normal(35, 10, n_churned).clip(15, 60)
        })

        # Combine all segments
        df = pd.concat([power_users, regular_users, occasional_users, churned_users],
                      ignore_index=True)

        # Add user IDs
        df.insert(0, 'user_id', [f'USER_{i:04d}' for i in range(len(df))])

        # Shuffle the dataframe
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        return df

    def elbow_method(self, X, max_k=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/elbow_method.png', dpi=300, bbox_inches='tight')
        plt.show()

        return K_range[np.argmax(silhouette_scores)]

    def compare_clustering_algorithms(self, X, n_clusters=4):
        """Compare different clustering algorithms"""
        results = {}

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        results['KMeans'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X, kmeans_labels),
            'davies_bouldin': davies_bouldin_score(X, kmeans_labels),
            'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels)
        }

        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan_labels = dbscan.fit_predict(X)
        if len(set(dbscan_labels)) > 1 and -1 not in set(dbscan_labels):
            results['DBSCAN'] = {
                'labels': dbscan_labels,
                'silhouette': silhouette_score(X, dbscan_labels),
                'davies_bouldin': davies_bouldin_score(X, dbscan_labels),
                'calinski_harabasz': calinski_harabasz_score(X, dbscan_labels)
            }

        # Agglomerative Clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg.fit_predict(X)
        results['Agglomerative'] = {
            'labels': agg_labels,
            'silhouette': silhouette_score(X, agg_labels),
            'davies_bouldin': davies_bouldin_score(X, agg_labels),
            'calinski_harabasz': calinski_harabasz_score(X, agg_labels)
        }

        return results

    def visualize_clusters(self, X, results, feature_names):
        """Visualize clustering results"""
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 5))
        if n_algorithms == 1:
            axes = [axes]

        for idx, (algo_name, result) in enumerate(results.items()):
            labels = result['labels']
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                       c=labels, cmap='viridis',
                                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            axes[idx].set_title(f'{algo_name}\n{n_clusters} clusters\nSilhouette: {result["silhouette"]:.3f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')

        plt.tight_layout()
        plt.savefig('/tmp/clustering_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_cluster_profiles(self, df, labels):
        """Analyze and profile each cluster"""
        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        print("\n" + "="*80)
        print("CLUSTER PROFILING ANALYSIS")
        print("="*80)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for cluster_id in sorted(df_analysis['cluster'].unique()):
            if cluster_id == -1:
                continue

            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
            print(f"\n{'='*80}")
            print(f"CLUSTER {cluster_id} - {len(cluster_data)} users ({len(cluster_data)/len(df)*100:.1f}%)")
            print(f"{'='*80}")

            for col in numeric_cols:
                mean_val = cluster_data[col].mean()
                overall_mean = df[col].mean()
                diff_pct = ((mean_val - overall_mean) / overall_mean) * 100
                print(f"{col:25s}: {mean_val:8.2f} (overall: {overall_mean:8.2f}, diff: {diff_pct:+6.1f}%)")


def main():
    print("="*80)
    print("USER BEHAVIOR CLUSTERING ANALYSIS")
    print("="*80)

    # Initialize
    clustering = UserBehaviorClustering(random_state=42)

    # Generate data
    print("\n[1/5] Generating user behavior data...")
    df = clustering.generate_user_data(n_users=1000)
    print(f"Generated data shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData statistics:")
    print(df.describe())

    # Prepare features
    feature_cols = [col for col in df.columns if col != 'user_id']
    X = clustering.scaler.fit_transform(df[feature_cols])

    # Elbow method
    print("\n[2/5] Running elbow method to find optimal k...")
    optimal_k = clustering.elbow_method(X, max_k=10)
    print(f"Suggested optimal k: {optimal_k}")

    # Compare algorithms
    print(f"\n[3/5] Comparing clustering algorithms with k={optimal_k}...")
    results = clustering.compare_clustering_algorithms(X, n_clusters=optimal_k)

    print("\nClustering Performance Metrics:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Silhouette':>12} {'Davies-Bouldin':>17} {'Calinski-Harabasz':>20}")
    print("-" * 80)
    for algo_name, metrics in results.items():
        print(f"{algo_name:<20} {metrics['silhouette']:>12.4f} {metrics['davies_bouldin']:>17.4f} "
              f"{metrics['calinski_harabasz']:>20.2f}")

    # Visualize clusters
    print("\n[4/5] Visualizing clusters...")
    clustering.visualize_clusters(X, results, feature_cols)

    # Analyze cluster profiles
    print("\n[5/5] Analyzing cluster profiles...")
    best_algo = max(results.items(), key=lambda x: x[1]['silhouette'])
    clustering.analyze_cluster_profiles(df[feature_cols], best_algo[1]['labels'])

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
