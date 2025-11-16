"""
Retail Store Clustering Analysis
Clustering retail stores for strategic planning and resource allocation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


class RetailStoreClustering:
    """Cluster retail stores based on performance and characteristics"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        np.random.seed(random_state)

    def generate_store_data(self, n_stores=500):
        """
        Generate synthetic retail store data
        Store types: Flagship, Urban High Traffic, Suburban, Small Town, Struggling
        """
        stores = []

        # Flagship Stores (15%) - High everything
        n_flagship = int(n_stores * 0.15)
        flagship = pd.DataFrame({
            'store_id': [f'STORE_{i:04d}' for i in range(n_flagship)],
            'daily_customers': np.random.normal(2000, 300, n_flagship).clip(1500, 3000),
            'daily_revenue': np.random.normal(50000, 8000, n_flagship).clip(35000, 80000),
            'avg_transaction_value': np.random.normal(75, 10, n_flagship).clip(50, 120),
            'staff_count': np.random.normal(25, 4, n_flagship).clip(18, 35),
            'store_size_sqft': np.random.normal(15000, 2000, n_flagship).clip(12000, 20000),
            'inventory_turnover': np.random.normal(8, 1, n_flagship).clip(6, 12),
            'customer_satisfaction': np.random.normal(4.5, 0.2, n_flagship).clip(4.0, 5.0),
            'conversion_rate': np.random.normal(0.35, 0.05, n_flagship).clip(0.25, 0.50),
            'location_type': 'Downtown',
            'true_cluster': 'Flagship'
        })
        stores.append(flagship)

        # Urban High Traffic (25%) - High volume, medium value
        n_urban = int(n_stores * 0.25)
        urban = pd.DataFrame({
            'store_id': [f'STORE_{i:04d}' for i in range(n_flagship, n_flagship + n_urban)],
            'daily_customers': np.random.normal(1500, 250, n_urban).clip(1000, 2200),
            'daily_revenue': np.random.normal(35000, 6000, n_urban).clip(25000, 50000),
            'avg_transaction_value': np.random.normal(55, 8, n_urban).clip(40, 80),
            'staff_count': np.random.normal(18, 3, n_urban).clip(12, 25),
            'store_size_sqft': np.random.normal(10000, 1500, n_urban).clip(7000, 14000),
            'inventory_turnover': np.random.normal(7, 1, n_urban).clip(5, 10),
            'customer_satisfaction': np.random.normal(4.2, 0.25, n_urban).clip(3.5, 4.8),
            'conversion_rate': np.random.normal(0.28, 0.05, n_urban).clip(0.18, 0.40),
            'location_type': 'Urban',
            'true_cluster': 'Urban High Traffic'
        })
        stores.append(urban)

        # Suburban (30%) - Moderate everything
        n_suburban = int(n_stores * 0.30)
        suburban = pd.DataFrame({
            'store_id': [f'STORE_{i:04d}' for i in range(n_flagship + n_urban,
                                                         n_flagship + n_urban + n_suburban)],
            'daily_customers': np.random.normal(800, 150, n_suburban).clip(500, 1200),
            'daily_revenue': np.random.normal(22000, 4000, n_suburban).clip(15000, 35000),
            'avg_transaction_value': np.random.normal(65, 10, n_suburban).clip(45, 95),
            'staff_count': np.random.normal(12, 2, n_suburban).clip(8, 18),
            'store_size_sqft': np.random.normal(8000, 1200, n_suburban).clip(5500, 12000),
            'inventory_turnover': np.random.normal(6, 1, n_suburban).clip(4, 9),
            'customer_satisfaction': np.random.normal(4.0, 0.3, n_suburban).clip(3.2, 4.7),
            'conversion_rate': np.random.normal(0.25, 0.04, n_suburban).clip(0.15, 0.35),
            'location_type': 'Suburban',
            'true_cluster': 'Suburban'
        })
        stores.append(suburban)

        # Small Town (20%) - Lower volume, loyal customers
        n_small = int(n_stores * 0.20)
        small_town = pd.DataFrame({
            'store_id': [f'STORE_{i:04d}' for i in range(n_flagship + n_urban + n_suburban,
                                                         n_flagship + n_urban + n_suburban + n_small)],
            'daily_customers': np.random.normal(300, 80, n_small).clip(150, 500),
            'daily_revenue': np.random.normal(12000, 2500, n_small).clip(7000, 20000),
            'avg_transaction_value': np.random.normal(70, 12, n_small).clip(45, 110),
            'staff_count': np.random.normal(6, 1.5, n_small).clip(4, 10),
            'store_size_sqft': np.random.normal(4500, 800, n_small).clip(3000, 7000),
            'inventory_turnover': np.random.normal(5, 1, n_small).clip(3, 8),
            'customer_satisfaction': np.random.normal(4.3, 0.25, n_small).clip(3.7, 5.0),
            'conversion_rate': np.random.normal(0.32, 0.05, n_small).clip(0.22, 0.45),
            'location_type': 'Small Town',
            'true_cluster': 'Small Town'
        })
        stores.append(small_town)

        # Struggling Stores (10%) - Low performance
        n_struggling = n_stores - (n_flagship + n_urban + n_suburban + n_small)
        struggling = pd.DataFrame({
            'store_id': [f'STORE_{i:04d}' for i in range(n_flagship + n_urban + n_suburban + n_small,
                                                         n_stores)],
            'daily_customers': np.random.normal(200, 60, n_struggling).clip(80, 400),
            'daily_revenue': np.random.normal(6000, 1500, n_struggling).clip(3000, 11000),
            'avg_transaction_value': np.random.normal(40, 8, n_struggling).clip(25, 65),
            'staff_count': np.random.normal(5, 1, n_struggling).clip(3, 8),
            'store_size_sqft': np.random.normal(3500, 700, n_struggling).clip(2000, 6000),
            'inventory_turnover': np.random.normal(3, 0.8, n_struggling).clip(1.5, 5),
            'customer_satisfaction': np.random.normal(3.2, 0.4, n_struggling).clip(2.0, 4.0),
            'conversion_rate': np.random.normal(0.15, 0.04, n_struggling).clip(0.05, 0.25),
            'location_type': np.random.choice(['Urban', 'Suburban', 'Rural'], n_struggling),
            'true_cluster': 'Struggling'
        })
        stores.append(struggling)

        # Combine all stores
        df = pd.concat(stores, ignore_index=True)

        # Add derived features
        df['revenue_per_customer'] = df['daily_revenue'] / df['daily_customers']
        df['revenue_per_sqft'] = df['daily_revenue'] / df['store_size_sqft']
        df['customers_per_staff'] = df['daily_customers'] / df['staff_count']
        df['staff_efficiency'] = df['daily_revenue'] / df['staff_count']

        # Shuffle
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        return df

    def optimal_clusters_analysis(self, X, max_k=10):
        """Find optimal number of clusters"""
        silhouette_scores = []
        inertias = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
            inertias.append(kmeans.inertia_)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/retail_optimal_k.png', dpi=300, bbox_inches='tight')
        plt.show()

        return K_range[np.argmax(silhouette_scores)]

    def compare_clustering(self, X, n_clusters=5):
        """Compare clustering algorithms"""
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

        # Hierarchical
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        agg_labels = agg.fit_predict(X)
        results['Hierarchical'] = {
            'labels': agg_labels,
            'silhouette': silhouette_score(X, agg_labels),
            'davies_bouldin': davies_bouldin_score(X, agg_labels),
            'calinski_harabasz': calinski_harabasz_score(X, agg_labels)
        }

        return results

    def visualize_clusters(self, X, results):
        """Visualize clusters using PCA"""
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(7*n_algorithms, 6))
        if n_algorithms == 1:
            axes = [axes]

        for idx, (algo_name, result) in enumerate(results.items()):
            labels = result['labels']
            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                       c=labels, cmap='Set2',
                                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
            axes[idx].set_title(f'{algo_name}\nSilhouette: {result["silhouette"]:.3f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')

        plt.tight_layout()
        plt.savefig('/tmp/retail_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_store_profiles(self, df, labels, feature_cols):
        """Analyze characteristics of each store cluster"""
        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        print("\n" + "="*90)
        print("RETAIL STORE CLUSTER PROFILES")
        print("="*90)

        for cluster_id in sorted(set(labels)):
            cluster_stores = df_analysis[df_analysis['cluster'] == cluster_id]
            print(f"\n{'='*90}")
            print(f"CLUSTER {cluster_id} - {len(cluster_stores)} stores "
                  f"({len(cluster_stores)/len(df)*100:.1f}%)")
            print(f"{'='*90}")

            # Key metrics
            metrics = ['daily_customers', 'daily_revenue', 'avg_transaction_value',
                      'staff_count', 'store_size_sqft', 'customer_satisfaction',
                      'conversion_rate', 'revenue_per_customer']

            for metric in metrics:
                if metric in cluster_stores.columns:
                    mean_val = cluster_stores[metric].mean()
                    print(f"{metric:30s}: {mean_val:12.2f}")

            # Location distribution
            print(f"\nLocation distribution:")
            loc_dist = cluster_stores['location_type'].value_counts()
            for loc, count in loc_dist.items():
                print(f"  {loc:20s}: {count:4d} ({count/len(cluster_stores)*100:.1f}%)")

    def create_cluster_comparison_plot(self, df, labels):
        """Create comparative plots for clusters"""
        df_plot = df.copy()
        df_plot['cluster'] = labels

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Revenue by cluster
        sns.boxplot(data=df_plot, x='cluster', y='daily_revenue', ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Daily Revenue by Cluster', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Daily Revenue ($)')

        # Customers by cluster
        sns.boxplot(data=df_plot, x='cluster', y='daily_customers', ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Daily Customers by Cluster', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Daily Customers')

        # Satisfaction by cluster
        sns.boxplot(data=df_plot, x='cluster', y='customer_satisfaction', ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_title('Customer Satisfaction by Cluster', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Satisfaction Score')

        # Conversion rate by cluster
        sns.boxplot(data=df_plot, x='cluster', y='conversion_rate', ax=axes[1, 1], palette='Set2')
        axes[1, 1].set_title('Conversion Rate by Cluster', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Conversion Rate')

        plt.tight_layout()
        plt.savefig('/tmp/retail_cluster_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    print("="*90)
    print("RETAIL STORE CLUSTERING ANALYSIS")
    print("="*90)

    # Initialize
    clustering = RetailStoreClustering(random_state=42)

    # Generate data
    print("\n[1/5] Generating retail store data...")
    df = clustering.generate_store_data(n_stores=500)
    print(f"Generated {len(df)} retail stores")
    print(f"\nTrue cluster distribution:")
    print(df['true_cluster'].value_counts())
    print(f"\nSample stores:")
    print(df[['store_id', 'daily_customers', 'daily_revenue', 'customer_satisfaction']].head(10))

    # Prepare features
    print("\n[2/5] Preparing features...")
    feature_cols = ['daily_customers', 'daily_revenue', 'avg_transaction_value',
                   'staff_count', 'store_size_sqft', 'inventory_turnover',
                   'customer_satisfaction', 'conversion_rate', 'revenue_per_customer',
                   'revenue_per_sqft', 'customers_per_staff', 'staff_efficiency']
    X = clustering.scaler.fit_transform(df[feature_cols])
    print(f"Feature matrix shape: {X.shape}")

    # Find optimal k
    print("\n[3/5] Finding optimal number of clusters...")
    optimal_k = clustering.optimal_clusters_analysis(X, max_k=10)
    print(f"Suggested optimal k: {optimal_k}")

    # Compare algorithms
    print(f"\n[4/5] Comparing clustering algorithms with k={optimal_k}...")
    results = clustering.compare_clustering(X, n_clusters=optimal_k)

    print("\nClustering Performance:")
    print("-" * 90)
    print(f"{'Algorithm':<15} {'Silhouette':>12} {'Davies-Bouldin':>17} {'Calinski-Harabasz':>20}")
    print("-" * 90)
    for algo_name, metrics in results.items():
        print(f"{algo_name:<15} {metrics['silhouette']:>12.4f} "
              f"{metrics['davies_bouldin']:>17.4f} {metrics['calinski_harabasz']:>20.2f}")

    # Visualize
    print("\n[5/5] Visualizing and analyzing clusters...")
    clustering.visualize_clusters(X, results)

    # Analyze profiles
    best_algo = max(results.items(), key=lambda x: x[1]['silhouette'])
    clustering.analyze_store_profiles(df, best_algo[1]['labels'], feature_cols)

    # Comparison plots
    clustering.create_cluster_comparison_plot(df, best_algo[1]['labels'])

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE!")
    print("="*90)


if __name__ == "__main__":
    main()
