"""
Network Traffic Clustering Analysis
Clustering network traffic patterns for anomaly detection and profiling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class NetworkTrafficClustering:
    """Cluster network traffic for pattern analysis and anomaly detection"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = RobustScaler()  # Better for outliers
        np.random.seed(random_state)

    def generate_traffic_data(self, n_flows=2000):
        """
        Generate realistic network traffic flow data
        Traffic types: Normal Web, Video Streaming, File Transfer, P2P, DDoS Attack
        """
        flows = []

        # Normal Web Traffic (HTTP/HTTPS) - 40%
        n_web = int(n_flows * 0.4)
        web_traffic = pd.DataFrame({
            'bytes_sent': np.random.lognormal(8, 1.5, n_web).clip(100, 100000),
            'bytes_received': np.random.lognormal(10, 2, n_web).clip(500, 500000),
            'packets_sent': np.random.poisson(50, n_web).clip(5, 200),
            'packets_received': np.random.poisson(80, n_web).clip(10, 300),
            'duration': np.random.exponential(5, n_web).clip(0.1, 60),
            'port': np.random.choice([80, 443, 8080], n_web),
            'protocol': 'TCP',
            'flow_rate': np.random.normal(5000, 2000, n_web).clip(1000, 20000),
            'packet_rate': np.random.normal(15, 5, n_web).clip(1, 50),
            'true_type': 'Normal Web'
        })
        flows.append(web_traffic)

        # Video Streaming - 25%
        n_video = int(n_flows * 0.25)
        video_traffic = pd.DataFrame({
            'bytes_sent': np.random.lognormal(7, 1, n_video).clip(100, 50000),
            'bytes_received': np.random.lognormal(14, 1.5, n_video).clip(10000, 5000000),
            'packets_sent': np.random.poisson(30, n_video).clip(5, 100),
            'packets_received': np.random.poisson(500, n_video).clip(100, 2000),
            'duration': np.random.normal(300, 100, n_video).clip(60, 3600),
            'port': np.random.choice([1935, 8080, 443], n_video),
            'protocol': 'TCP',
            'flow_rate': np.random.normal(50000, 15000, n_video).clip(10000, 150000),
            'packet_rate': np.random.normal(100, 30, n_video).clip(30, 300),
            'true_type': 'Video Streaming'
        })
        flows.append(video_traffic)

        # File Transfer (FTP/SFTP) - 15%
        n_ftp = int(n_flows * 0.15)
        ftp_traffic = pd.DataFrame({
            'bytes_sent': np.random.lognormal(9, 2, n_ftp).clip(1000, 200000),
            'bytes_received': np.random.lognormal(16, 2, n_ftp).clip(100000, 10000000),
            'packets_sent': np.random.poisson(100, n_ftp).clip(10, 500),
            'packets_received': np.random.poisson(1000, n_ftp).clip(100, 5000),
            'duration': np.random.normal(180, 60, n_ftp).clip(30, 600),
            'port': np.random.choice([21, 22, 990], n_ftp),
            'protocol': 'TCP',
            'flow_rate': np.random.normal(80000, 25000, n_ftp).clip(20000, 200000),
            'packet_rate': np.random.normal(200, 60, n_ftp).clip(50, 500),
            'true_type': 'File Transfer'
        })
        flows.append(ftp_traffic)

        # P2P Traffic - 10%
        n_p2p = int(n_flows * 0.1)
        p2p_traffic = pd.DataFrame({
            'bytes_sent': np.random.lognormal(12, 2, n_p2p).clip(10000, 1000000),
            'bytes_received': np.random.lognormal(12, 2, n_p2p).clip(10000, 1000000),
            'packets_sent': np.random.poisson(300, n_p2p).clip(50, 1000),
            'packets_received': np.random.poisson(300, n_p2p).clip(50, 1000),
            'duration': np.random.normal(600, 200, n_p2p).clip(120, 3600),
            'port': np.random.randint(10000, 60000, n_p2p),
            'protocol': np.random.choice(['TCP', 'UDP'], n_p2p),
            'flow_rate': np.random.normal(30000, 10000, n_p2p).clip(5000, 100000),
            'packet_rate': np.random.normal(80, 30, n_p2p).clip(20, 200),
            'true_type': 'P2P'
        })
        flows.append(p2p_traffic)

        # DDoS Attack Traffic - 10%
        n_ddos = int(n_flows * 0.1)
        ddos_traffic = pd.DataFrame({
            'bytes_sent': np.random.exponential(500, n_ddos).clip(50, 5000),
            'bytes_received': np.random.exponential(200, n_ddos).clip(20, 2000),
            'packets_sent': np.random.poisson(200, n_ddos).clip(100, 1000),
            'packets_received': np.random.poisson(10, n_ddos).clip(1, 50),
            'duration': np.random.exponential(0.5, n_ddos).clip(0.01, 5),
            'port': np.random.choice([80, 443, 53], n_ddos),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_ddos),
            'flow_rate': np.random.normal(1000, 500, n_ddos).clip(100, 5000),
            'packet_rate': np.random.normal(500, 200, n_ddos).clip(100, 2000),
            'true_type': 'DDoS Attack'
        })
        flows.append(ddos_traffic)

        # Combine and add features
        df = pd.concat(flows, ignore_index=True)

        # Add derived features
        df['bytes_ratio'] = df['bytes_sent'] / (df['bytes_received'] + 1)
        df['packets_ratio'] = df['packets_sent'] / (df['packets_received'] + 1)
        df['avg_packet_size'] = (df['bytes_sent'] + df['bytes_received']) / \
                                (df['packets_sent'] + df['packets_received'] + 1)

        # Add flow ID
        df.insert(0, 'flow_id', [f'FLOW_{i:05d}' for i in range(len(df))])

        # Shuffle
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        return df

    def feature_engineering(self, df):
        """Extract and engineer features for clustering"""
        feature_cols = ['bytes_sent', 'bytes_received', 'packets_sent', 'packets_received',
                       'duration', 'flow_rate', 'packet_rate', 'bytes_ratio',
                       'packets_ratio', 'avg_packet_size']
        return feature_cols

    def detect_optimal_clusters(self, X, max_k=10):
        """Find optimal number of clusters"""
        silhouette_scores = []
        inertias = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=self.random_state,
                                    batch_size=100, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels, sample_size=min(1000, len(X))))
            inertias.append(kmeans.inertia_)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/network_optimal_k.png', dpi=300, bbox_inches='tight')
        plt.show()

        return K_range[np.argmax(silhouette_scores)]

    def compare_clustering(self, X, n_clusters=5):
        """Compare different clustering algorithms"""
        results = {}

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        results['KMeans'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X, kmeans_labels, sample_size=min(1000, len(X))),
            'davies_bouldin': davies_bouldin_score(X, kmeans_labels),
            'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels)
        }

        # DBSCAN for anomaly detection
        dbscan = DBSCAN(eps=0.8, min_samples=15)
        dbscan_labels = dbscan.fit_predict(X)
        n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)

        if n_clusters_db > 1:
            # Only evaluate if we have valid clusters
            valid_mask = dbscan_labels != -1
            if valid_mask.sum() > 1:
                results['DBSCAN'] = {
                    'labels': dbscan_labels,
                    'silhouette': silhouette_score(X[valid_mask], dbscan_labels[valid_mask])
                                 if len(set(dbscan_labels[valid_mask])) > 1 else -1,
                    'davies_bouldin': davies_bouldin_score(X[valid_mask], dbscan_labels[valid_mask])
                                     if len(set(dbscan_labels[valid_mask])) > 1 else -1,
                    'calinski_harabasz': calinski_harabasz_score(X[valid_mask], dbscan_labels[valid_mask])
                                        if len(set(dbscan_labels[valid_mask])) > 1 else -1,
                    'n_noise': n_noise,
                    'n_clusters': n_clusters_db
                }

        return results

    def visualize_clusters(self, X, results, feature_names):
        """Visualize clustering results"""
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(7*n_algorithms, 6))
        if n_algorithms == 1:
            axes = [axes]

        for idx, (algo_name, result) in enumerate(results.items()):
            labels = result['labels']
            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                       c=labels, cmap='tab10',
                                       s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

            title = f'{algo_name}\n'
            if 'n_clusters' in result:
                title += f"{result['n_clusters']} clusters"
                if 'n_noise' in result:
                    title += f" ({result['n_noise']} anomalies)\n"
            title += f"Silhouette: {result['silhouette']:.3f}"

            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')

        plt.tight_layout()
        plt.savefig('/tmp/network_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_traffic_profiles(self, df, labels, feature_cols):
        """Analyze traffic characteristics for each cluster"""
        df_analysis = df[feature_cols].copy()
        df_analysis['cluster'] = labels

        print("\n" + "="*90)
        print("NETWORK TRAFFIC CLUSTER PROFILES")
        print("="*90)

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                print(f"\n{'='*90}")
                print(f"ANOMALIES (Cluster -1) - {(labels == -1).sum()} flows")
                print(f"{'='*90}")
                continue

            cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
            print(f"\n{'='*90}")
            print(f"CLUSTER {cluster_id} - {len(cluster_data)} flows "
                  f"({len(cluster_data)/len(df)*100:.1f}%)")
            print(f"{'='*90}")

            for col in feature_cols[:8]:  # Main features
                mean_val = cluster_data[col].mean()
                median_val = cluster_data[col].median()
                print(f"{col:25s}: mean={mean_val:12.2f}, median={median_val:12.2f}")


def main():
    print("="*90)
    print("NETWORK TRAFFIC CLUSTERING ANALYSIS")
    print("="*90)

    # Initialize
    clustering = NetworkTrafficClustering(random_state=42)

    # Generate data
    print("\n[1/5] Generating network traffic data...")
    df = clustering.generate_traffic_data(n_flows=2000)
    print(f"Generated {len(df)} network flows")
    print(f"\nTrue traffic distribution:")
    print(df['true_type'].value_counts())
    print(f"\nSample flows:")
    print(df[['flow_id', 'bytes_sent', 'bytes_received', 'duration', 'true_type']].head())

    # Feature engineering
    print("\n[2/5] Preparing features...")
    feature_cols = clustering.feature_engineering(df)
    X = clustering.scaler.fit_transform(df[feature_cols])
    print(f"Feature matrix shape: {X.shape}")

    # Find optimal k
    print("\n[3/5] Finding optimal number of clusters...")
    optimal_k = clustering.detect_optimal_clusters(X, max_k=10)
    print(f"Suggested optimal k: {optimal_k}")

    # Compare algorithms
    print(f"\n[4/5] Comparing clustering algorithms with k={optimal_k}...")
    results = clustering.compare_clustering(X, n_clusters=optimal_k)

    print("\nClustering Performance:")
    print("-" * 90)
    for algo_name, metrics in results.items():
        print(f"\n{algo_name}:")
        for metric, value in metrics.items():
            if metric != 'labels':
                print(f"  {metric:20s}: {value}")

    # Visualize
    print("\n[5/5] Visualizing clusters...")
    clustering.visualize_clusters(X, results, feature_cols)

    # Analyze profiles
    best_algo = max(results.items(), key=lambda x: x[1]['silhouette'])
    clustering.analyze_traffic_profiles(df, best_algo[1]['labels'], feature_cols)

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE!")
    print("="*90)


if __name__ == "__main__":
    main()
