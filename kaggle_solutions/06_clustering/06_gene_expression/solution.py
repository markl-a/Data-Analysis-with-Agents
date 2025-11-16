"""
Gene Expression Clustering Analysis
Clustering genes and samples based on expression patterns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class GeneExpressionClustering:
    """Cluster genes and samples based on expression patterns"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        np.random.seed(random_state)

    def generate_expression_data(self, n_genes=200, n_samples=50):
        """
        Generate synthetic gene expression data
        Simulates different expression patterns across sample groups
        """
        # Create sample groups (e.g., different conditions or tissues)
        samples_per_group = n_samples // 5

        # Group 1: Control/Normal
        # Group 2: Disease Type A
        # Group 3: Disease Type B
        # Group 4: Treatment Response High
        # Group 5: Treatment Response Low

        sample_groups = np.repeat(['Control', 'Disease_A', 'Disease_B',
                                  'Treatment_High', 'Treatment_Low'],
                                 samples_per_group)

        # Generate gene expression patterns
        expression_matrix = np.zeros((n_genes, n_samples))

        # Gene cluster 1: Upregulated in Disease A (40 genes)
        for i in range(40):
            # High in Disease A, normal elsewhere
            expression_matrix[i, samples_per_group:2*samples_per_group] = \
                np.random.normal(8, 1, samples_per_group)
            expression_matrix[i, :samples_per_group] = np.random.normal(3, 0.5, samples_per_group)
            expression_matrix[i, 2*samples_per_group:] = \
                np.random.normal(3.5, 0.5, n_samples - 2*samples_per_group)

        # Gene cluster 2: Upregulated in Disease B (40 genes)
        for i in range(40, 80):
            expression_matrix[i, 2*samples_per_group:3*samples_per_group] = \
                np.random.normal(9, 1, samples_per_group)
            expression_matrix[i, :2*samples_per_group] = np.random.normal(3, 0.5, 2*samples_per_group)
            expression_matrix[i, 3*samples_per_group:] = \
                np.random.normal(3.5, 0.5, n_samples - 3*samples_per_group)

        # Gene cluster 3: Treatment response markers (40 genes)
        for i in range(80, 120):
            expression_matrix[i, 3*samples_per_group:4*samples_per_group] = \
                np.random.normal(10, 1, samples_per_group)
            expression_matrix[i, :3*samples_per_group] = \
                np.random.normal(3, 0.5, 3*samples_per_group)
            expression_matrix[i, 4*samples_per_group:] = \
                np.random.normal(4, 0.5, samples_per_group)

        # Gene cluster 4: Housekeeping genes (stable expression) (40 genes)
        for i in range(120, 160):
            expression_matrix[i, :] = np.random.normal(6, 0.3, n_samples)

        # Gene cluster 5: Downregulated in all diseases (40 genes)
        for i in range(160, 200):
            expression_matrix[i, :samples_per_group] = np.random.normal(8, 0.5, samples_per_group)
            expression_matrix[i, samples_per_group:] = \
                np.random.normal(3, 0.5, n_samples - samples_per_group)

        # Add some noise
        expression_matrix += np.random.normal(0, 0.2, (n_genes, n_samples))
        expression_matrix = np.maximum(expression_matrix, 0)  # No negative expression

        # Create DataFrame
        gene_names = [f'GENE_{i:04d}' for i in range(n_genes)]
        sample_names = [f'{group}_{i:02d}' for i, group in enumerate(sample_groups)]

        df = pd.DataFrame(expression_matrix, index=gene_names, columns=sample_names)

        return df, sample_groups

    def plot_heatmap(self, df, row_clusters=None, col_clusters=None, title="Gene Expression Heatmap"):
        """Plot expression heatmap with optional cluster annotations"""
        plt.figure(figsize=(14, 10))

        # Create annotations
        if row_clusters is not None:
            row_colors = pd.Series(row_clusters, index=df.index).map(
                lambda x: sns.color_palette("Set2", len(set(row_clusters)))[x]
            )
        else:
            row_colors = None

        # Create clustermap
        g = sns.clustermap(df, cmap='RdYlBu_r', center=df.values.mean(),
                          row_colors=row_colors, figsize=(14, 10),
                          cbar_kws={'label': 'Expression Level'},
                          xticklabels=True, yticklabels=False)

        g.fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.savefig('/tmp/gene_expression_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def cluster_genes(self, df, n_clusters=5):
        """Cluster genes based on expression patterns"""
        X = self.scaler.fit_transform(df.values)

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

        # Spectral Clustering
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=self.random_state,
                                     affinity='nearest_neighbors')
        spectral_labels = spectral.fit_predict(X)
        results['Spectral'] = {
            'labels': spectral_labels,
            'silhouette': silhouette_score(X, spectral_labels),
            'davies_bouldin': davies_bouldin_score(X, spectral_labels),
            'calinski_harabasz': calinski_harabasz_score(X, spectral_labels)
        }

        return results

    def visualize_gene_clusters(self, df, results):
        """Visualize gene clusters using PCA"""
        X = self.scaler.fit_transform(df.values)
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 5))
        if n_algorithms == 1:
            axes = [axes]

        for idx, (algo_name, result) in enumerate(results.items()):
            labels = result['labels']
            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                       c=labels, cmap='tab10',
                                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            axes[idx].set_title(f'{algo_name}\nSilhouette: {result["silhouette"]:.3f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')

        plt.tight_layout()
        plt.savefig('/tmp/gene_clusters_pca.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_dendrogram(self, df, max_display=50):
        """Plot hierarchical clustering dendrogram"""
        # Sample genes if too many
        if len(df) > max_display:
            df_sample = df.sample(n=max_display, random_state=self.random_state)
        else:
            df_sample = df

        X = self.scaler.fit_transform(df_sample.values)
        linkage_matrix = linkage(X, method='ward')

        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, labels=df_sample.index.values,
                  leaf_rotation=90, leaf_font_size=8)
        plt.title('Hierarchical Clustering Dendrogram (Sample of Genes)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Gene ID', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        plt.savefig('/tmp/gene_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_cluster_patterns(self, df, labels, sample_groups):
        """Analyze expression patterns for each gene cluster"""
        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        print("\n" + "="*80)
        print("GENE CLUSTER EXPRESSION PATTERNS")
        print("="*80)

        unique_groups = np.unique(sample_groups)

        for cluster_id in sorted(set(labels)):
            cluster_genes = df_analysis[df_analysis['cluster'] == cluster_id]
            print(f"\n{'='*80}")
            print(f"CLUSTER {cluster_id} - {len(cluster_genes)} genes "
                  f"({len(cluster_genes)/len(df)*100:.1f}%)")
            print(f"{'='*80}")

            # Calculate mean expression per sample group
            print("\nMean expression by sample group:")
            for group in unique_groups:
                group_cols = [col for col in df.columns if col.startswith(group)]
                mean_expr = cluster_genes[group_cols].mean().mean()
                std_expr = cluster_genes[group_cols].mean().std()
                print(f"  {group:20s}: {mean_expr:7.3f} ± {std_expr:6.3f}")

            # Show top genes by variance
            print("\nTop 5 most variable genes in cluster:")
            gene_variances = cluster_genes.drop('cluster', axis=1).var(axis=1)
            top_genes = gene_variances.nlargest(5)
            for gene, var in top_genes.items():
                print(f"  {gene}: variance = {var:.3f}")

    def plot_cluster_expression_profiles(self, df, labels, n_clusters):
        """Plot average expression profiles for each cluster"""
        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for cluster_id in range(min(n_clusters, 6)):
            cluster_genes = df_analysis[df_analysis['cluster'] == cluster_id]
            if len(cluster_genes) == 0:
                continue

            # Calculate mean and std across samples
            cluster_data = cluster_genes.drop('cluster', axis=1)
            mean_profile = cluster_data.mean(axis=0)
            std_profile = cluster_data.std(axis=0)

            x = np.arange(len(mean_profile))
            axes[cluster_id].plot(x, mean_profile, 'b-', linewidth=2, label='Mean')
            axes[cluster_id].fill_between(x, mean_profile - std_profile,
                                         mean_profile + std_profile,
                                         alpha=0.3, label='±1 std')
            axes[cluster_id].set_title(f'Cluster {cluster_id} ({len(cluster_genes)} genes)',
                                      fontsize=12, fontweight='bold')
            axes[cluster_id].set_xlabel('Sample Index', fontsize=10)
            axes[cluster_id].set_ylabel('Expression Level', fontsize=10)
            axes[cluster_id].grid(True, alpha=0.3)
            axes[cluster_id].legend()

        plt.tight_layout()
        plt.savefig('/tmp/cluster_expression_profiles.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    print("="*80)
    print("GENE EXPRESSION CLUSTERING ANALYSIS")
    print("="*80)

    # Initialize
    clustering = GeneExpressionClustering(random_state=42)

    # Generate data
    print("\n[1/6] Generating gene expression data...")
    df, sample_groups = clustering.generate_expression_data(n_genes=200, n_samples=50)
    print(f"Expression matrix shape: {df.shape}")
    print(f"Genes: {df.shape[0]}, Samples: {df.shape[1]}")
    print(f"\nSample groups: {np.unique(sample_groups)}")
    print(f"\nExpression statistics:")
    print(df.describe())

    # Plot heatmap
    print("\n[2/6] Generating expression heatmap...")
    clustering.plot_heatmap(df, title="Gene Expression Heatmap")

    # Cluster genes
    print("\n[3/6] Clustering genes...")
    n_clusters = 5
    results = clustering.cluster_genes(df, n_clusters=n_clusters)

    print("\nClustering Performance:")
    print("-" * 90)
    print(f"{'Algorithm':<15} {'Silhouette':>12} {'Davies-Bouldin':>17} "
          f"{'Calinski-Harabasz':>20}")
    print("-" * 90)
    for algo_name, metrics in results.items():
        print(f"{algo_name:<15} {metrics['silhouette']:>12.4f} "
              f"{metrics['davies_bouldin']:>17.4f} "
              f"{metrics['calinski_harabasz']:>20.2f}")

    # Visualize clusters
    print("\n[4/6] Visualizing gene clusters...")
    clustering.visualize_gene_clusters(df, results)

    # Plot dendrogram
    print("\n[5/6] Creating dendrogram...")
    clustering.plot_dendrogram(df, max_display=50)

    # Analyze patterns
    print("\n[6/6] Analyzing cluster patterns...")
    best_algo = max(results.items(), key=lambda x: x[1]['silhouette'])
    clustering.analyze_cluster_patterns(df, best_algo[1]['labels'], sample_groups)

    # Plot expression profiles
    clustering.plot_cluster_expression_profiles(df, best_algo[1]['labels'], n_clusters)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
