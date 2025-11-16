"""
Network Centrality Analysis - Kaggle Solution
=============================================
Comprehensive analysis of various centrality measures in networks.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CentralityAnalyzer:
    """Comprehensive centrality analysis toolkit"""

    def __init__(self, n_nodes=80, seed=42):
        """
        Initialize centrality analyzer

        Args:
            n_nodes: Number of nodes
            seed: Random seed
        """
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        self.G = None
        self.centrality_df = None

    def generate_network(self):
        """Generate network for centrality analysis"""
        print("Generating network...")

        # Create small-world network (common in real networks)
        self.G = nx.watts_strogatz_graph(self.n_nodes, k=6, p=0.3, seed=self.seed)

        print(f"Created network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        print(f"Average degree: {2*self.G.number_of_edges()/self.G.number_of_nodes():.2f}")

        return self.G

    def compute_all_centralities(self):
        """Compute all major centrality measures"""
        print("\n" + "="*60)
        print("COMPUTING CENTRALITY MEASURES")
        print("="*60)

        centralities = {}

        # 1. Degree Centrality
        print("\nComputing Degree Centrality...")
        centralities['degree'] = nx.degree_centrality(self.G)

        # 2. Betweenness Centrality
        print("Computing Betweenness Centrality...")
        centralities['betweenness'] = nx.betweenness_centrality(self.G)

        # 3. Closeness Centrality
        print("Computing Closeness Centrality...")
        centralities['closeness'] = nx.closeness_centrality(self.G)

        # 4. Eigenvector Centrality
        print("Computing Eigenvector Centrality...")
        try:
            centralities['eigenvector'] = nx.eigenvector_centrality(self.G, max_iter=1000)
        except:
            centralities['eigenvector'] = {node: 0 for node in self.G.nodes()}

        # 5. PageRank
        print("Computing PageRank...")
        centralities['pagerank'] = nx.pagerank(self.G)

        # 6. Katz Centrality
        print("Computing Katz Centrality...")
        try:
            centralities['katz'] = nx.katz_centrality(self.G, alpha=0.1, beta=1.0)
        except:
            centralities['katz'] = {node: 0 for node in self.G.nodes()}

        # 7. Harmonic Centrality
        print("Computing Harmonic Centrality...")
        centralities['harmonic'] = nx.harmonic_centrality(self.G)

        # 8. Load Centrality (variant of betweenness)
        print("Computing Load Centrality...")
        centralities['load'] = nx.load_centrality(self.G)

        # Create dataframe
        self.centrality_df = pd.DataFrame({
            'node': range(self.n_nodes),
            'degree': [centralities['degree'][i] for i in range(self.n_nodes)],
            'betweenness': [centralities['betweenness'][i] for i in range(self.n_nodes)],
            'closeness': [centralities['closeness'][i] for i in range(self.n_nodes)],
            'eigenvector': [centralities['eigenvector'][i] for i in range(self.n_nodes)],
            'pagerank': [centralities['pagerank'][i] for i in range(self.n_nodes)],
            'katz': [centralities['katz'][i] for i in range(self.n_nodes)],
            'harmonic': [centralities['harmonic'][i] for i in range(self.n_nodes)],
            'load': [centralities['load'][i] for i in range(self.n_nodes)]
        })

        print("\nCentrality computation complete!")
        return centralities

    def analyze_centrality_statistics(self):
        """Analyze statistical properties of centrality measures"""
        print("\n" + "="*60)
        print("CENTRALITY STATISTICS")
        print("="*60)

        centrality_cols = [col for col in self.centrality_df.columns if col != 'node']

        # Summary statistics
        print("\nSummary Statistics:")
        summary = self.centrality_df[centrality_cols].describe()
        print(summary.round(6))

        # Identify top nodes for each measure
        print("\n" + "="*60)
        print("TOP 5 NODES BY EACH CENTRALITY MEASURE")
        print("="*60)

        for measure in centrality_cols:
            print(f"\n{measure.upper()}:")
            top_nodes = self.centrality_df.nlargest(5, measure)[['node', measure]]
            print(top_nodes.to_string(index=False))

        return summary

    def compute_centrality_correlations(self):
        """Compute correlations between centrality measures"""
        print("\n" + "="*60)
        print("CENTRALITY CORRELATIONS")
        print("="*60)

        centrality_cols = [col for col in self.centrality_df.columns if col != 'node']

        # Pearson correlation
        print("\nPearson Correlation Matrix:")
        pearson_corr = self.centrality_df[centrality_cols].corr()
        print(pearson_corr.round(3))

        # Spearman correlation (rank-based)
        print("\nSpearman Correlation Matrix:")
        spearman_corr = self.centrality_df[centrality_cols].corr(method='spearman')
        print(spearman_corr.round(3))

        return pearson_corr, spearman_corr

    def identify_key_nodes(self):
        """Identify key nodes using multiple centrality measures"""
        print("\n" + "="*60)
        print("KEY NODE IDENTIFICATION")
        print("="*60)

        centrality_cols = [col for col in self.centrality_df.columns if col != 'node']

        # Normalize each centrality measure
        normalized_df = self.centrality_df.copy()
        for col in centrality_cols:
            max_val = normalized_df[col].max()
            if max_val > 0:
                normalized_df[col] = normalized_df[col] / max_val

        # Composite score (average of all normalized centralities)
        normalized_df['composite_score'] = normalized_df[centrality_cols].mean(axis=1)

        # Rank by composite score
        normalized_df['rank'] = normalized_df['composite_score'].rank(ascending=False)

        # Top nodes
        print("\nTop 10 Nodes by Composite Score:")
        top_nodes = normalized_df.nlargest(10, 'composite_score')[
            ['node', 'composite_score', 'rank', 'degree', 'betweenness', 'pagerank']
        ]
        print(top_nodes.to_string(index=False))

        # Specialist nodes (high in one measure, low in others)
        print("\nSpecialist Nodes:")

        # High betweenness but low degree (bridge nodes)
        bridge_nodes = normalized_df[
            (normalized_df['betweenness'] > 0.7) & (normalized_df['degree'] < 0.5)
        ]
        if len(bridge_nodes) > 0:
            print(f"\nBridge Nodes (high betweenness, low degree):")
            print(bridge_nodes[['node', 'betweenness', 'degree']].head().to_string(index=False))

        # High degree but low betweenness (local hubs)
        local_hubs = normalized_df[
            (normalized_df['degree'] > 0.7) & (normalized_df['betweenness'] < 0.3)
        ]
        if len(local_hubs) > 0:
            print(f"\nLocal Hubs (high degree, low betweenness):")
            print(local_hubs[['node', 'degree', 'betweenness']].head().to_string(index=False))

        return normalized_df

    def compare_centrality_distributions(self):
        """Compare distributions of different centrality measures"""
        print("\n" + "="*60)
        print("CENTRALITY DISTRIBUTION ANALYSIS")
        print("="*60)

        centrality_cols = [col for col in self.centrality_df.columns if col != 'node']

        # Calculate distribution statistics
        for measure in centrality_cols:
            values = self.centrality_df[measure].values
            gini = self._gini_coefficient(values)
            skewness = pd.Series(values).skew()

            print(f"\n{measure.upper()}:")
            print(f"  Gini coefficient: {gini:.4f} (inequality)")
            print(f"  Skewness: {skewness:.4f}")
            print(f"  % of nodes above mean: {(values > values.mean()).sum() / len(values) * 100:.1f}%")

    def _gini_coefficient(self, values):
        """Calculate Gini coefficient (inequality measure)"""
        sorted_values = sorted(values)
        n = len(values)
        index = np.arange(1, n + 1)
        total = sum(values)
        if total == 0:
            return 0
        return (2 * np.sum(index * sorted_values)) / (n * total) - (n + 1) / n

    def visualize_centralities(self, normalized_df):
        """Visualize centrality measures"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Network with node sizes by composite score
        ax = axes[0, 0]

        pos = nx.spring_layout(self.G, k=0.8, iterations=50, seed=self.seed)

        # Node sizes by composite score
        node_sizes = normalized_df['composite_score'].values * 500 + 50

        # Node colors by degree
        node_colors = normalized_df['degree'].values

        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes,
                              node_color=node_colors, cmap='YlOrRd',
                              ax=ax, alpha=0.7)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray',
                              alpha=0.3, ax=ax)

        # Label top 5 nodes
        top_5 = normalized_df.nlargest(5, 'composite_score')['node'].values
        labels = {node: str(node) for node in top_5}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10,
                               font_weight='bold', ax=ax)

        ax.set_title('Network (Size=Composite Score, Color=Degree)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. Correlation heatmap
        ax = axes[0, 1]

        centrality_cols = ['degree', 'betweenness', 'closeness', 'eigenvector',
                          'pagerank']
        corr_matrix = self.centrality_df[centrality_cols].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, square=True, cbar_kws={'shrink': 0.8})
        ax.set_title('Centrality Correlation Matrix', fontsize=14, fontweight='bold')

        # 3. Centrality distributions
        ax = axes[1, 0]

        centrality_cols_plot = ['degree', 'betweenness', 'closeness', 'pagerank']
        data_to_plot = [self.centrality_df[col].values for col in centrality_cols_plot]

        bp = ax.boxplot(data_to_plot, labels=centrality_cols_plot, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)

        ax.set_ylabel('Centrality Value', fontsize=11)
        ax.set_title('Centrality Distributions', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Top nodes comparison
        ax = axes[1, 1]

        top_nodes = normalized_df.nlargest(8, 'composite_score')
        x = np.arange(len(top_nodes))
        width = 0.25

        degree_norm = top_nodes['degree'].values
        between_norm = top_nodes['betweenness'].values
        close_norm = top_nodes['closeness'].values

        ax.bar(x - width, degree_norm, width, label='Degree', alpha=0.8, color='steelblue')
        ax.bar(x, between_norm, width, label='Betweenness', alpha=0.8, color='coral')
        ax.bar(x + width, close_norm, width, label='Closeness', alpha=0.8, color='mediumseagreen')

        ax.set_ylabel('Normalized Centrality', fontsize=11)
        ax.set_title('Top 8 Nodes - Centrality Profile', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_nodes['node'].values)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('centrality_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'centrality_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("NETWORK CENTRALITY ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = CentralityAnalyzer(n_nodes=80, seed=42)

    # Generate network
    G = analyzer.generate_network()

    # Compute all centralities
    centralities = analyzer.compute_all_centralities()

    # Statistical analysis
    summary_stats = analyzer.analyze_centrality_statistics()

    # Correlations
    pearson_corr, spearman_corr = analyzer.compute_centrality_correlations()

    # Key nodes
    normalized_df = analyzer.identify_key_nodes()

    # Distribution analysis
    analyzer.compare_centrality_distributions()

    # Visualize
    analyzer.visualize_centralities(normalized_df)

    # Save results
    analyzer.centrality_df.to_csv('centrality_measures.csv', index=False)
    normalized_df.to_csv('normalized_centralities.csv', index=False)
    print("\nResults saved to CSV files")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    top_node = normalized_df.nlargest(1, 'composite_score')
    print(f"Most central node: {top_node['node'].values[0]}")
    print(f"Composite score: {top_node['composite_score'].values[0]:.4f}")
    print(f"Total centrality measures computed: 8")

if __name__ == "__main__":
    main()
