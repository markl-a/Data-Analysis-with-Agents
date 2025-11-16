"""
Community Detection - Kaggle Solution
=====================================
Detects communities in networks using multiple algorithms including Louvain.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CommunityDetector:
    """Community detection using multiple algorithms"""

    def __init__(self, n_nodes=150, n_communities=5, seed=42):
        """
        Initialize community detector

        Args:
            n_nodes: Number of nodes
            n_communities: Expected number of communities
            seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.seed = seed
        np.random.seed(seed)
        self.G = None

    def generate_network_with_communities(self):
        """Generate network with known community structure"""
        print("Generating network with community structure...")

        # Calculate nodes per community
        nodes_per_comm = self.n_nodes // self.n_communities
        sizes = [nodes_per_comm] * self.n_communities

        # Adjust for remainder
        remainder = self.n_nodes - sum(sizes)
        for i in range(remainder):
            sizes[i] += 1

        # Create stochastic block model
        # High probability within communities, low between
        p_in = 0.3  # Probability of edge within community
        p_out = 0.02  # Probability of edge between communities

        probs = [[p_out] * self.n_communities for _ in range(self.n_communities)]
        for i in range(self.n_communities):
            probs[i][i] = p_in

        self.G = nx.stochastic_block_model(sizes, probs, seed=self.seed)

        # Store ground truth communities
        ground_truth = {}
        node_id = 0
        for comm_id, size in enumerate(sizes):
            for _ in range(size):
                ground_truth[node_id] = comm_id
                node_id += 1

        nx.set_node_attributes(self.G, ground_truth, 'true_community')

        print(f"Created network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        print(f"Ground truth: {self.n_communities} communities")

        return self.G, ground_truth

    def louvain_community_detection(self):
        """
        Implement Louvain algorithm for community detection
        """
        print("\n" + "="*60)
        print("LOUVAIN COMMUNITY DETECTION")
        print("="*60)

        from networkx.algorithms import community

        # Greedy modularity communities (Louvain-like algorithm)
        communities = community.greedy_modularity_communities(self.G, weight=None)

        # Create community mapping
        louvain_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                louvain_map[node] = idx

        # Calculate modularity
        modularity = community.modularity(self.G, communities)

        print(f"Number of communities detected: {len(communities)}")
        print(f"Modularity score: {modularity:.4f}")

        # Community sizes
        community_sizes = Counter(louvain_map.values())
        print(f"\nCommunity sizes:")
        for comm_id in sorted(community_sizes.keys()):
            print(f"  Community {comm_id}: {community_sizes[comm_id]} nodes")

        return louvain_map, modularity, communities

    def label_propagation(self):
        """Label propagation algorithm"""
        print("\n" + "="*60)
        print("LABEL PROPAGATION")
        print("="*60)

        from networkx.algorithms import community

        communities = community.label_propagation_communities(self.G)
        communities = list(communities)

        lp_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                lp_map[node] = idx

        modularity = community.modularity(self.G, communities)

        print(f"Number of communities detected: {len(communities)}")
        print(f"Modularity score: {modularity:.4f}")

        return lp_map, modularity

    def girvan_newman(self, k=None):
        """Girvan-Newman algorithm (edge betweenness)"""
        print("\n" + "="*60)
        print("GIRVAN-NEWMAN (Edge Betweenness)")
        print("="*60)

        from networkx.algorithms import community

        # Run algorithm
        comp = community.girvan_newman(self.G)

        # Get k communities
        if k is None:
            k = self.n_communities

        for communities in comp:
            if len(communities) >= k:
                break

        communities = list(communities)

        gn_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                gn_map[node] = idx

        modularity = community.modularity(self.G, communities)

        print(f"Number of communities detected: {len(communities)}")
        print(f"Modularity score: {modularity:.4f}")

        return gn_map, modularity

    def evaluate_communities(self, detected_map, ground_truth):
        """
        Evaluate detected communities against ground truth

        Args:
            detected_map: Dictionary of node -> detected community
            ground_truth: Dictionary of node -> true community
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        # Convert to lists
        true_labels = [ground_truth[i] for i in range(self.n_nodes)]
        detected_labels = [detected_map[i] for i in range(self.n_nodes)]

        # Calculate metrics
        ari = adjusted_rand_score(true_labels, detected_labels)
        nmi = normalized_mutual_info_score(true_labels, detected_labels)

        return ari, nmi

    def analyze_community_structure(self, communities_map):
        """Analyze properties of detected communities"""
        print("\n" + "="*60)
        print("COMMUNITY STRUCTURE ANALYSIS")
        print("="*60)

        # Group nodes by community
        communities = defaultdict(list)
        for node, comm in communities_map.items():
            communities[comm].append(node)

        results = []

        for comm_id, nodes in communities.items():
            subgraph = self.G.subgraph(nodes)

            # Internal edges
            internal_edges = subgraph.number_of_edges()

            # External edges
            external_edges = 0
            for node in nodes:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in nodes:
                        external_edges += 1

            # Metrics
            density = nx.density(subgraph) if len(nodes) > 1 else 0
            avg_degree = np.mean([d for n, d in subgraph.degree()]) if len(nodes) > 0 else 0

            # Clustering
            try:
                avg_clustering = nx.average_clustering(subgraph)
            except:
                avg_clustering = 0

            results.append({
                'community': comm_id,
                'size': len(nodes),
                'internal_edges': internal_edges,
                'external_edges': external_edges,
                'density': density,
                'avg_degree': avg_degree,
                'avg_clustering': avg_clustering
            })

        results_df = pd.DataFrame(results).sort_values('size', ascending=False)

        print("\nCommunity Statistics:")
        print(results_df.to_string(index=False))

        return results_df

    def compare_algorithms(self, louvain_map, lp_map, gn_map, ground_truth):
        """Compare different community detection algorithms"""
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON")
        print("="*60)

        # Evaluate each algorithm
        louvain_ari, louvain_nmi = self.evaluate_communities(louvain_map, ground_truth)
        lp_ari, lp_nmi = self.evaluate_communities(lp_map, ground_truth)
        gn_ari, gn_nmi = self.evaluate_communities(gn_map, ground_truth)

        comparison = pd.DataFrame({
            'Algorithm': ['Louvain', 'Label Propagation', 'Girvan-Newman'],
            'ARI': [louvain_ari, lp_ari, gn_ari],
            'NMI': [louvain_nmi, lp_nmi, gn_nmi],
            'N_Communities': [len(set(louvain_map.values())),
                            len(set(lp_map.values())),
                            len(set(gn_map.values()))]
        })

        print("\nPerformance Comparison:")
        print(comparison.to_string(index=False))

        print("\nMetric Explanation:")
        print("  ARI (Adjusted Rand Index): -1 to 1, higher is better")
        print("  NMI (Normalized Mutual Information): 0 to 1, higher is better")

        return comparison

    def visualize_communities(self, louvain_map, ground_truth):
        """Visualize detected communities"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Layout
        pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=self.seed)

        # 1. Ground truth communities
        ax = axes[0, 0]
        colors = [ground_truth[node] for node in self.G.nodes()]
        nx.draw_networkx(self.G, pos, node_color=colors, cmap='tab10',
                        node_size=100, with_labels=False, ax=ax,
                        edge_color='gray', alpha=0.4)
        ax.set_title('Ground Truth Communities', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. Detected communities (Louvain)
        ax = axes[0, 1]
        colors = [louvain_map[node] for node in self.G.nodes()]
        nx.draw_networkx(self.G, pos, node_color=colors, cmap='tab10',
                        node_size=100, with_labels=False, ax=ax,
                        edge_color='gray', alpha=0.4)
        ax.set_title('Detected Communities (Louvain)', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 3. Community size distribution
        ax = axes[1, 0]
        true_sizes = Counter(ground_truth.values())
        detected_sizes = Counter(louvain_map.values())

        x = np.arange(max(len(true_sizes), len(detected_sizes)))
        width = 0.35

        true_counts = [true_sizes.get(i, 0) for i in range(len(x))]
        detected_counts = [detected_sizes.get(i, 0) for i in range(len(x))]

        ax.bar(x - width/2, true_counts, width, label='Ground Truth',
              alpha=0.8, color='steelblue')
        ax.bar(x + width/2, detected_counts, width, label='Detected (Louvain)',
              alpha=0.8, color='coral')

        ax.set_xlabel('Community ID', fontsize=11)
        ax.set_ylabel('Size (Number of Nodes)', fontsize=11)
        ax.set_title('Community Size Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Modularity by algorithm
        ax = axes[1, 1]
        from networkx.algorithms import community

        louvain_comm = []
        for i in range(len(set(louvain_map.values()))):
            louvain_comm.append(set([n for n, c in louvain_map.items() if c == i]))

        louvain_mod = community.modularity(self.G, louvain_comm)

        algorithms = ['Louvain', 'Ground\nTruth']
        ground_truth_comm = []
        for i in range(len(set(ground_truth.values()))):
            ground_truth_comm.append(set([n for n, c in ground_truth.items() if c == i]))
        true_mod = community.modularity(self.G, ground_truth_comm)

        modularities = [louvain_mod, true_mod]

        ax.bar(algorithms, modularities, color=['steelblue', 'coral'], alpha=0.8)
        ax.set_ylabel('Modularity Score', fontsize=11)
        ax.set_title('Modularity Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(modularities) * 1.2])

        plt.tight_layout()
        plt.savefig('community_detection_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'community_detection_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("COMMUNITY DETECTION ANALYSIS")
    print("="*60)

    # Initialize detector
    detector = CommunityDetector(n_nodes=150, n_communities=5, seed=42)

    # Generate network
    G, ground_truth = detector.generate_network_with_communities()

    # Run Louvain
    louvain_map, louvain_mod, louvain_communities = detector.louvain_community_detection()

    # Run Label Propagation
    lp_map, lp_mod = detector.label_propagation()

    # Run Girvan-Newman
    gn_map, gn_mod = detector.girvan_newman(k=5)

    # Analyze community structure
    community_stats = detector.analyze_community_structure(louvain_map)

    # Compare algorithms
    comparison = detector.compare_algorithms(louvain_map, lp_map, gn_map, ground_truth)

    # Visualize
    detector.visualize_communities(louvain_map, ground_truth)

    # Save results
    community_stats.to_csv('community_statistics.csv', index=False)
    comparison.to_csv('algorithm_comparison.csv', index=False)
    print("\nResults saved to CSV files")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Ground truth communities: {len(set(ground_truth.values()))}")
    print(f"Detected communities (Louvain): {len(set(louvain_map.values()))}")
    print(f"Best modularity: {louvain_mod:.4f}")

if __name__ == "__main__":
    main()
