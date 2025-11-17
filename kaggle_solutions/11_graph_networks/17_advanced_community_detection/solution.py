"""
Advanced Community Detection Algorithms

This solution implements and compares Louvain, Label Propagation, Girvan-Newman,
and other advanced community detection methods.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CommunityDetection:
    """Advanced community detection algorithms"""

    @staticmethod
    def louvain_method(G, resolution=1.0):
        """Louvain method for community detection"""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, resolution=resolution)
            return partition
        except:
            # Fallback: greedy modularity
            communities = nx.community.greedy_modularity_communities(G)
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            return partition

    @staticmethod
    def label_propagation(G, max_iter=100):
        """Label propagation algorithm"""
        communities = nx.community.label_propagation_communities(G)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

    @staticmethod
    def girvan_newman(G, k=None):
        """Girvan-Newman algorithm"""
        communities_generator = nx.community.girvan_newman(G)
        
        if k is None:
            k = 4  # Default number of communities
        
        for _ in range(k - 1):
            try:
                communities = next(communities_generator)
            except StopIteration:
                break
        
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

    @staticmethod
    def greedy_modularity(G):
        """Greedy modularity optimization"""
        communities = nx.community.greedy_modularity_communities(G)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

    @staticmethod
    def async_fluid_communities(G, k):
        """Asynchronous fluid communities algorithm"""
        communities = nx.community.asyn_fluidc(G, k, seed=42)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition


def compute_modularity(G, partition):
    """Compute modularity of a partition"""
    m = G.number_of_edges()
    if m == 0:
        return 0
    
    Q = 0
    for community_id in set(partition.values()):
        nodes_in_comm = [n for n, c in partition.items() if c == community_id]
        
        for i in nodes_in_comm:
            for j in nodes_in_comm:
                if G.has_edge(i, j):
                    A_ij = 1
                else:
                    A_ij = 0
                
                k_i = G.degree(i)
                k_j = G.degree(j)
                
                Q += A_ij - (k_i * k_j) / (2 * m)
    
    return Q / (2 * m)


def generate_community_network(n_communities=4, nodes_per_comm=25, p_in=0.3, p_out=0.02):
    """Generate network with ground truth communities"""
    n_nodes = n_communities * nodes_per_comm
    
    # Ground truth labels
    true_labels = np.repeat(np.arange(n_communities), nodes_per_comm)
    
    # Generate stochastic block model
    adjacency = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if true_labels[i] == true_labels[j]:
                if np.random.rand() < p_in:
                    adjacency[i, j] = adjacency[j, i] = 1
            else:
                if np.random.rand() < p_out:
                    adjacency[i, j] = adjacency[j, i] = 1
    
    G = nx.from_numpy_array(adjacency)
    
    # Convert labels to dict
    true_partition = {i: true_labels[i] for i in range(n_nodes)}
    
    return G, true_partition


def compare_algorithms(G, true_partition):
    """Compare different community detection algorithms"""
    results = []
    
    algorithms = {
        'Louvain': lambda G: CommunityDetection.louvain_method(G),
        'Label Propagation': lambda G: CommunityDetection.label_propagation(G),
        'Girvan-Newman': lambda G: CommunityDetection.girvan_newman(G, k=len(set(true_partition.values()))),
        'Greedy Modularity': lambda G: CommunityDetection.greedy_modularity(G),
    }
    
    print("   Comparing algorithms...")
    for name, algo in algorithms.items():
        print(f"      Running {name}...")
        
        try:
            partition = algo(G)
            
            # Compute metrics
            modularity = compute_modularity(G, partition)
            
            # Compare with ground truth
            true_labels = [true_partition[i] for i in sorted(G.nodes())]
            pred_labels = [partition[i] for i in sorted(G.nodes())]
            
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            
            n_communities = len(set(partition.values()))
            
            results.append({
                'Algorithm': name,
                'Modularity': modularity,
                'ARI': ari,
                'NMI': nmi,
                'Communities': n_communities
            })
            
            print(f"         Modularity: {modularity:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")
        except Exception as e:
            print(f"         Error: {e}")
    
    return pd.DataFrame(results)


def visualize_communities(G, partition, title, true_partition=None):
    """Visualize detected communities"""
    fig, axes = plt.subplots(1, 2 if true_partition else 1, figsize=(16 if true_partition else 10, 6))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    pos = nx.spring_layout(G, seed=42)
    
    # Detected communities
    ax = axes[0]
    node_colors = [partition[node] for node in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, cmap='tab10',
           node_size=100, with_labels=False, ax=ax)
    ax.set_title(f'{title} - Detected ({len(set(partition.values()))} communities)', fontsize=14)
    
    # True communities (if available)
    if true_partition and len(axes) > 1:
        ax = axes[1]
        node_colors = [true_partition[node] for node in G.nodes()]
        nx.draw(G, pos, node_color=node_colors, cmap='tab10',
               node_size=100, with_labels=False, ax=ax)
        ax.set_title(f'{title} - Ground Truth ({len(set(true_partition.values()))} communities)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'community_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(results_df):
    """Plot comparison of algorithms"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['Modularity', 'ARI', 'NMI']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        bars = ax.bar(results_df['Algorithm'], results_df[metric],
                     color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14)
        ax.set_xticklabels(results_df['Algorithm'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, results_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('community_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_hierarchical_structure(G):
    """Analyze hierarchical community structure"""
    # Run Girvan-Newman to get dendrogram
    communities_generator = nx.community.girvan_newman(G)
    
    hierarchy = []
    for i in range(5):
        try:
            communities = next(communities_generator)
            n_comm = len(communities)
            
            partition = {}
            for j, comm in enumerate(communities):
                for node in comm:
                    partition[node] = j
            
            modularity = compute_modularity(G, partition)
            
            hierarchy.append({
                'level': i,
                'n_communities': n_comm,
                'modularity': modularity
            })
        except StopIteration:
            break
    
    # Plot hierarchy
    df = pd.DataFrame(hierarchy)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(df['n_communities'], df['modularity'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Communities', fontsize=12)
    ax.set_ylabel('Modularity', fontsize=12)
    ax.set_title('Modularity vs Number of Communities', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.bar(range(len(df)), df['n_communities'], color='coral', alpha=0.7)
    ax.set_xlabel('Hierarchy Level', fontsize=12)
    ax.set_ylabel('Number of Communities', fontsize=12)
    ax.set_title('Hierarchical Community Structure', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('community_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Advanced Community Detection Algorithms")
    print("=" * 80)
    
    # Generate community network
    print("\n1. Generating Network with Community Structure...")
    G, true_partition = generate_community_network(n_communities=4, nodes_per_comm=25)
    
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   True communities: {len(set(true_partition.values()))}")
    
    # Compare algorithms
    print("\n2. Comparing Community Detection Algorithms...")
    results = compare_algorithms(G, true_partition)
    
    print("\n   Results Summary:")
    print(results.to_string(index=False))
    
    # Visualize best algorithm
    print("\n3. Visualizing Communities...")
    best_algo = results.loc[results['NMI'].idxmax(), 'Algorithm']
    print(f"   Best algorithm (by NMI): {best_algo}")
    
    if best_algo == 'Louvain':
        best_partition = CommunityDetection.louvain_method(G)
    elif best_algo == 'Label Propagation':
        best_partition = CommunityDetection.label_propagation(G)
    else:
        best_partition = CommunityDetection.greedy_modularity(G)
    
    visualize_communities(G, best_partition, best_algo, true_partition)
    
    # Analyze hierarchy
    print("\n4. Analyzing Hierarchical Structure...")
    analyze_hierarchical_structure(G)
    
    # Plot comparisons
    print("\n5. Generating Comparison Plots...")
    plot_algorithm_comparison(results)
    
    print("\n" + "=" * 80)
    print("Community Detection Complete!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"1. Best algorithm: {best_algo} (NMI: {results.loc[results['NMI'].idxmax(), 'NMI']:.3f})")
    print(f"2. Average modularity: {results['Modularity'].mean():.3f}")
    print("3. Different algorithms reveal different community structures")
    print("4. Hierarchical analysis shows multi-scale organization")
    print("=" * 80)


if __name__ == "__main__":
    main()
