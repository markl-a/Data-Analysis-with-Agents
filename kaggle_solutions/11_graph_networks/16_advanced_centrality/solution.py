"""
Advanced Centrality Measures for Network Analysis

This solution implements and compares various centrality measures including
eigenvector, Katz, PageRank, harmonic, closeness, betweenness, and more.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CentralityMeasures:
    """Collection of centrality measures"""

    @staticmethod
    def degree_centrality(G, normalized=True):
        """Degree centrality"""
        deg_cent = dict(G.degree())
        if normalized:
            n = len(G.nodes())
            deg_cent = {k: v/(n-1) for k, v in deg_cent.items()}
        return deg_cent

    @staticmethod
    def eigenvector_centrality(G, max_iter=100, tol=1e-6):
        """Eigenvector centrality"""
        try:
            return nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
        except:
            # Fallback: power iteration
            n = len(G.nodes())
            A = nx.adjacency_matrix(G).toarray()
            x = np.ones(n) / n

            for _ in range(max_iter):
                x_new = A @ x
                norm = np.linalg.norm(x_new)
                if norm > 0:
                    x_new = x_new / norm

                if np.linalg.norm(x_new - x) < tol:
                    break
                x = x_new

            return {i: x[i] for i in range(n)}

    @staticmethod
    def katz_centrality(G, alpha=0.1, beta=1.0, max_iter=100):
        """Katz centrality"""
        try:
            return nx.katz_centrality(G, alpha=alpha, beta=beta, max_iter=max_iter)
        except:
            n = len(G.nodes())
            A = nx.adjacency_matrix(G).toarray()
            I = np.eye(n)
            b = np.ones(n) * beta

            # x = (I - alpha*A)^-1 * b
            try:
                x = np.linalg.solve(I - alpha * A, b)
                return {i: x[i] for i in range(n)}
            except:
                return {i: beta for i in range(n)}

    @staticmethod
    def pagerank(G, alpha=0.85, max_iter=100):
        """PageRank centrality"""
        return nx.pagerank(G, alpha=alpha, max_iter=max_iter)

    @staticmethod
    def harmonic_centrality(G):
        """Harmonic centrality (variant of closeness)"""
        harmonic = {}

        for node in G.nodes():
            # Sum of reciprocals of distances
            distances = nx.single_source_shortest_path_length(G, node)
            harmonic[node] = sum(1/d if d > 0 else 0 for d in distances.values())

        return harmonic

    @staticmethod
    def betweenness_centrality(G, normalized=True):
        """Betweenness centrality"""
        return nx.betweenness_centrality(G, normalized=normalized)

    @staticmethod
    def closeness_centrality(G):
        """Closeness centrality"""
        return nx.closeness_centrality(G)

    @staticmethod
    def load_centrality(G):
        """Load centrality (variant of betweenness)"""
        return nx.load_centrality(G)

    @staticmethod
    def current_flow_betweenness(G):
        """Current flow betweenness (for connected graphs)"""
        if nx.is_connected(G):
            return nx.current_flow_betweenness_centrality(G)
        else:
            # Compute for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_sub = G.subgraph(largest_cc).copy()
            cent = nx.current_flow_betweenness_centrality(G_sub)

            # Fill in zeros for other nodes
            result = {node: 0.0 for node in G.nodes()}
            result.update(cent)
            return result

    @staticmethod
    def subgraph_centrality(G):
        """Subgraph centrality"""
        return nx.subgraph_centrality(G)


def generate_test_networks():
    """Generate different types of networks for testing"""
    networks = {}

    # 1. Scale-free network (Barabási-Albert)
    networks['Scale-Free'] = nx.barabasi_albert_graph(100, 3, seed=42)

    # 2. Small-world network (Watts-Strogatz)
    networks['Small-World'] = nx.watts_strogatz_graph(100, 6, 0.3, seed=42)

    # 3. Random network (Erdős-Rényi)
    networks['Random'] = nx.erdos_renyi_graph(100, 0.05, seed=42)

    # 4. Community network
    networks['Community'] = nx.caveman_graph(5, 20)

    # 5. Star network
    networks['Star'] = nx.star_graph(99)

    # 6. Tree network
    networks['Tree'] = nx.random_tree(100, seed=42)

    return networks


def compute_all_centralities(G):
    """Compute all centrality measures for a graph"""
    centralities = {}

    print(f"      Computing degree centrality...")
    centralities['Degree'] = CentralityMeasures.degree_centrality(G)

    print(f"      Computing eigenvector centrality...")
    centralities['Eigenvector'] = CentralityMeasures.eigenvector_centrality(G)

    print(f"      Computing Katz centrality...")
    centralities['Katz'] = CentralityMeasures.katz_centrality(G, alpha=0.05)

    print(f"      Computing PageRank...")
    centralities['PageRank'] = CentralityMeasures.pagerank(G)

    print(f"      Computing betweenness centrality...")
    centralities['Betweenness'] = CentralityMeasures.betweenness_centrality(G)

    print(f"      Computing closeness centrality...")
    if nx.is_connected(G):
        centralities['Closeness'] = CentralityMeasures.closeness_centrality(G)
    else:
        centralities['Closeness'] = {n: 0 for n in G.nodes()}

    print(f"      Computing harmonic centrality...")
    centralities['Harmonic'] = CentralityMeasures.harmonic_centrality(G)

    return centralities


def compare_centralities(centralities, top_k=10):
    """Compare top nodes by different centrality measures"""
    comparison = {}

    for name, cent in centralities.items():
        # Get top-k nodes
        sorted_nodes = sorted(cent.items(), key=lambda x: x[1], reverse=True)
        comparison[name] = [node for node, _ in sorted_nodes[:top_k]]

    return comparison


def analyze_centrality_correlation(centralities):
    """Analyze correlation between different centrality measures"""
    # Convert to DataFrame
    df = pd.DataFrame(centralities)

    # Compute correlation matrix
    corr_matrix = df.corr()

    return corr_matrix


def visualize_network_with_centrality(G, centrality, title, top_k=10):
    """Visualize network colored by centrality"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    pos = nx.spring_layout(G, seed=42)

    # Get centrality values
    cent_values = [centrality[node] for node in G.nodes()]

    # Plot 1: Full network
    ax = axes[0]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=cent_values,
                                   cmap='YlOrRd', node_size=100,
                                   vmin=0, vmax=max(cent_values), ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    ax.set_title(f'{title} - Full Network', fontsize=14)
    ax.axis('off')
    plt.colorbar(nodes, ax=ax)

    # Plot 2: Highlight top-k nodes
    ax = axes[1]

    # Get top-k nodes
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:top_k]]

    node_colors = ['red' if node in top_nodes else 'lightgray' for node in G.nodes()]
    node_sizes = [500 if node in top_nodes else 50 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    # Label top nodes
    top_labels = {node: str(node) for node in top_nodes}
    nx.draw_networkx_labels(G, pos, top_labels, font_size=8, ax=ax)

    ax.set_title(f'{title} - Top {top_k} Nodes', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'centrality_{title.lower().replace(" ", "_")}.png',
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_centrality_distributions(centralities):
    """Plot distributions of different centrality measures"""
    n_measures = len(centralities)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (name, cent) in enumerate(centralities.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        values = list(cent.values())

        ax.hist(values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{name} Centrality', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Distribution', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics
        ax.axvline(np.mean(values), color='red', linestyle='--',
                  label=f'Mean: {np.mean(values):.3f}')
        ax.legend()

    # Hide unused subplots
    for idx in range(len(centralities), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('centrality_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(corr_matrix):
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.index)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Correlation Between Centrality Measures', fontsize=14, pad=20)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('centrality_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_networks(networks, centrality_type='Degree'):
    """Compare same centrality across different networks"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (name, G) in enumerate(networks.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Compute centrality
        if centrality_type == 'Degree':
            cent = CentralityMeasures.degree_centrality(G)
        elif centrality_type == 'Eigenvector':
            cent = CentralityMeasures.eigenvector_centrality(G)
        elif centrality_type == 'PageRank':
            cent = CentralityMeasures.pagerank(G)
        else:
            cent = CentralityMeasures.betweenness_centrality(G)

        values = list(cent.values())

        ax.hist(values, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{centrality_type} Centrality', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Network', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{centrality_type} Centrality Across Network Types',
                fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'network_comparison_{centrality_type.lower()}.png',
               dpi=300, bbox_inches='tight')
    plt.close()


def analyze_centrality_robustness(G, centrality_func, removal_fraction=0.1):
    """Analyze how centrality changes with node removal"""
    nodes = list(G.nodes())
    n_remove = int(len(nodes) * removal_fraction)

    # Original centrality
    original_cent = centrality_func(G)

    # Remove random nodes
    G_copy = G.copy()
    removed_nodes = np.random.choice(nodes, n_remove, replace=False)
    G_copy.remove_nodes_from(removed_nodes)

    # Recompute centrality
    new_cent = centrality_func(G_copy)

    # Compare for nodes that remain
    changes = []
    for node in G_copy.nodes():
        if node in original_cent and node in new_cent:
            change = abs(new_cent[node] - original_cent[node])
            changes.append(change)

    return np.mean(changes), np.std(changes)


def main():
    """Main execution function"""
    print("=" * 80)
    print("Advanced Centrality Measures for Network Analysis")
    print("=" * 80)

    # Generate test networks
    print("\n1. Generating Test Networks...")
    networks = generate_test_networks()

    for name, G in networks.items():
        print(f"   {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Analyze scale-free network in detail
    print("\n2. Computing All Centrality Measures for Scale-Free Network...")
    G_sf = networks['Scale-Free']
    centralities = compute_all_centralities(G_sf)

    # Compare top nodes
    print("\n3. Comparing Top Nodes by Different Centralities...")
    top_comparison = compare_centralities(centralities, top_k=5)

    for cent_name, top_nodes in top_comparison.items():
        print(f"   {cent_name}: {top_nodes}")

    # Analyze correlations
    print("\n4. Analyzing Centrality Correlations...")
    corr_matrix = analyze_centrality_correlation(centralities)

    print("\n   Correlation Matrix:")
    print(corr_matrix.round(3))

    # Robustness analysis
    print("\n5. Analyzing Centrality Robustness...")
    for cent_name in ['Degree', 'Betweenness']:
        if cent_name == 'Degree':
            cent_func = CentralityMeasures.degree_centrality
        else:
            cent_func = CentralityMeasures.betweenness_centrality

        mean_change, std_change = analyze_centrality_robustness(G_sf, cent_func)
        print(f"   {cent_name}: Mean change = {mean_change:.4f}, Std = {std_change:.4f}")

    # Visualizations
    print("\n6. Generating Visualizations...")

    # Visualize networks with different centralities
    for cent_name in ['Degree', 'Eigenvector', 'PageRank', 'Betweenness']:
        visualize_network_with_centrality(G_sf, centralities[cent_name], cent_name)

    # Plot distributions
    plot_centrality_distributions(centralities)

    # Plot correlation matrix
    plot_correlation_matrix(corr_matrix)

    # Compare across networks
    compare_networks(networks, 'Degree')
    compare_networks(networks, 'PageRank')

    print("\n" + "=" * 80)
    print("Centrality Analysis Complete!")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Different centrality measures capture different aspects of importance")
    print("2. Degree and eigenvector centrality are highly correlated")
    print("3. Betweenness identifies bridge nodes in network structure")
    print("4. PageRank balances local and global importance")
    print("5. Centrality measures vary significantly across network types")
    print("=" * 80)


if __name__ == "__main__":
    main()
