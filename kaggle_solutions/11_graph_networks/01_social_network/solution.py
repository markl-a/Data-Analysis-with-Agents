"""
Social Network Analysis - Kaggle Solution
==========================================
Analyzes social network connections, communities, and user influence.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class SocialNetworkAnalyzer:
    """Comprehensive social network analysis toolkit"""

    def __init__(self, n_users=100, seed=42):
        """
        Initialize social network analyzer

        Args:
            n_users: Number of users in the network
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.seed = seed
        np.random.seed(seed)
        self.G = None
        self.user_data = None

    def generate_social_network(self):
        """Generate realistic social network with communities"""
        print("Generating social network...")

        # Create network with multiple communities
        # Use Barabasi-Albert model for scale-free network
        self.G = nx.barabasi_albert_graph(self.n_users, 3, seed=self.seed)

        # Add some random edges for realism
        for _ in range(self.n_users // 2):
            u, v = np.random.choice(self.n_users, 2, replace=False)
            self.G.add_edge(u, v)

        # Generate user attributes
        user_names = [f"User_{i}" for i in range(self.n_users)]
        ages = np.random.randint(18, 65, self.n_users)
        join_dates = pd.date_range('2020-01-01', periods=self.n_users, freq='D')

        # Calculate initial metrics
        degrees = dict(self.G.degree())

        self.user_data = pd.DataFrame({
            'user_id': range(self.n_users),
            'name': user_names,
            'age': ages,
            'join_date': join_dates,
            'connections': [degrees[i] for i in range(self.n_users)]
        })

        print(f"Created network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G, self.user_data

    def analyze_network_properties(self):
        """Analyze basic network properties"""
        print("\n" + "="*60)
        print("NETWORK PROPERTIES")
        print("="*60)

        # Basic stats
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        density = nx.density(self.G)

        print(f"Number of nodes: {n_nodes}")
        print(f"Number of edges: {n_edges}")
        print(f"Network density: {density:.4f}")

        # Degree statistics
        degrees = [d for n, d in self.G.degree()]
        print(f"\nDegree Statistics:")
        print(f"  Average degree: {np.mean(degrees):.2f}")
        print(f"  Max degree: {np.max(degrees)}")
        print(f"  Min degree: {np.min(degrees)}")

        # Connectivity
        is_connected = nx.is_connected(self.G)
        print(f"\nNetwork connected: {is_connected}")

        if not is_connected:
            n_components = nx.number_connected_components(self.G)
            largest_cc = max(nx.connected_components(self.G), key=len)
            print(f"Number of components: {n_components}")
            print(f"Largest component size: {len(largest_cc)}")

        # Clustering
        avg_clustering = nx.average_clustering(self.G)
        print(f"\nAverage clustering coefficient: {avg_clustering:.4f}")

        # Diameter (for largest component if disconnected)
        if is_connected:
            diameter = nx.diameter(self.G)
            avg_shortest_path = nx.average_shortest_path_length(self.G)
            print(f"Network diameter: {diameter}")
            print(f"Average shortest path: {avg_shortest_path:.4f}")
        else:
            largest_cc_graph = self.G.subgraph(largest_cc)
            diameter = nx.diameter(largest_cc_graph)
            print(f"Diameter (largest component): {diameter}")

        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'avg_degree': np.mean(degrees),
            'avg_clustering': avg_clustering
        }

    def find_influential_users(self):
        """Identify influential users using various centrality measures"""
        print("\n" + "="*60)
        print("INFLUENTIAL USERS ANALYSIS")
        print("="*60)

        # Degree centrality (most connections)
        degree_cent = nx.degree_centrality(self.G)

        # Betweenness centrality (bridge users)
        betweenness_cent = nx.betweenness_centrality(self.G)

        # Closeness centrality (reach to others)
        closeness_cent = nx.closeness_centrality(self.G)

        # Eigenvector centrality (influence)
        eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=1000)

        # PageRank
        pagerank = nx.pagerank(self.G)

        # Create centrality dataframe
        centrality_df = pd.DataFrame({
            'user_id': range(self.n_users),
            'degree_centrality': [degree_cent[i] for i in range(self.n_users)],
            'betweenness_centrality': [betweenness_cent[i] for i in range(self.n_users)],
            'closeness_centrality': [closeness_cent[i] for i in range(self.n_users)],
            'eigenvector_centrality': [eigenvector_cent[i] for i in range(self.n_users)],
            'pagerank': [pagerank[i] for i in range(self.n_users)]
        })

        # Merge with user data
        self.user_data = self.user_data.merge(centrality_df, on='user_id')

        # Top influential users
        print("\nTop 5 Users by Different Metrics:")
        print("\n1. Degree Centrality (Most Connections):")
        top_degree = self.user_data.nlargest(5, 'degree_centrality')[['name', 'connections', 'degree_centrality']]
        print(top_degree.to_string(index=False))

        print("\n2. Betweenness Centrality (Key Bridges):")
        top_between = self.user_data.nlargest(5, 'betweenness_centrality')[['name', 'betweenness_centrality']]
        print(top_between.to_string(index=False))

        print("\n3. PageRank (Overall Influence):")
        top_pagerank = self.user_data.nlargest(5, 'pagerank')[['name', 'pagerank']]
        print(top_pagerank.to_string(index=False))

        return centrality_df

    def detect_communities(self):
        """Detect communities in the network"""
        print("\n" + "="*60)
        print("COMMUNITY DETECTION")
        print("="*60)

        # Louvain community detection
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(self.G)

        # Create community mapping
        community_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                community_map[node] = idx

        # Add to graph
        nx.set_node_attributes(self.G, community_map, 'community')

        # Add to user data
        self.user_data['community'] = self.user_data['user_id'].map(community_map)

        print(f"Number of communities detected: {len(communities)}")
        print("\nCommunity sizes:")
        community_sizes = Counter(community_map.values())
        for comm_id, size in sorted(community_sizes.items()):
            print(f"  Community {comm_id}: {size} users")

        # Modularity score
        modularity = community.modularity(self.G, communities)
        print(f"\nModularity score: {modularity:.4f}")

        return communities, community_map

    def visualize_network(self, communities=None):
        """Visualize the social network"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Network graph with communities
        ax = axes[0, 0]
        pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=self.seed)

        if communities:
            community_map = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = idx

            colors = [community_map[node] for node in self.G.nodes()]
            nx.draw_networkx(self.G, pos, node_color=colors, cmap='tab10',
                           node_size=100, with_labels=False, ax=ax,
                           edge_color='gray', alpha=0.6)
            ax.set_title('Social Network with Communities', fontsize=14, fontweight='bold')
        else:
            nx.draw_networkx(self.G, pos, node_size=100, with_labels=False,
                           ax=ax, node_color='lightblue', edge_color='gray', alpha=0.6)
            ax.set_title('Social Network Graph', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. Degree distribution
        ax = axes[0, 1]
        degrees = [d for n, d in self.G.degree()]
        ax.hist(degrees, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Degree', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Degree Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Centrality comparison
        ax = axes[1, 0]
        top_users = self.user_data.nlargest(10, 'pagerank')
        x_pos = np.arange(len(top_users))
        width = 0.35

        ax.barh(x_pos - width/2, top_users['degree_centrality'], width,
               label='Degree', alpha=0.8, color='steelblue')
        ax.barh(x_pos + width/2, top_users['betweenness_centrality'], width,
               label='Betweenness', alpha=0.8, color='coral')

        ax.set_yticks(x_pos)
        ax.set_yticklabels(top_users['name'], fontsize=9)
        ax.set_xlabel('Centrality Score', fontsize=11)
        ax.set_title('Top 10 Users - Centrality Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # 4. Community size distribution
        ax = axes[1, 1]
        if 'community' in self.user_data.columns:
            comm_sizes = self.user_data['community'].value_counts().sort_index()
            ax.bar(comm_sizes.index, comm_sizes.values, edgecolor='black',
                  alpha=0.7, color='mediumseagreen')
            ax.set_xlabel('Community ID', fontsize=11)
            ax.set_ylabel('Number of Users', fontsize=11)
            ax.set_title('Community Size Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('social_network_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'social_network_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("SOCIAL NETWORK ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = SocialNetworkAnalyzer(n_users=100, seed=42)

    # Generate network
    G, user_data = analyzer.generate_social_network()

    # Analyze properties
    properties = analyzer.analyze_network_properties()

    # Find influential users
    centrality_df = analyzer.find_influential_users()

    # Detect communities
    communities, community_map = analyzer.detect_communities()

    # Visualize
    analyzer.visualize_network(communities)

    # Save results
    user_data_export = analyzer.user_data.copy()
    user_data_export.to_csv('social_network_users.csv', index=False)
    print("\nUser data saved to 'social_network_users.csv'")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total users analyzed: {len(user_data_export)}")
    print(f"Total connections: {G.number_of_edges()}")
    print(f"Communities found: {len(communities)}")
    print("\nKey insights:")
    print(f"- Average connections per user: {properties['avg_degree']:.2f}")
    print(f"- Network clustering: {properties['avg_clustering']:.4f}")
    print(f"- Network density: {properties['density']:.4f}")

if __name__ == "__main__":
    main()
