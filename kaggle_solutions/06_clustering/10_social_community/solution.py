"""
Social Network Community Detection
Clustering users in social networks to discover communities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class SocialNetworkClustering:
    """Detect communities in social networks using clustering"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        np.random.seed(random_state)

    def generate_social_network(self, n_users=200, n_communities=5):
        """
        Generate a synthetic social network with community structure
        Features: connections, posts, likes, comments, followers, following
        """
        users_per_community = n_users // n_communities
        users = []
        edges = []

        # Generate users in communities
        for comm_id in range(n_communities):
            start_idx = comm_id * users_per_community
            end_idx = start_idx + users_per_community

            # Community characteristics vary
            if comm_id == 0:  # Tech enthusiasts
                activity_level = 'high'
                avg_posts = 50
                avg_connections = 150
            elif comm_id == 1:  # Casual users
                activity_level = 'medium'
                avg_posts = 20
                avg_connections = 80
            elif comm_id == 2:  # Influencers
                activity_level = 'very_high'
                avg_posts = 100
                avg_connections = 300
            elif comm_id == 3:  # Lurkers
                activity_level = 'low'
                avg_posts = 5
                avg_connections = 30
            else:  # Content creators
                activity_level = 'high'
                avg_posts = 80
                avg_connections = 200

            for i in range(start_idx, min(end_idx, n_users)):
                user = {
                    'user_id': f'USER_{i:04d}',
                    'posts_count': max(1, int(np.random.normal(avg_posts, avg_posts * 0.3))),
                    'likes_given': max(1, int(np.random.normal(avg_posts * 5, avg_posts * 2))),
                    'comments_made': max(1, int(np.random.normal(avg_posts * 2, avg_posts * 0.8))),
                    'followers_count': max(1, int(np.random.normal(avg_connections, avg_connections * 0.4))),
                    'following_count': max(1, int(np.random.normal(avg_connections * 0.8, avg_connections * 0.3))),
                    'true_community': comm_id,
                    'activity_level': activity_level
                }
                users.append(user)

            # Create edges within community (high intra-community connectivity)
            community_members = list(range(start_idx, min(end_idx, n_users)))
            for i in community_members:
                # Connect to random members within community
                n_connections = min(len(community_members) - 1,
                                  int(np.random.normal(len(community_members) * 0.4, 5)))
                connections = np.random.choice(
                    [u for u in community_members if u != i],
                    size=max(1, n_connections),
                    replace=False
                )
                for j in connections:
                    if i < j:  # Avoid duplicates
                        edges.append({'source': i, 'target': j, 'weight': 1.0})

        # Add inter-community edges (bridge connections)
        n_bridges = int(n_users * 0.3)  # 30% cross-community connections
        for _ in range(n_bridges):
            u1 = np.random.randint(0, n_users)
            u2 = np.random.randint(0, n_users)
            if u1 != u2 and u1 < u2:
                edges.append({'source': u1, 'target': u2, 'weight': 0.5})

        df_users = pd.DataFrame(users)
        df_edges = pd.DataFrame(edges)

        # Calculate derived features
        df_users['engagement_ratio'] = df_users['likes_given'] / (df_users['posts_count'] + 1)
        df_users['interaction_ratio'] = df_users['comments_made'] / (df_users['posts_count'] + 1)
        df_users['follower_ratio'] = df_users['followers_count'] / (df_users['following_count'] + 1)

        return df_users, df_edges

    def create_network_features(self, df_users, df_edges):
        """Extract network-based features using NetworkX"""
        # Create graph
        G = nx.Graph()

        # Add nodes
        for idx, row in df_users.iterrows():
            G.add_node(idx)

        # Add edges
        for _, edge in df_edges.iterrows():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

        # Calculate network metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(50, len(G)))
        clustering_coef = nx.clustering(G)
        pagerank = nx.pagerank(G)

        # Add to dataframe
        df_users['degree_centrality'] = df_users.index.map(degree_centrality)
        df_users['betweenness_centrality'] = df_users.index.map(betweenness_centrality)
        df_users['clustering_coefficient'] = df_users.index.map(clustering_coef)
        df_users['pagerank'] = df_users.index.map(pagerank)

        return df_users, G

    def visualize_network(self, G, df_users, labels=None, title="Social Network"):
        """Visualize the social network graph"""
        plt.figure(figsize=(14, 10))

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=self.random_state)

        # Color by cluster if labels provided, otherwise by true community
        if labels is not None:
            colors = labels
            cmap = plt.cm.Set3
        else:
            colors = df_users['true_community'].values
            cmap = plt.cm.Set2

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=100,
                              cmap=cmap, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('/tmp/social_network_graph.png', dpi=300, bbox_inches='tight')
        plt.show()

    def optimal_communities_analysis(self, X, max_k=10):
        """Find optimal number of communities"""
        silhouette_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Communities', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Optimal Number of Communities', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/tmp/social_optimal_k.png', dpi=300, bbox_inches='tight')
        plt.show()

        return K_range[np.argmax(silhouette_scores)]

    def compare_clustering_methods(self, X, n_clusters=5):
        """Compare different community detection methods"""
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

        # Spectral Clustering (good for graph data)
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=self.random_state,
                                     affinity='nearest_neighbors', n_neighbors=10)
        spectral_labels = spectral.fit_predict(X)
        results['Spectral'] = {
            'labels': spectral_labels,
            'silhouette': silhouette_score(X, spectral_labels),
            'davies_bouldin': davies_bouldin_score(X, spectral_labels),
            'calinski_harabasz': calinski_harabasz_score(X, spectral_labels)
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
        """Visualize community clusters using PCA"""
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
                                       s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
            axes[idx].set_title(f'{algo_name}\nSilhouette: {result["silhouette"]:.3f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
            axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Community')

        plt.tight_layout()
        plt.savefig('/tmp/social_communities_pca.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_communities(self, df_users, labels):
        """Analyze characteristics of each community"""
        df_analysis = df_users.copy()
        df_analysis['detected_community'] = labels

        print("\n" + "="*90)
        print("COMMUNITY ANALYSIS")
        print("="*90)

        for comm_id in sorted(set(labels)):
            community = df_analysis[df_analysis['detected_community'] == comm_id]
            print(f"\n{'='*90}")
            print(f"COMMUNITY {comm_id} - {len(community)} users "
                  f"({len(community)/len(df_users)*100:.1f}%)")
            print(f"{'='*90}")

            # Key metrics
            metrics = ['posts_count', 'likes_given', 'comments_made',
                      'followers_count', 'following_count', 'degree_centrality',
                      'betweenness_centrality', 'pagerank']

            for metric in metrics:
                if metric in community.columns:
                    mean_val = community[metric].mean()
                    median_val = community[metric].median()
                    print(f"{metric:30s}: mean={mean_val:10.3f}, median={median_val:10.3f}")

            # Activity level distribution
            if 'activity_level' in community.columns:
                print(f"\nActivity level distribution:")
                activity_dist = community['activity_level'].value_counts()
                for level, count in activity_dist.items():
                    print(f"  {level:20s}: {count:4d} ({count/len(community)*100:.1f}%)")

    def create_community_comparison(self, df_users, labels):
        """Create comparative visualizations for communities"""
        df_plot = df_users.copy()
        df_plot['community'] = labels

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Posts by community
        sns.boxplot(data=df_plot, x='community', y='posts_count', ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_title('Posts Count by Community', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Posts')

        # Followers by community
        sns.boxplot(data=df_plot, x='community', y='followers_count', ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Followers by Community', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Followers')

        # Degree centrality by community
        sns.boxplot(data=df_plot, x='community', y='degree_centrality', ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_title('Network Centrality by Community', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Degree Centrality')

        # PageRank by community
        sns.boxplot(data=df_plot, x='community', y='pagerank', ax=axes[1, 1], palette='Set2')
        axes[1, 1].set_title('Influence (PageRank) by Community', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('PageRank Score')

        plt.tight_layout()
        plt.savefig('/tmp/community_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    print("="*90)
    print("SOCIAL NETWORK COMMUNITY DETECTION")
    print("="*90)

    # Initialize
    clustering = SocialNetworkClustering(random_state=42)

    # Generate social network
    print("\n[1/6] Generating social network...")
    df_users, df_edges = clustering.generate_social_network(n_users=200, n_communities=5)
    print(f"Generated {len(df_users)} users and {len(df_edges)} connections")
    print(f"\nTrue community distribution:")
    print(df_users['true_community'].value_counts().sort_index())
    print(f"\nSample users:")
    print(df_users[['user_id', 'posts_count', 'followers_count', 'activity_level']].head(10))

    # Extract network features
    print("\n[2/6] Extracting network features...")
    df_users, G = clustering.create_network_features(df_users, df_edges)
    print(f"Network stats:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")
    print(f"  Average clustering: {nx.average_clustering(G):.4f}")

    # Visualize network
    print("\n[3/6] Visualizing social network...")
    clustering.visualize_network(G, df_users, title="Social Network (Colored by True Community)")

    # Prepare features
    feature_cols = ['posts_count', 'likes_given', 'comments_made', 'followers_count',
                   'following_count', 'engagement_ratio', 'interaction_ratio',
                   'follower_ratio', 'degree_centrality', 'betweenness_centrality',
                   'clustering_coefficient', 'pagerank']
    X = clustering.scaler.fit_transform(df_users[feature_cols])

    # Find optimal communities
    print("\n[4/6] Finding optimal number of communities...")
    optimal_k = clustering.optimal_communities_analysis(X, max_k=10)
    print(f"Suggested optimal communities: {optimal_k}")

    # Compare methods
    print(f"\n[5/6] Comparing community detection methods with k={optimal_k}...")
    results = clustering.compare_clustering_methods(X, n_clusters=optimal_k)

    print("\nClustering Performance:")
    print("-" * 90)
    print(f"{'Method':<15} {'Silhouette':>12} {'Davies-Bouldin':>17} {'Calinski-Harabasz':>20}")
    print("-" * 90)
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['silhouette']:>12.4f} "
              f"{metrics['davies_bouldin']:>17.4f} {metrics['calinski_harabasz']:>20.2f}")

    # Visualize communities
    print("\n[6/6] Analyzing detected communities...")
    clustering.visualize_clusters(X, results)

    # Analyze best method
    best_method = max(results.items(), key=lambda x: x[1]['silhouette'])
    clustering.analyze_communities(df_users, best_method[1]['labels'])

    # Create comparison plots
    clustering.create_community_comparison(df_users, best_method[1]['labels'])

    # Visualize network with detected communities
    clustering.visualize_network(G, df_users, labels=best_method[1]['labels'],
                                title=f"Social Network (Colored by {best_method[0]} Communities)")

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE!")
    print("="*90)


if __name__ == "__main__":
    main()
