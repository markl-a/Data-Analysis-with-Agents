"""
PageRank Implementation - Kaggle Solution
=========================================
Implements and analyzes Google's PageRank algorithm for web page ranking.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class PageRankAnalyzer:
    """PageRank implementation and analysis toolkit"""

    def __init__(self, n_pages=50, seed=42):
        """
        Initialize PageRank analyzer

        Args:
            n_pages: Number of web pages
            seed: Random seed for reproducibility
        """
        self.n_pages = n_pages
        self.seed = seed
        np.random.seed(seed)
        self.G = None
        self.page_data = None

    def generate_web_graph(self):
        """Generate directed web graph with links"""
        print("Generating web graph...")

        # Create directed graph (web pages with links)
        self.G = nx.DiGraph()

        # Add nodes (web pages)
        page_names = [f"page_{i}.html" for i in range(self.n_pages)]
        topics = np.random.choice(['tech', 'science', 'sports', 'news', 'entertainment'],
                                 self.n_pages, p=[0.3, 0.2, 0.2, 0.15, 0.15])

        for i in range(self.n_pages):
            self.G.add_node(i, name=page_names[i], topic=topics[i])

        # Add edges (links between pages)
        # Pages tend to link to pages in same topic
        for i in range(self.n_pages):
            # Number of outgoing links
            n_links = np.random.randint(1, 8)

            # Prefer same-topic pages (80%) vs random (20%)
            same_topic_pages = [j for j in range(self.n_pages)
                               if j != i and topics[j] == topics[i]]
            different_topic_pages = [j for j in range(self.n_pages)
                                    if j != i and topics[j] != topics[i]]

            links = []
            for _ in range(n_links):
                if np.random.rand() < 0.8 and same_topic_pages:
                    target = np.random.choice(same_topic_pages)
                elif different_topic_pages:
                    target = np.random.choice(different_topic_pages)
                else:
                    continue

                if target not in links:  # Avoid duplicate links
                    links.append(target)

            for target in links:
                self.G.add_edge(i, target)

        # Create page dataframe
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())

        self.page_data = pd.DataFrame({
            'page_id': range(self.n_pages),
            'name': page_names,
            'topic': topics,
            'incoming_links': [in_degrees[i] for i in range(self.n_pages)],
            'outgoing_links': [out_degrees[i] for i in range(self.n_pages)]
        })

        print(f"Created web graph with {self.G.number_of_nodes()} pages and {self.G.number_of_edges()} links")
        return self.G, self.page_data

    def custom_pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        """
        Custom PageRank implementation from scratch

        Args:
            damping: Damping factor (probability of following link)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            dict: PageRank scores for each node
        """
        print("\nRunning custom PageRank implementation...")

        N = self.G.number_of_nodes()

        # Initialize PageRank scores uniformly
        pagerank = {node: 1.0 / N for node in self.G.nodes()}

        # Get outgoing links for each node
        out_links = {node: list(self.G.successors(node)) for node in self.G.nodes()}

        for iteration in range(max_iter):
            new_pagerank = {}

            for node in self.G.nodes():
                # Random surfer component
                rank = (1 - damping) / N

                # Link following component
                for predecessor in self.G.predecessors(node):
                    n_out = len(out_links[predecessor])
                    if n_out > 0:
                        rank += damping * pagerank[predecessor] / n_out

                new_pagerank[node] = rank

            # Check convergence
            diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in self.G.nodes())
            if diff < tol:
                print(f"Converged after {iteration + 1} iterations")
                break

            pagerank = new_pagerank

        return pagerank

    def analyze_pagerank(self, damping=0.85):
        """Analyze PageRank results"""
        print("\n" + "="*60)
        print("PAGERANK ANALYSIS")
        print("="*60)

        # Custom implementation
        custom_pr = self.custom_pagerank(damping=damping)

        # NetworkX implementation (for comparison)
        nx_pr = nx.pagerank(self.G, alpha=damping)

        # Add to page data
        self.page_data['pagerank_custom'] = [custom_pr[i] for i in range(self.n_pages)]
        self.page_data['pagerank_nx'] = [nx_pr[i] for i in range(self.n_pages)]

        # Verify implementations match
        diff = np.abs(self.page_data['pagerank_custom'] - self.page_data['pagerank_nx']).max()
        print(f"\nMax difference between implementations: {diff:.10f}")

        # Statistics
        print(f"\nPageRank Statistics:")
        print(f"  Mean PageRank: {self.page_data['pagerank_nx'].mean():.6f}")
        print(f"  Std PageRank: {self.page_data['pagerank_nx'].std():.6f}")
        print(f"  Max PageRank: {self.page_data['pagerank_nx'].max():.6f}")
        print(f"  Min PageRank: {self.page_data['pagerank_nx'].min():.6f}")

        # Top pages
        print("\nTop 10 Pages by PageRank:")
        top_pages = self.page_data.nlargest(10, 'pagerank_nx')[
            ['name', 'topic', 'incoming_links', 'pagerank_nx']
        ]
        print(top_pages.to_string(index=False))

        # PageRank by topic
        print("\nAverage PageRank by Topic:")
        topic_pr = self.page_data.groupby('topic')['pagerank_nx'].agg(['mean', 'sum', 'count'])
        topic_pr = topic_pr.sort_values('mean', ascending=False)
        print(topic_pr)

        return custom_pr, nx_pr

    def compare_metrics(self):
        """Compare PageRank with other centrality measures"""
        print("\n" + "="*60)
        print("COMPARING PAGERANK WITH OTHER METRICS")
        print("="*60)

        # In-degree centrality (simple link count)
        in_degree_cent = nx.in_degree_centrality(self.G)

        # HITS algorithm
        hubs, authorities = nx.hits(self.G)

        # Add to dataframe
        self.page_data['in_degree_centrality'] = [in_degree_cent[i] for i in range(self.n_pages)]
        self.page_data['authority_score'] = [authorities[i] for i in range(self.n_pages)]
        self.page_data['hub_score'] = [hubs[i] for i in range(self.n_pages)]

        # Correlations
        print("\nCorrelation Matrix:")
        corr_cols = ['pagerank_nx', 'in_degree_centrality', 'authority_score', 'hub_score']
        correlations = self.page_data[corr_cols].corr()
        print(correlations.round(4))

        # Top pages by different metrics
        print("\nTop 5 Pages by Different Metrics:")

        print("\n1. PageRank:")
        print(self.page_data.nlargest(5, 'pagerank_nx')[['name', 'pagerank_nx']].to_string(index=False))

        print("\n2. In-Degree (Incoming Links):")
        print(self.page_data.nlargest(5, 'in_degree_centrality')[['name', 'incoming_links']].to_string(index=False))

        print("\n3. Authority Score (HITS):")
        print(self.page_data.nlargest(5, 'authority_score')[['name', 'authority_score']].to_string(index=False))

        print("\n4. Hub Score (HITS):")
        print(self.page_data.nlargest(5, 'hub_score')[['name', 'hub_score']].to_string(index=False))

    def test_damping_factor(self):
        """Test different damping factors"""
        print("\n" + "="*60)
        print("DAMPING FACTOR SENSITIVITY ANALYSIS")
        print("="*60)

        damping_factors = [0.5, 0.7, 0.85, 0.9, 0.95]
        results = []

        for d in damping_factors:
            pr = nx.pagerank(self.G, alpha=d)
            scores = list(pr.values())

            results.append({
                'damping': d,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'gini': self._gini_coefficient(scores)
            })

        results_df = pd.DataFrame(results)
        print("\nPageRank Statistics for Different Damping Factors:")
        print(results_df.to_string(index=False))

        return results_df

    def _gini_coefficient(self, scores):
        """Calculate Gini coefficient (inequality measure)"""
        scores = sorted(scores)
        n = len(scores)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * scores)) / (n * np.sum(scores)) - (n + 1) / n

    def visualize_pagerank(self):
        """Visualize PageRank results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Web graph with PageRank sizes
        ax = axes[0, 0]
        pos = nx.spring_layout(self.G, k=1, iterations=50, seed=self.seed)

        # Node sizes based on PageRank
        pagerank = nx.pagerank(self.G)
        node_sizes = [pagerank[node] * 10000 for node in self.G.nodes()]

        # Node colors based on topic
        topic_colors = {'tech': 0, 'science': 1, 'sports': 2, 'news': 3, 'entertainment': 4}
        node_colors = [topic_colors[self.G.nodes[node]['topic']] for node in self.G.nodes()]

        nx.draw_networkx(self.G, pos, node_size=node_sizes, node_color=node_colors,
                        cmap='tab10', with_labels=False, ax=ax, arrows=True,
                        edge_color='gray', alpha=0.6, arrowsize=10)
        ax.set_title('Web Graph (Node Size = PageRank)', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. PageRank distribution
        ax = axes[0, 1]
        ax.hist(self.page_data['pagerank_nx'], bins=30, edgecolor='black',
               alpha=0.7, color='steelblue')
        ax.set_xlabel('PageRank Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('PageRank Distribution', fontsize=14, fontweight='bold')
        ax.axvline(self.page_data['pagerank_nx'].mean(), color='red',
                  linestyle='--', label='Mean', linewidth=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. PageRank vs Incoming Links
        ax = axes[1, 0]
        topics = self.page_data['topic'].unique()
        for topic in topics:
            data = self.page_data[self.page_data['topic'] == topic]
            ax.scatter(data['incoming_links'], data['pagerank_nx'],
                      label=topic, alpha=0.6, s=100)

        ax.set_xlabel('Incoming Links', fontsize=11)
        ax.set_ylabel('PageRank Score', fontsize=11)
        ax.set_title('PageRank vs Incoming Links', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Top pages comparison
        ax = axes[1, 1]
        top_pages = self.page_data.nlargest(10, 'pagerank_nx')
        x_pos = np.arange(len(top_pages))
        width = 0.35

        # Normalize for comparison
        pr_norm = top_pages['pagerank_nx'] / top_pages['pagerank_nx'].max()
        auth_norm = top_pages['authority_score'] / top_pages['authority_score'].max()

        ax.barh(x_pos - width/2, pr_norm, width, label='PageRank', alpha=0.8, color='steelblue')
        ax.barh(x_pos + width/2, auth_norm, width, label='Authority (HITS)', alpha=0.8, color='coral')

        ax.set_yticks(x_pos)
        ax.set_yticklabels([name.replace('.html', '') for name in top_pages['name']], fontsize=9)
        ax.set_xlabel('Normalized Score', fontsize=11)
        ax.set_title('Top 10 Pages - PageRank vs Authority', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('pagerank_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'pagerank_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("PAGERANK IMPLEMENTATION AND ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = PageRankAnalyzer(n_pages=50, seed=42)

    # Generate web graph
    G, page_data = analyzer.generate_web_graph()

    # Analyze PageRank
    custom_pr, nx_pr = analyzer.analyze_pagerank(damping=0.85)

    # Compare with other metrics
    analyzer.compare_metrics()

    # Test damping factors
    damping_results = analyzer.test_damping_factor()

    # Visualize
    analyzer.visualize_pagerank()

    # Save results
    page_data_export = analyzer.page_data.copy()
    page_data_export.to_csv('pagerank_results.csv', index=False)
    print("\nResults saved to 'pagerank_results.csv'")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total pages analyzed: {len(page_data_export)}")
    print(f"Total links: {G.number_of_edges()}")
    print(f"Average PageRank: {page_data_export['pagerank_nx'].mean():.6f}")
    print(f"Top page: {page_data_export.nlargest(1, 'pagerank_nx')['name'].values[0]}")

if __name__ == "__main__":
    main()
