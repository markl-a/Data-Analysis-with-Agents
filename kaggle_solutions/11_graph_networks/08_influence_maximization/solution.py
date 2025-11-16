"""
Influence Maximization - Kaggle Solution
========================================
Finds optimal seed nodes for maximum influence spread in networks.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class InfluenceMaximizer:
    """Influence maximization using various algorithms"""

    def __init__(self, n_nodes=100, seed=42):
        """
        Initialize influence maximizer

        Args:
            n_nodes: Number of nodes
            seed: Random seed
        """
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        self.G = None

    def generate_social_network(self):
        """Generate social network for influence propagation"""
        print("Generating social network...")

        # Create scale-free network (realistic for social networks)
        self.G = nx.barabasi_albert_graph(self.n_nodes, 3, seed=self.seed)

        # Add edge weights (influence probabilities)
        for u, v in self.G.edges():
            # Random influence probability
            self.G[u][v]['prob'] = np.random.uniform(0.1, 0.5)

        print(f"Created network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G

    def independent_cascade(self, seeds, prob='prob', n_simulations=100):
        """
        Independent Cascade model for influence propagation

        Args:
            seeds: Initial seed nodes
            prob: Edge attribute for influence probability
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Average number of influenced nodes
        """
        total_influenced = 0

        for _ in range(n_simulations):
            # Initialize
            influenced = set(seeds)
            active = list(seeds)

            while active:
                new_active = []

                for node in active:
                    # Try to influence neighbors
                    for neighbor in self.G.neighbors(node):
                        if neighbor not in influenced:
                            # Influence with probability
                            influence_prob = self.G[node][neighbor].get(prob, 0.1)
                            if np.random.rand() < influence_prob:
                                influenced.add(neighbor)
                                new_active.append(neighbor)

                active = new_active

            total_influenced += len(influenced)

        return total_influenced / n_simulations

    def greedy_influence_maximization(self, k=5, n_simulations=50):
        """
        Greedy algorithm for influence maximization

        Args:
            k: Number of seed nodes to select
            n_simulations: Simulations per evaluation

        Returns:
            List of selected seed nodes
        """
        print(f"\n" + "="*60)
        print(f"GREEDY INFLUENCE MAXIMIZATION (k={k})")
        print("="*60)

        seeds = []
        spreads = []

        for i in range(k):
            best_node = None
            best_spread = 0

            # Try each node not yet selected
            candidates = set(range(self.n_nodes)) - set(seeds)

            print(f"\nSelecting seed {i+1}/{k}...")

            for node in candidates:
                # Compute spread with this node added
                test_seeds = seeds + [node]
                spread = self.independent_cascade(test_seeds, n_simulations=n_simulations)

                if spread > best_spread:
                    best_spread = spread
                    best_node = node

            seeds.append(best_node)
            spreads.append(best_spread)

            print(f"  Selected node {best_node}, expected influence: {best_spread:.2f}")

        return seeds, spreads

    def degree_based_selection(self, k=5):
        """Select top-k nodes by degree centrality"""
        print(f"\n" + "="*60)
        print(f"DEGREE-BASED SELECTION (k={k})")
        print("="*60)

        degree_cent = nx.degree_centrality(self.G)
        seeds = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:k]
        seed_nodes = [node for node, _ in seeds]

        spread = self.independent_cascade(seed_nodes, n_simulations=100)

        print(f"Selected nodes: {seed_nodes}")
        print(f"Expected influence: {spread:.2f}")

        return seed_nodes, spread

    def pagerank_based_selection(self, k=5):
        """Select top-k nodes by PageRank"""
        print(f"\n" + "="*60)
        print(f"PAGERANK-BASED SELECTION (k={k})")
        print("="*60)

        pagerank = nx.pagerank(self.G)
        seeds = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:k]
        seed_nodes = [node for node, _ in seeds]

        spread = self.independent_cascade(seed_nodes, n_simulations=100)

        print(f"Selected nodes: {seed_nodes}")
        print(f"Expected influence: {spread:.2f}")

        return seed_nodes, spread

    def betweenness_based_selection(self, k=5):
        """Select top-k nodes by betweenness centrality"""
        print(f"\n" + "="*60)
        print(f"BETWEENNESS-BASED SELECTION (k={k})")
        print("="*60)

        betweenness = nx.betweenness_centrality(self.G)
        seeds = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:k]
        seed_nodes = [node for node, _ in seeds]

        spread = self.independent_cascade(seed_nodes, n_simulations=100)

        print(f"Selected nodes: {seed_nodes}")
        print(f"Expected influence: {spread:.2f}")

        return seed_nodes, spread

    def random_selection(self, k=5):
        """Random seed selection (baseline)"""
        print(f"\n" + "="*60)
        print(f"RANDOM SELECTION (k={k})")
        print("="*60)

        seed_nodes = list(np.random.choice(self.n_nodes, k, replace=False))
        spread = self.independent_cascade(seed_nodes, n_simulations=100)

        print(f"Selected nodes: {seed_nodes}")
        print(f"Expected influence: {spread:.2f}")

        return seed_nodes, spread

    def compare_algorithms(self, k=5):
        """Compare different seed selection algorithms"""
        print(f"\n" + "="*60)
        print("ALGORITHM COMPARISON")
        print("="*60)

        results = []

        # Greedy (best but slow)
        greedy_seeds, greedy_spreads = self.greedy_influence_maximization(k, n_simulations=30)
        greedy_final = greedy_spreads[-1]
        results.append({'Algorithm': 'Greedy', 'Influence': greedy_final, 'Seeds': greedy_seeds})

        # Degree-based (fast heuristic)
        degree_seeds, degree_spread = self.degree_based_selection(k)
        results.append({'Algorithm': 'Degree', 'Influence': degree_spread, 'Seeds': degree_seeds})

        # PageRank-based
        pr_seeds, pr_spread = self.pagerank_based_selection(k)
        results.append({'Algorithm': 'PageRank', 'Influence': pr_spread, 'Seeds': pr_seeds})

        # Betweenness-based
        btw_seeds, btw_spread = self.betweenness_based_selection(k)
        results.append({'Algorithm': 'Betweenness', 'Influence': btw_spread, 'Seeds': btw_seeds})

        # Random (baseline)
        rand_seeds, rand_spread = self.random_selection(k)
        results.append({'Algorithm': 'Random', 'Influence': rand_spread, 'Seeds': rand_seeds})

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Influence', ascending=False)

        print("\nComparison Results:")
        print(results_df[['Algorithm', 'Influence']].to_string(index=False))

        return results_df

    def analyze_spread_over_time(self, seeds, max_steps=10):
        """Analyze how influence spreads over time"""
        print(f"\n" + "="*60)
        print("SPREAD OVER TIME ANALYSIS")
        print("="*60)

        # Run multiple simulations and track spread at each step
        n_simulations = 50
        step_spreads = defaultdict(list)

        for sim in range(n_simulations):
            influenced = set(seeds)
            active = list(seeds)
            step = 0

            step_spreads[step].append(len(influenced))

            while active and step < max_steps:
                new_active = []
                step += 1

                for node in active:
                    for neighbor in self.G.neighbors(node):
                        if neighbor not in influenced:
                            influence_prob = self.G[node][neighbor].get('prob', 0.1)
                            if np.random.rand() < influence_prob:
                                influenced.add(neighbor)
                                new_active.append(neighbor)

                active = new_active
                step_spreads[step].append(len(influenced))

        # Calculate average spread at each step
        avg_spread_by_step = {}
        for step, spreads in step_spreads.items():
            avg_spread_by_step[step] = np.mean(spreads)

        print("\nAverage Influence by Time Step:")
        for step in sorted(avg_spread_by_step.keys())[:6]:
            print(f"  Step {step}: {avg_spread_by_step[step]:.2f} nodes influenced")

        return avg_spread_by_step

    def visualize_influence(self, seeds, results_df):
        """Visualize influence maximization results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Network with seed nodes highlighted
        ax = axes[0, 0]

        pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=self.seed)

        # Draw network
        node_colors = ['red' if node in seeds else 'lightblue'
                      for node in self.G.nodes()]
        node_sizes = [500 if node in seeds else 100 for node in self.G.nodes()]

        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                              node_size=node_sizes, ax=ax, alpha=0.7)
        nx.draw_networkx_edges(self.G, pos, edge_color='gray',
                              alpha=0.3, ax=ax)

        # Highlight seed nodes
        nx.draw_networkx_labels(self.G, pos,
                               labels={node: str(node) for node in seeds},
                               font_size=10, font_weight='bold', ax=ax)

        ax.set_title('Network with Seed Nodes (Red)', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. Algorithm comparison
        ax = axes[0, 1]

        algorithms = results_df['Algorithm'].values
        influences = results_df['Influence'].values

        colors = ['green' if alg == 'Greedy' else 'steelblue' for alg in algorithms]
        ax.barh(range(len(algorithms)), influences, color=colors, alpha=0.8)
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels(algorithms)
        ax.set_xlabel('Expected Influence (# of nodes)', fontsize=11)
        ax.set_title('Algorithm Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 3. Influence spread over time
        ax = axes[1, 0]

        spread_over_time = self.analyze_spread_over_time(seeds, max_steps=10)
        steps = sorted(spread_over_time.keys())
        spreads = [spread_over_time[s] for s in steps]

        ax.plot(steps, spreads, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Average Influenced Nodes', fontsize=11)
        ax.set_title('Influence Propagation Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. Degree distribution with seeds highlighted
        ax = axes[1, 1]

        degrees = [d for n, d in self.G.degree()]
        seed_degrees = [self.G.degree(s) for s in seeds]

        ax.hist(degrees, bins=20, alpha=0.7, color='lightblue',
               edgecolor='black', label='All nodes')
        ax.hist(seed_degrees, bins=20, alpha=0.8, color='red',
               edgecolor='black', label='Seed nodes')

        ax.set_xlabel('Degree', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Degree Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('influence_maximization_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'influence_maximization_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("INFLUENCE MAXIMIZATION ANALYSIS")
    print("="*60)

    # Initialize maximizer
    maximizer = InfluenceMaximizer(n_nodes=100, seed=42)

    # Generate network
    G = maximizer.generate_social_network()

    # Compare algorithms
    k = 5  # Number of seed nodes
    results_df = maximizer.compare_algorithms(k=k)

    # Use best seeds for detailed analysis
    best_seeds = results_df.iloc[0]['Seeds']

    # Analyze spread over time
    spread_over_time = maximizer.analyze_spread_over_time(best_seeds)

    # Visualize
    maximizer.visualize_influence(best_seeds, results_df)

    # Save results
    results_df_export = results_df[['Algorithm', 'Influence']].copy()
    results_df_export.to_csv('influence_maximization_results.csv', index=False)
    print("\nResults saved to 'influence_maximization_results.csv'")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    best_algo = results_df.iloc[0]['Algorithm']
    best_influence = results_df.iloc[0]['Influence']
    print(f"Best algorithm: {best_algo}")
    print(f"Expected influence: {best_influence:.2f} nodes")
    print(f"Seed nodes: {best_seeds}")
    print(f"Influence ratio: {best_influence/maximizer.n_nodes:.1%}")

if __name__ == "__main__":
    main()
