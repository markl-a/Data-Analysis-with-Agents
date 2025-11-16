"""
Graph Classification - Kaggle Solution
======================================
Classifies entire graphs into categories using graph-level features.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class GraphClassifier:
    """Graph classification using structural features"""

    def __init__(self, n_graphs_per_class=50, seed=42):
        """
        Initialize graph classifier

        Args:
            n_graphs_per_class: Number of graphs per class
            seed: Random seed
        """
        self.n_graphs_per_class = n_graphs_per_class
        self.seed = seed
        np.random.seed(seed)
        self.graphs = []
        self.labels = []

    def generate_graph_dataset(self):
        """Generate dataset of graphs from different classes"""
        print("Generating graph dataset...")

        # Define graph classes
        graph_types = {
            'random': self._generate_random_graph,
            'small_world': self._generate_small_world_graph,
            'scale_free': self._generate_scale_free_graph,
            'community': self._generate_community_graph
        }

        for graph_type, generator_func in graph_types.items():
            print(f"  Generating {self.n_graphs_per_class} {graph_type} graphs...")

            for i in range(self.n_graphs_per_class):
                G = generator_func()
                self.graphs.append(G)
                self.labels.append(graph_type)

        print(f"\nGenerated {len(self.graphs)} graphs across {len(graph_types)} classes")
        print(f"Class distribution:")
        for graph_type in graph_types.keys():
            count = self.labels.count(graph_type)
            print(f"  {graph_type}: {count}")

        return self.graphs, self.labels

    def _generate_random_graph(self):
        """Generate Erdos-Renyi random graph"""
        n = np.random.randint(20, 50)
        p = np.random.uniform(0.1, 0.3)
        return nx.erdos_renyi_graph(n, p)

    def _generate_small_world_graph(self):
        """Generate small-world graph"""
        n = np.random.randint(20, 50)
        k = np.random.randint(4, 8)
        p = np.random.uniform(0.1, 0.4)
        return nx.watts_strogatz_graph(n, k, p)

    def _generate_scale_free_graph(self):
        """Generate scale-free graph"""
        n = np.random.randint(20, 50)
        m = np.random.randint(2, 5)
        return nx.barabasi_albert_graph(n, m)

    def _generate_community_graph(self):
        """Generate graph with community structure"""
        # Create multiple small complete graphs and connect them sparsely
        n_communities = np.random.randint(3, 6)
        community_sizes = [np.random.randint(5, 12) for _ in range(n_communities)]

        # Build communities
        G = nx.Graph()
        node_id = 0
        community_nodes = []

        for size in community_sizes:
            comm_nodes = list(range(node_id, node_id + size))
            community_nodes.append(comm_nodes)

            # Complete graph within community
            for i in comm_nodes:
                for j in comm_nodes:
                    if i < j:
                        G.add_edge(i, j)

            node_id += size

        # Add sparse inter-community edges
        for i in range(n_communities):
            for j in range(i + 1, n_communities):
                # Connect a few nodes between communities
                n_inter_edges = np.random.randint(1, 3)
                for _ in range(n_inter_edges):
                    u = np.random.choice(community_nodes[i])
                    v = np.random.choice(community_nodes[j])
                    G.add_edge(u, v)

        return G

    def extract_graph_features(self, G):
        """
        Extract structural features from a graph

        Args:
            G: NetworkX graph

        Returns:
            dict: Feature dictionary
        """
        features = {}

        # Basic stats
        n = G.number_of_nodes()
        m = G.number_of_edges()

        features['n_nodes'] = n
        features['n_edges'] = m
        features['density'] = nx.density(G)

        # Degree statistics
        degrees = [d for _, d in G.degree()]
        features['avg_degree'] = np.mean(degrees)
        features['std_degree'] = np.std(degrees)
        features['max_degree'] = np.max(degrees)
        features['min_degree'] = np.min(degrees)

        # Clustering
        features['avg_clustering'] = nx.average_clustering(G)
        features['transitivity'] = nx.transitivity(G)

        # Connectivity
        features['is_connected'] = 1 if nx.is_connected(G) else 0
        features['n_components'] = nx.number_connected_components(G)

        # Path-based features (use largest component if disconnected)
        if nx.is_connected(G):
            features['diameter'] = nx.diameter(G)
            features['avg_shortest_path'] = nx.average_shortest_path_length(G)
            features['radius'] = nx.radius(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc)
            features['diameter'] = nx.diameter(G_largest)
            features['avg_shortest_path'] = nx.average_shortest_path_length(G_largest)
            features['radius'] = nx.radius(G_largest)

        # Assortativity
        try:
            features['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            features['assortativity'] = 0

        # Centrality statistics (use subset for speed)
        sample_nodes = list(G.nodes())[:min(30, len(G.nodes()))]
        betweenness = nx.betweenness_centrality(G.subgraph(sample_nodes))
        features['avg_betweenness'] = np.mean(list(betweenness.values()))

        # Degree distribution properties
        degree_sequence = sorted(degrees, reverse=True)
        features['degree_skewness'] = pd.Series(degrees).skew()
        features['degree_kurtosis'] = pd.Series(degrees).kurtosis()

        # Graph energy (sum of absolute eigenvalues)
        try:
            adj_matrix = nx.adjacency_matrix(G).todense()
            eigenvalues = np.linalg.eigvalsh(adj_matrix)
            features['graph_energy'] = np.sum(np.abs(eigenvalues))
        except:
            features['graph_energy'] = 0

        return features

    def build_feature_dataset(self):
        """Extract features from all graphs"""
        print("\n" + "="*60)
        print("EXTRACTING GRAPH FEATURES")
        print("="*60)

        features_list = []

        for i, G in enumerate(self.graphs):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(self.graphs)} graphs...")

            features = self.extract_graph_features(G)
            features['graph_id'] = i
            features['label'] = self.labels[i]
            features_list.append(features)

        features_df = pd.DataFrame(features_list)

        print(f"\nExtracted {len(features_df.columns)-2} features from {len(features_df)} graphs")

        return features_df

    def train_classifier(self, features_df):
        """Train graph classification model"""
        print("\n" + "="*60)
        print("TRAINING GRAPH CLASSIFIER")
        print("="*60)

        # Prepare data
        feature_cols = [col for col in features_df.columns
                       if col not in ['graph_id', 'label']]

        X = features_df[feature_cols].fillna(0)
        y = features_df['label']

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=self.seed, stratify=y_encoded
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest
        print("\nTraining Random Forest classifier...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                    random_state=self.seed, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = rf.predict(X_test_scaled)
        y_pred_proba = rf.predict_proba(X_test_scaled)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=le.classes_))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

        return rf, le, scaler, feature_importance, X_test_scaled, y_test, y_pred

    def analyze_feature_distributions(self, features_df):
        """Analyze feature distributions by class"""
        print("\n" + "="*60)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("="*60)

        # Select key features
        key_features = ['avg_clustering', 'avg_degree', 'density',
                       'diameter', 'assortativity']

        print("\nAverage Feature Values by Graph Class:")
        feature_summary = features_df.groupby('label')[key_features].mean()
        print(feature_summary.round(4))

        return feature_summary

    def visualize_results(self, features_df, feature_importance, y_test, y_pred, le):
        """Visualize classification results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Feature importance
        ax = axes[0, 0]
        top_features = feature_importance.head(10)
        ax.barh(range(len(top_features)), top_features['importance'],
               color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=9)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 2. Confusion matrix
        ax = axes[0, 1]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        # 3. Feature distributions by class
        ax = axes[1, 0]

        # Plot clustering coefficient by class
        graph_types = features_df['label'].unique()
        data_to_plot = [features_df[features_df['label'] == gt]['avg_clustering'].values
                       for gt in sorted(graph_types)]

        bp = ax.boxplot(data_to_plot, labels=sorted(graph_types), patch_artist=True)
        for patch, color in zip(bp['boxes'], ['steelblue', 'coral', 'mediumseagreen', 'orange']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Average Clustering Coefficient', fontsize=11)
        ax.set_title('Clustering by Graph Type', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Feature scatter plot
        ax = axes[1, 1]

        colors = {'random': 'steelblue', 'small_world': 'coral',
                 'scale_free': 'mediumseagreen', 'community': 'orange'}

        for graph_type in graph_types:
            data = features_df[features_df['label'] == graph_type]
            ax.scatter(data['avg_clustering'], data['avg_degree'],
                      label=graph_type, alpha=0.6, s=50,
                      color=colors.get(graph_type, 'gray'))

        ax.set_xlabel('Average Clustering', fontsize=11)
        ax.set_ylabel('Average Degree', fontsize=11)
        ax.set_title('Graph Types in Feature Space', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('graph_classification_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'graph_classification_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("GRAPH CLASSIFICATION ANALYSIS")
    print("="*60)

    # Initialize classifier
    classifier = GraphClassifier(n_graphs_per_class=50, seed=42)

    # Generate dataset
    graphs, labels = classifier.generate_graph_dataset()

    # Extract features
    features_df = classifier.build_feature_dataset()

    # Analyze feature distributions
    feature_summary = classifier.analyze_feature_distributions(features_df)

    # Train classifier
    rf, le, scaler, feature_importance, X_test, y_test, y_pred = classifier.train_classifier(features_df)

    # Visualize
    classifier.visualize_results(features_df, feature_importance, y_test, y_pred, le)

    # Save results
    features_df.to_csv('graph_features.csv', index=False)
    feature_importance.to_csv('graph_feature_importance.csv', index=False)
    print("\nResults saved to CSV files")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Total graphs: {len(graphs)}")
    print(f"Graph classes: {len(set(labels))}")
    print(f"Classification accuracy: {accuracy:.2%}")
    print(f"Most important feature: {feature_importance.iloc[0]['feature']}")

if __name__ == "__main__":
    main()
