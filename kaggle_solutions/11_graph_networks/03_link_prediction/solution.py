"""
Link Prediction - Kaggle Solution
==================================
Predicts future connections in networks using graph-based features.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class LinkPredictor:
    """Link prediction using graph features and machine learning"""

    def __init__(self, n_nodes=100, seed=42):
        """
        Initialize link predictor

        Args:
            n_nodes: Number of nodes in network
            seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        self.G = None
        self.G_train = None

    def generate_evolving_network(self):
        """Generate network that evolves over time"""
        print("Generating evolving network...")

        # Start with initial network
        self.G = nx.barabasi_albert_graph(self.n_nodes, 3, seed=self.seed)

        print(f"Generated network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G

    def create_train_test_split(self, test_ratio=0.2):
        """
        Split edges into training and test sets

        Args:
            test_ratio: Proportion of edges to hold out for testing
        """
        print(f"\nCreating train/test split (test={test_ratio*100}%)...")

        # Get all edges
        all_edges = list(self.G.edges())
        n_test = int(len(all_edges) * test_ratio)

        # Randomly select test edges
        np.random.shuffle(all_edges)
        test_edges = all_edges[:n_test]
        train_edges = all_edges[n_test:]

        # Create training graph (without test edges)
        self.G_train = nx.Graph()
        self.G_train.add_nodes_from(self.G.nodes())
        self.G_train.add_edges_from(train_edges)

        # Generate negative samples (non-edges)
        all_possible_edges = set()
        for u in range(self.n_nodes):
            for v in range(u + 1, self.n_nodes):
                all_possible_edges.add((u, v))

        existing_edges = set(self.G.edges())
        non_edges = list(all_possible_edges - existing_edges)

        # Sample equal number of negative examples
        test_non_edges = np.random.choice(len(non_edges), n_test, replace=False)
        test_non_edges = [non_edges[i] for i in test_non_edges]

        print(f"Training edges: {len(train_edges)}")
        print(f"Test edges (positive): {len(test_edges)}")
        print(f"Test edges (negative): {len(test_non_edges)}")

        return train_edges, test_edges, test_non_edges

    def compute_link_features(self, edge_list, graph):
        """
        Compute various graph-based features for edges

        Args:
            edge_list: List of edges to compute features for
            graph: Graph to use for feature computation

        Returns:
            DataFrame with features
        """
        features = []

        for u, v in edge_list:
            feat = {}

            # Common neighbors
            common_neighbors = list(nx.common_neighbors(graph, u, v))
            feat['common_neighbors'] = len(common_neighbors)

            # Jaccard coefficient
            jaccard = list(nx.jaccard_coefficient(graph, [(u, v)]))
            feat['jaccard_coef'] = jaccard[0][2] if jaccard else 0

            # Adamic-Adar index
            adamic_adar = list(nx.adamic_adar_index(graph, [(u, v)]))
            feat['adamic_adar'] = adamic_adar[0][2] if adamic_adar else 0

            # Preferential attachment
            pref_attach = list(nx.preferential_attachment(graph, [(u, v)]))
            feat['pref_attachment'] = pref_attach[0][2] if pref_attach else 0

            # Resource allocation
            resource_alloc = list(nx.resource_allocation_index(graph, [(u, v)]))
            feat['resource_alloc'] = resource_alloc[0][2] if resource_alloc else 0

            # Node degrees
            feat['degree_u'] = graph.degree(u)
            feat['degree_v'] = graph.degree(v)
            feat['degree_product'] = feat['degree_u'] * feat['degree_v']
            feat['degree_sum'] = feat['degree_u'] + feat['degree_v']

            # Clustering coefficients
            try:
                feat['clustering_u'] = nx.clustering(graph, u)
                feat['clustering_v'] = nx.clustering(graph, v)
            except:
                feat['clustering_u'] = 0
                feat['clustering_v'] = 0

            # Shortest path (if exists)
            try:
                feat['shortest_path'] = nx.shortest_path_length(graph, u, v)
            except nx.NetworkXNoPath:
                feat['shortest_path'] = -1  # No path exists

            features.append(feat)

        return pd.DataFrame(features)

    def build_prediction_model(self, train_edges, test_edges, test_non_edges):
        """
        Build and evaluate link prediction model

        Args:
            train_edges: Edges in training graph
            test_edges: Positive test examples
            test_non_edges: Negative test examples
        """
        print("\n" + "="*60)
        print("BUILDING LINK PREDICTION MODEL")
        print("="*60)

        # Compute features for training data
        print("\nComputing features for training data...")

        # Positive examples from training
        train_pos_edges = list(train_edges)[:500]  # Sample for speed
        train_features_pos = self.compute_link_features(train_pos_edges, self.G_train)
        train_features_pos['label'] = 1

        # Negative examples from training
        all_possible_edges = set()
        for u in range(self.n_nodes):
            for v in range(u + 1, self.n_nodes):
                all_possible_edges.add((u, v))

        existing_edges = set(self.G.edges())
        train_non_edges = list(all_possible_edges - existing_edges)
        train_neg_edges = [train_non_edges[i] for i in
                          np.random.choice(len(train_non_edges), len(train_pos_edges), replace=False)]

        train_features_neg = self.compute_link_features(train_neg_edges, self.G_train)
        train_features_neg['label'] = 0

        # Combine training data
        train_features = pd.concat([train_features_pos, train_features_neg], ignore_index=True)

        # Compute features for test data
        print("Computing features for test data...")
        test_features_pos = self.compute_link_features(test_edges, self.G_train)
        test_features_pos['label'] = 1

        test_features_neg = self.compute_link_features(test_non_edges, self.G_train)
        test_features_neg['label'] = 0

        test_features = pd.concat([test_features_pos, test_features_neg], ignore_index=True)

        # Prepare data for modeling
        feature_cols = [col for col in train_features.columns if col != 'label']

        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features['label']

        X_test = test_features[feature_cols].fillna(0)
        y_test = test_features['label']

        # Train Random Forest model
        print("\nTraining Random Forest classifier...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=self.seed, n_jobs=-1)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Evaluation
        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = (y_pred == y_test).mean()

        print(f"\nModel Performance:")
        print(f"  AUC-ROC: {auc_score:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 Most Important Features:")
        print(feature_importance.head().to_string(index=False))

        return model, feature_importance, X_test, y_test, y_pred_proba

    def analyze_similarity_metrics(self, test_edges, test_non_edges):
        """Analyze different similarity metrics"""
        print("\n" + "="*60)
        print("SIMILARITY METRICS COMPARISON")
        print("="*60)

        # Compute metrics for positive examples
        metrics_pos = {
            'common_neighbors': [],
            'jaccard': [],
            'adamic_adar': [],
            'preferential_attachment': []
        }

        for u, v in test_edges[:100]:  # Sample for speed
            common_neighbors = len(list(nx.common_neighbors(self.G_train, u, v)))
            metrics_pos['common_neighbors'].append(common_neighbors)

            jaccard = list(nx.jaccard_coefficient(self.G_train, [(u, v)]))
            metrics_pos['jaccard'].append(jaccard[0][2] if jaccard else 0)

            adamic_adar = list(nx.adamic_adar_index(self.G_train, [(u, v)]))
            metrics_pos['adamic_adar'].append(adamic_adar[0][2] if adamic_adar else 0)

            pref_attach = list(nx.preferential_attachment(self.G_train, [(u, v)]))
            metrics_pos['preferential_attachment'].append(pref_attach[0][2] if pref_attach else 0)

        # Compute metrics for negative examples
        metrics_neg = {
            'common_neighbors': [],
            'jaccard': [],
            'adamic_adar': [],
            'preferential_attachment': []
        }

        for u, v in test_non_edges[:100]:
            common_neighbors = len(list(nx.common_neighbors(self.G_train, u, v)))
            metrics_neg['common_neighbors'].append(common_neighbors)

            jaccard = list(nx.jaccard_coefficient(self.G_train, [(u, v)]))
            metrics_neg['jaccard'].append(jaccard[0][2] if jaccard else 0)

            adamic_adar = list(nx.adamic_adar_index(self.G_train, [(u, v)]))
            metrics_neg['adamic_adar'].append(adamic_adar[0][2] if adamic_adar else 0)

            pref_attach = list(nx.preferential_attachment(self.G_train, [(u, v)]))
            metrics_neg['preferential_attachment'].append(pref_attach[0][2] if pref_attach else 0)

        # Print comparison
        print("\nAverage Metric Values:")
        print(f"{'Metric':<25} {'Positive (Links)':<20} {'Negative (Non-links)':<20}")
        print("-" * 65)

        for metric in metrics_pos.keys():
            avg_pos = np.mean(metrics_pos[metric])
            avg_neg = np.mean(metrics_neg[metric])
            print(f"{metric:<25} {avg_pos:<20.4f} {avg_neg:<20.4f}")

        return metrics_pos, metrics_neg

    def visualize_results(self, feature_importance, y_test, y_pred_proba, metrics_pos, metrics_neg):
        """Visualize link prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Feature importance
        ax = axes[0, 0]
        top_features = feature_importance.head(10)
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 2. ROC Curve
        ax = axes[0, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})', color='steelblue')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        ax = axes[1, 0]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

        ax.plot(recall, precision, linewidth=2, color='coral')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. Similarity metrics comparison
        ax = axes[1, 1]
        metrics_to_plot = ['common_neighbors', 'jaccard', 'adamic_adar']
        x = np.arange(len(metrics_to_plot))
        width = 0.35

        pos_means = [np.mean(metrics_pos[m]) for m in metrics_to_plot]
        neg_means = [np.mean(metrics_neg[m]) for m in metrics_to_plot]

        ax.bar(x - width/2, pos_means, width, label='Positive (Links)', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, neg_means, width, label='Negative (Non-links)', alpha=0.8, color='coral')

        ax.set_xlabel('Similarity Metric', fontsize=11)
        ax.set_ylabel('Average Value', fontsize=11)
        ax.set_title('Similarity Metrics: Links vs Non-Links', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Common\nNeighbors', 'Jaccard', 'Adamic-Adar'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('link_prediction_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'link_prediction_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("LINK PREDICTION ANALYSIS")
    print("="*60)

    # Initialize predictor
    predictor = LinkPredictor(n_nodes=100, seed=42)

    # Generate network
    G = predictor.generate_evolving_network()

    # Create train/test split
    train_edges, test_edges, test_non_edges = predictor.create_train_test_split(test_ratio=0.2)

    # Build prediction model
    model, feature_importance, X_test, y_test, y_pred_proba = predictor.build_prediction_model(
        train_edges, test_edges, test_non_edges
    )

    # Analyze similarity metrics
    metrics_pos, metrics_neg = predictor.analyze_similarity_metrics(test_edges, test_non_edges)

    # Visualize results
    predictor.visualize_results(feature_importance, y_test, y_pred_proba, metrics_pos, metrics_neg)

    # Save feature importance
    feature_importance.to_csv('link_prediction_features.csv', index=False)
    print("\nFeature importance saved to 'link_prediction_features.csv'")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Final AUC-ROC Score: {auc_score:.4f}")
    print(f"Number of features used: {len(feature_importance)}")
    print(f"Most important feature: {feature_importance.iloc[0]['feature']}")

if __name__ == "__main__":
    main()
