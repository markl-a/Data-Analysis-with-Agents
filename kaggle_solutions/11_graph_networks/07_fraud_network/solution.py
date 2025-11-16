"""
Fraud Detection in Networks - Kaggle Solution
=============================================
Detects fraudulent behavior in transaction networks using graph analysis.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class FraudNetworkDetector:
    """Fraud detection using network analysis"""

    def __init__(self, n_users=200, fraud_ratio=0.1, seed=42):
        """
        Initialize fraud detector

        Args:
            n_users: Number of users
            fraud_ratio: Proportion of fraudsters
            seed: Random seed
        """
        self.n_users = n_users
        self.fraud_ratio = fraud_ratio
        self.seed = seed
        np.random.seed(seed)
        self.G = None
        self.users = None

    def generate_transaction_network(self):
        """Generate transaction network with fraudulent users"""
        print("Generating transaction network with fraud...")

        self.G = nx.DiGraph()

        # Determine fraudsters
        n_fraudsters = int(self.n_users * self.fraud_ratio)
        fraudsters = set(np.random.choice(self.n_users, n_fraudsters, replace=False))

        # Generate user data
        users_data = []
        for i in range(self.n_users):
            is_fraud = i in fraudsters

            if is_fraud:
                # Fraudsters have different patterns
                account_age_days = np.random.randint(1, 100)  # Newer accounts
                n_transactions = np.random.randint(20, 100)  # More transactions
            else:
                # Legitimate users
                account_age_days = np.random.randint(100, 1000)
                n_transactions = np.random.randint(5, 50)

            users_data.append({
                'user_id': i,
                'account_age_days': account_age_days,
                'is_fraud': is_fraud
            })

            self.G.add_node(i, is_fraud=is_fraud, account_age=account_age_days)

        # Create transactions
        transactions = []
        transaction_id = 0

        for user_id in range(self.n_users):
            is_fraud = user_id in fraudsters

            if is_fraud:
                # Fraudsters: Send money to other fraudsters, receive from legitimate users
                # Send to other fraudsters
                n_fraud_sends = np.random.randint(5, 15)
                targets = np.random.choice(list(fraudsters - {user_id}),
                                         min(n_fraud_sends, len(fraudsters) - 1),
                                         replace=False)

                for target in targets:
                    amount = np.random.uniform(100, 5000)
                    self.G.add_edge(user_id, target, amount=amount, transaction_id=transaction_id)
                    transactions.append({
                        'transaction_id': transaction_id,
                        'from_user': user_id,
                        'to_user': target,
                        'amount': amount,
                        'is_fraud': True
                    })
                    transaction_id += 1

                # Receive from legitimate users (phishing/scam)
                n_receives = np.random.randint(10, 30)
                legitimate_users = list(set(range(self.n_users)) - fraudsters)
                sources = np.random.choice(legitimate_users,
                                          min(n_receives, len(legitimate_users)),
                                          replace=True)

                for source in sources:
                    amount = np.random.uniform(50, 2000)
                    if not self.G.has_edge(source, user_id):  # Avoid duplicates
                        self.G.add_edge(source, user_id, amount=amount, transaction_id=transaction_id)
                        transactions.append({
                            'transaction_id': transaction_id,
                            'from_user': source,
                            'to_user': user_id,
                            'amount': amount,
                            'is_fraud': False  # Transaction itself looks normal
                        })
                        transaction_id += 1

            else:
                # Legitimate users: Normal transaction patterns
                n_sends = np.random.randint(2, 10)
                targets = np.random.choice(self.n_users, n_sends, replace=False)

                for target in targets:
                    if target != user_id:
                        amount = np.random.uniform(10, 1000)
                        self.G.add_edge(user_id, target, amount=amount, transaction_id=transaction_id)
                        transactions.append({
                            'transaction_id': transaction_id,
                            'from_user': user_id,
                            'to_user': target,
                            'amount': amount,
                            'is_fraud': False
                        })
                        transaction_id += 1

        self.users = pd.DataFrame(users_data)
        self.transactions = pd.DataFrame(transactions)

        print(f"Created network with {self.G.number_of_nodes()} users")
        print(f"Total transactions: {self.G.number_of_edges()}")
        print(f"Fraudsters: {n_fraudsters} ({self.fraud_ratio*100:.1f}%)")
        print(f"Legitimate users: {self.n_users - n_fraudsters}")

        return self.G, self.users, self.transactions

    def compute_fraud_features(self):
        """Compute graph-based fraud detection features"""
        print("\n" + "="*60)
        print("COMPUTING FRAUD DETECTION FEATURES")
        print("="*60)

        features_list = []

        for user_id in range(self.n_users):
            features = {'user_id': user_id}

            # Degree features
            features['in_degree'] = self.G.in_degree(user_id)
            features['out_degree'] = self.G.out_degree(user_id)
            features['total_degree'] = features['in_degree'] + features['out_degree']

            # Transaction amounts
            outgoing_amounts = [self.G[user_id][neighbor]['amount']
                              for neighbor in self.G.successors(user_id)]
            incoming_amounts = [self.G[predecessor][user_id]['amount']
                              for predecessor in self.G.predecessors(user_id)]

            features['total_sent'] = sum(outgoing_amounts) if outgoing_amounts else 0
            features['total_received'] = sum(incoming_amounts) if incoming_amounts else 0
            features['avg_sent'] = np.mean(outgoing_amounts) if outgoing_amounts else 0
            features['avg_received'] = np.mean(incoming_amounts) if incoming_amounts else 0
            features['max_sent'] = max(outgoing_amounts) if outgoing_amounts else 0
            features['max_received'] = max(incoming_amounts) if incoming_amounts else 0

            # Network position features
            try:
                features['pagerank'] = nx.pagerank(self.G)[user_id]
            except:
                features['pagerank'] = 0

            try:
                features['betweenness'] = nx.betweenness_centrality(self.G)[user_id]
            except:
                features['betweenness'] = 0

            # Clustering coefficient (undirected)
            G_undirected = self.G.to_undirected()
            features['clustering'] = nx.clustering(G_undirected, user_id)

            # Neighbor analysis
            neighbors = set(self.G.successors(user_id)) | set(self.G.predecessors(user_id))
            features['n_neighbors'] = len(neighbors)

            # Account age
            features['account_age_days'] = self.G.nodes[user_id]['account_age']

            # Money flow imbalance
            if features['total_received'] + features['total_sent'] > 0:
                features['money_imbalance'] = abs(features['total_received'] - features['total_sent']) / \
                                             (features['total_received'] + features['total_sent'])
            else:
                features['money_imbalance'] = 0

            features_list.append(features)

        features_df = pd.DataFrame(features_list)

        # Merge with fraud labels
        features_df = features_df.merge(self.users[['user_id', 'is_fraud']], on='user_id')

        print(f"\nComputed {len(features_df.columns)-2} features for {len(features_df)} users")
        print(f"\nFeature correlation with fraud:")

        # Calculate correlations
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f not in ['user_id', 'is_fraud']]

        correlations = []
        for feature in numeric_features:
            corr = features_df[['is_fraud', feature]].corr().iloc[0, 1]
            correlations.append({'feature': feature, 'correlation': abs(corr)})

        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        print(corr_df.head(5).to_string(index=False))

        return features_df

    def detect_fraud_unsupervised(self, features_df):
        """Unsupervised fraud detection using Isolation Forest"""
        print("\n" + "="*60)
        print("UNSUPERVISED FRAUD DETECTION")
        print("="*60)

        # Select features
        feature_cols = [col for col in features_df.columns
                       if col not in ['user_id', 'is_fraud']]

        X = features_df[feature_cols].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=self.fraud_ratio,
                                     random_state=self.seed)
        predictions = iso_forest.fit_predict(X_scaled)

        # Convert to 0/1 (IsolationForest returns -1 for outliers)
        predictions = (predictions == -1).astype(int)

        # Evaluate
        from sklearn.metrics import classification_report, accuracy_score, f1_score

        print("\nIsolation Forest Results:")
        print(f"Accuracy: {accuracy_score(features_df['is_fraud'], predictions):.4f}")
        print(f"F1-Score: {f1_score(features_df['is_fraud'], predictions):.4f}")

        print("\nClassification Report:")
        print(classification_report(features_df['is_fraud'], predictions,
                                   target_names=['Legitimate', 'Fraud']))

        return predictions

    def detect_fraud_supervised(self, features_df):
        """Supervised fraud detection using Random Forest"""
        print("\n" + "="*60)
        print("SUPERVISED FRAUD DETECTION")
        print("="*60)

        # Select features
        feature_cols = [col for col in features_df.columns
                       if col not in ['user_id', 'is_fraud']]

        X = features_df[feature_cols].fillna(0)
        y = features_df['is_fraud'].astype(int)

        # Split train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=y
        )

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    random_state=self.seed)
        rf.fit(X_train, y_train)

        # Predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]

        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        print("\nRandom Forest Results:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Legitimate', 'Fraud']))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 5 Most Important Features:")
        print(feature_importance.head().to_string(index=False))

        return rf, feature_importance, y_test, y_pred, y_pred_proba

    def analyze_fraud_patterns(self):
        """Analyze patterns in fraudulent behavior"""
        print("\n" + "="*60)
        print("FRAUD PATTERN ANALYSIS")
        print("="*60)

        fraud_users = self.users[self.users['is_fraud'] == True]['user_id'].values
        legit_users = self.users[self.users['is_fraud'] == False]['user_id'].values

        # Transaction patterns
        fraud_out_degrees = [self.G.out_degree(u) for u in fraud_users]
        fraud_in_degrees = [self.G.in_degree(u) for u in fraud_users]

        legit_out_degrees = [self.G.out_degree(u) for u in legit_users]
        legit_in_degrees = [self.G.in_degree(u) for u in legit_users]

        print("\nTransaction Patterns:")
        print(f"Fraud - Avg outgoing: {np.mean(fraud_out_degrees):.2f}, Avg incoming: {np.mean(fraud_in_degrees):.2f}")
        print(f"Legit - Avg outgoing: {np.mean(legit_out_degrees):.2f}, Avg incoming: {np.mean(legit_in_degrees):.2f}")

        # Account age
        fraud_age = self.users[self.users['is_fraud'] == True]['account_age_days'].mean()
        legit_age = self.users[self.users['is_fraud'] == False]['account_age_days'].mean()

        print(f"\nAccount Age:")
        print(f"Fraud - Avg: {fraud_age:.1f} days")
        print(f"Legit - Avg: {legit_age:.1f} days")

    def visualize_fraud_network(self, features_df, predictions):
        """Visualize fraud detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Network visualization (subset)
        ax = axes[0, 0]

        # Select subset for visualization
        fraud_users = self.users[self.users['is_fraud'] == True]['user_id'].values[:10]
        sample_legit = self.users[self.users['is_fraud'] == False]['user_id'].values[:20]
        sample_users = list(fraud_users) + list(sample_legit)

        subgraph = self.G.subgraph(sample_users)
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=self.seed)

        # Color by fraud status
        colors = ['red' if self.G.nodes[node]['is_fraud'] else 'lightblue'
                 for node in subgraph.nodes()]

        nx.draw_networkx(subgraph, pos, node_color=colors, node_size=300,
                        with_labels=True, ax=ax, font_size=8,
                        edge_color='gray', alpha=0.6, arrows=True)
        ax.set_title('Transaction Network (Red=Fraud, Blue=Legitimate)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # 2. Confusion matrix
        ax = axes[0, 1]
        cm = confusion_matrix(features_df['is_fraud'], predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title('Confusion Matrix (Unsupervised)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Legitimate', 'Fraud'])
        ax.set_yticklabels(['Legitimate', 'Fraud'])

        # 3. Feature comparison
        ax = axes[1, 0]

        fraud_data = features_df[features_df['is_fraud'] == True]
        legit_data = features_df[features_df['is_fraud'] == False]

        metrics = ['in_degree', 'out_degree', 'total_sent']
        x = np.arange(len(metrics))
        width = 0.35

        fraud_means = [fraud_data[m].mean() for m in metrics]
        legit_means = [legit_data[m].mean() for m in metrics]

        # Normalize for visualization
        fraud_means_norm = [f / max(f, l) if max(f, l) > 0 else 0
                           for f, l in zip(fraud_means, legit_means)]
        legit_means_norm = [l / max(f, l) if max(f, l) > 0 else 0
                           for f, l in zip(fraud_means, legit_means)]

        ax.bar(x - width/2, fraud_means_norm, width, label='Fraud', alpha=0.8, color='red')
        ax.bar(x + width/2, legit_means_norm, width, label='Legitimate', alpha=0.8, color='blue')

        ax.set_ylabel('Normalized Average Value', fontsize=11)
        ax.set_title('Behavior Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Account age distribution
        ax = axes[1, 1]

        ax.hist([fraud_data['account_age_days'], legit_data['account_age_days']],
               bins=20, label=['Fraud', 'Legitimate'], alpha=0.7,
               color=['red', 'blue'], edgecolor='black')
        ax.set_xlabel('Account Age (days)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Account Age Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('fraud_network_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'fraud_network_analysis.png'")
        plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("FRAUD DETECTION IN TRANSACTION NETWORKS")
    print("="*60)

    # Initialize detector
    detector = FraudNetworkDetector(n_users=200, fraud_ratio=0.1, seed=42)

    # Generate network
    G, users, transactions = detector.generate_transaction_network()

    # Compute features
    features_df = detector.compute_fraud_features()

    # Unsupervised detection
    predictions_unsup = detector.detect_fraud_unsupervised(features_df)

    # Supervised detection
    rf_model, feature_importance, y_test, y_pred, y_pred_proba = detector.detect_fraud_supervised(features_df)

    # Analyze fraud patterns
    detector.analyze_fraud_patterns()

    # Visualize
    detector.visualize_fraud_network(features_df, predictions_unsup)

    # Save results
    features_df['predicted_fraud_unsupervised'] = predictions_unsup
    features_df.to_csv('fraud_detection_results.csv', index=False)
    feature_importance.to_csv('fraud_feature_importance.csv', index=False)
    print("\nResults saved to CSV files")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total users: {len(users)}")
    print(f"Fraudsters: {users['is_fraud'].sum()}")
    print(f"Detection accuracy: {(predictions_unsup == features_df['is_fraud'].values).mean():.2%}")

if __name__ == "__main__":
    main()
