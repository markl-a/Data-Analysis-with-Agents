"""
Random Forest Deep Dive - Comprehensive Analysis
Exploring Random Forest classifier with detailed parameter analysis

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RandomForestDeepDive:
    """Comprehensive Random Forest Analysis"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def create_dataset(self):
        """Create synthetic classification dataset"""
        print("Creating synthetic dataset...")

        # Create dataset with varying complexity
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_repeated=2,
            n_classes=3,
            n_clusters_per_class=2,
            weights=[0.4, 0.35, 0.25],
            flip_y=0.05,
            class_sep=0.8,
            random_state=42
        )

        # Create feature names
        feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['target'].value_counts()}")

        return df, feature_names

    def train_single_tree(self, X_train, X_test, y_train, y_test):
        """Train a single decision tree for comparison"""
        print("\n" + "="*60)
        print("Training Single Decision Tree (for comparison)")
        print("="*60)

        tree = DecisionTreeClassifier(random_state=42, max_depth=10)
        tree.fit(X_train, y_train)

        y_pred = tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['single_tree'] = tree
        self.results['single_tree'] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }

        print(f"Single Tree Accuracy: {accuracy:.4f}")
        return tree, accuracy

    def analyze_n_estimators(self, X_train, X_test, y_train, y_test):
        """Analyze the effect of number of trees"""
        print("\n" + "="*60)
        print("Analyzing Number of Trees (n_estimators)")
        print("="*60)

        n_estimators_range = [1, 5, 10, 25, 50, 100, 200, 300, 500]
        train_scores = []
        test_scores = []

        for n_est in n_estimators_range:
            print(f"Training with {n_est} trees...")
            rf = RandomForestClassifier(
                n_estimators=n_est,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            train_score = rf.score(X_train, y_train)
            test_score = rf.score(X_test, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

            print(f"  Train: {train_score:.4f}, Test: {test_score:.4f}")

        self.results['n_estimators'] = {
            'range': n_estimators_range,
            'train_scores': train_scores,
            'test_scores': test_scores
        }

        return n_estimators_range, train_scores, test_scores

    def analyze_max_depth(self, X_train, X_test, y_train, y_test):
        """Analyze the effect of maximum tree depth"""
        print("\n" + "="*60)
        print("Analyzing Maximum Tree Depth")
        print("="*60)

        max_depth_range = [3, 5, 10, 15, 20, 30, None]
        train_scores = []
        test_scores = []

        for depth in max_depth_range:
            depth_str = str(depth) if depth is not None else "unlimited"
            print(f"Training with max_depth={depth_str}...")

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=depth,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            train_score = rf.score(X_train, y_train)
            test_score = rf.score(X_test, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

            print(f"  Train: {train_score:.4f}, Test: {test_score:.4f}")

        self.results['max_depth'] = {
            'range': max_depth_range,
            'train_scores': train_scores,
            'test_scores': test_scores
        }

        return max_depth_range, train_scores, test_scores

    def analyze_min_samples(self, X_train, X_test, y_train, y_test):
        """Analyze min_samples_split and min_samples_leaf"""
        print("\n" + "="*60)
        print("Analyzing Minimum Sample Parameters")
        print("="*60)

        min_samples_range = [2, 5, 10, 20, 50, 100]
        split_scores = []
        leaf_scores = []

        # Analyze min_samples_split
        print("\nTesting min_samples_split:")
        for min_samples in min_samples_range:
            rf = RandomForestClassifier(
                n_estimators=100,
                min_samples_split=min_samples,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            score = rf.score(X_test, y_test)
            split_scores.append(score)
            print(f"  min_samples_split={min_samples}: {score:.4f}")

        # Analyze min_samples_leaf
        print("\nTesting min_samples_leaf:")
        for min_samples in min_samples_range:
            rf = RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=min_samples,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            score = rf.score(X_test, y_test)
            leaf_scores.append(score)
            print(f"  min_samples_leaf={min_samples}: {score:.4f}")

        self.results['min_samples'] = {
            'range': min_samples_range,
            'split_scores': split_scores,
            'leaf_scores': leaf_scores
        }

        return min_samples_range, split_scores, leaf_scores

    def train_optimal_model(self, X_train, X_test, y_train, y_test, feature_names):
        """Train the optimal Random Forest model"""
        print("\n" + "="*60)
        print("Training Optimal Random Forest Model")
        print("="*60)

        rf_optimal = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        rf_optimal.fit(X_train, y_train)

        # Predictions
        y_pred = rf_optimal.predict(X_test)
        y_pred_proba = rf_optimal.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nOptimal Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"OOB Score: {rf_optimal.oob_score_:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_optimal.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        self.models['optimal'] = rf_optimal
        self.results['optimal'] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return rf_optimal

    def visualize_results(self, X_test, y_test):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # 1. Number of Estimators Effect
        ax1 = plt.subplot(3, 3, 1)
        n_est = self.results['n_estimators']
        ax1.plot(n_est['range'], n_est['train_scores'], 'o-', label='Train', linewidth=2)
        ax1.plot(n_est['range'], n_est['test_scores'], 's-', label='Test', linewidth=2)
        ax1.set_xlabel('Number of Trees', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Effect of Number of Trees', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Max Depth Effect
        ax2 = plt.subplot(3, 3, 2)
        depth = self.results['max_depth']
        depth_labels = [str(d) if d is not None else 'None' for d in depth['range']]
        x_pos = np.arange(len(depth_labels))
        ax2.plot(x_pos, depth['train_scores'], 'o-', label='Train', linewidth=2)
        ax2.plot(x_pos, depth['test_scores'], 's-', label='Test', linewidth=2)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(depth_labels, rotation=45)
        ax2.set_xlabel('Max Depth', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Effect of Maximum Depth', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Min Samples Effect
        ax3 = plt.subplot(3, 3, 3)
        min_samp = self.results['min_samples']
        ax3.plot(min_samp['range'], min_samp['split_scores'], 'o-', label='min_samples_split', linewidth=2)
        ax3.plot(min_samp['range'], min_samp['leaf_scores'], 's-', label='min_samples_leaf', linewidth=2)
        ax3.set_xlabel('Min Samples Value', fontsize=10)
        ax3.set_ylabel('Test Accuracy', fontsize=10)
        ax3.set_title('Effect of Min Samples Parameters', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Feature Importance
        ax4 = plt.subplot(3, 3, 4)
        feat_imp = self.results['optimal']['feature_importance'].head(10)
        ax4.barh(range(len(feat_imp)), feat_imp['importance'])
        ax4.set_yticks(range(len(feat_imp)))
        ax4.set_yticklabels(feat_imp['feature'])
        ax4.set_xlabel('Importance', fontsize=10)
        ax4.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()

        # 5. Confusion Matrix
        ax5 = plt.subplot(3, 3, 5)
        cm = self.results['optimal']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=True)
        ax5.set_xlabel('Predicted Label', fontsize=10)
        ax5.set_ylabel('True Label', fontsize=10)
        ax5.set_title('Confusion Matrix - Optimal Model', fontsize=12, fontweight='bold')

        # 6. Model Comparison
        ax6 = plt.subplot(3, 3, 6)
        models = ['Single Tree', 'RF (n=50)', 'RF (n=100)', 'RF (n=200)']
        scores = [
            self.results['single_tree']['accuracy'],
            self.results['n_estimators']['test_scores'][4],  # 50 trees
            self.results['n_estimators']['test_scores'][5],  # 100 trees
            self.results['optimal']['accuracy']
        ]
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
        bars = ax6.bar(range(len(models)), scores, color=colors)
        ax6.set_xticks(range(len(models)))
        ax6.set_xticklabels(models, rotation=45, ha='right')
        ax6.set_ylabel('Accuracy', fontsize=10)
        ax6.set_title('Model Comparison', fontsize=12, fontweight='bold')
        ax6.set_ylim([min(scores) - 0.05, 1.0])
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax6.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom')
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. ROC Curve (One-vs-Rest)
        ax7 = plt.subplot(3, 3, 7)
        y_proba = self.results['optimal']['probabilities']
        n_classes = len(np.unique(y_test))

        for i in range(n_classes):
            y_test_binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax7.plot(fpr, tpr, linewidth=2, label=f'Class {i} (AUC = {roc_auc:.3f})')

        ax7.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax7.set_xlabel('False Positive Rate', fontsize=10)
        ax7.set_ylabel('True Positive Rate', fontsize=10)
        ax7.set_title('ROC Curves (One-vs-Rest)', fontsize=12, fontweight='bold')
        ax7.legend(loc='lower right')
        ax7.grid(True, alpha=0.3)

        # 8. Bootstrap Sampling Analysis
        ax8 = plt.subplot(3, 3, 8)
        rf = self.models['optimal']
        tree_accuracies = []
        for tree in rf.estimators_[:50]:  # Sample first 50 trees
            pred = tree.predict(X_test)
            acc = accuracy_score(y_test, pred)
            tree_accuracies.append(acc)

        ax8.hist(tree_accuracies, bins=20, edgecolor='black', alpha=0.7)
        ax8.axvline(np.mean(tree_accuracies), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(tree_accuracies):.4f}')
        ax8.set_xlabel('Individual Tree Accuracy', fontsize=10)
        ax8.set_ylabel('Frequency', fontsize=10)
        ax8.set_title('Distribution of Individual Tree Accuracies', fontsize=12, fontweight='bold')
        ax8.legend()

        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = f"""
        Random Forest Deep Dive - Summary
        {'='*40}

        Dataset:
        • Total Samples: 2000
        • Features: 20
        • Classes: 3

        Optimal Model Configuration:
        • n_estimators: 200
        • max_depth: 15
        • min_samples_split: 5
        • min_samples_leaf: 2

        Performance:
        • Test Accuracy: {self.results['optimal']['accuracy']:.4f}
        • OOB Score: {self.models['optimal'].oob_score_:.4f}

        Comparison:
        • Single Tree: {self.results['single_tree']['accuracy']:.4f}
        • Random Forest: {self.results['optimal']['accuracy']:.4f}
        • Improvement: {(self.results['optimal']['accuracy'] - self.results['single_tree']['accuracy'])*100:.2f}%
        """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/random_forest_deep_dive.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/random_forest_deep_dive.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Random Forest Deep Dive Analysis")
    print("="*60)

    # Initialize
    rf_analysis = RandomForestDeepDive()

    # Create dataset
    df, feature_names = rf_analysis.create_dataset()

    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train single tree
    rf_analysis.train_single_tree(X_train, X_test, y_train, y_test)

    # Analyze parameters
    rf_analysis.analyze_n_estimators(X_train, X_test, y_train, y_test)
    rf_analysis.analyze_max_depth(X_train, X_test, y_train, y_test)
    rf_analysis.analyze_min_samples(X_train, X_test, y_train, y_test)

    # Train optimal model
    rf_analysis.train_optimal_model(X_train, X_test, y_train, y_test, feature_names)

    # Visualize
    rf_analysis.visualize_results(X_test, y_test)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
