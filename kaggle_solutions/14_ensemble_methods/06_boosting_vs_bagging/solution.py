"""
Boosting vs Bagging - Comprehensive Comparison
Deep dive into two fundamental ensemble strategies

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BoostingVsBagging:
    """Compare Boosting and Bagging ensemble methods"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def create_dataset(self):
        """Create synthetic classification dataset"""
        print("Creating synthetic dataset...")

        X, y = make_classification(
            n_samples=3000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_repeated=2,
            n_classes=2,
            n_clusters_per_class=2,
            weights=[0.6, 0.4],
            flip_y=0.05,
            class_sep=0.7,
            random_state=42
        )

        feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['target'].value_counts()}")

        return df, feature_names

    def train_baseline(self, X_train, X_test, y_train, y_test):
        """Train baseline decision tree"""
        print("\n" + "="*60)
        print("Training Baseline Decision Tree")
        print("="*60)

        dt = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Baseline Decision Tree Accuracy: {accuracy:.4f}")

        self.models['baseline'] = dt
        self.results['Baseline Tree'] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }

        return dt

    def train_bagging_models(self, X_train, X_test, y_train, y_test):
        """Train various bagging-based models"""
        print("\n" + "="*60)
        print("Training Bagging Models")
        print("="*60)

        # 1. Bagging Classifier
        print("\nTraining Bagging Classifier...")
        bagging = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        bagging.fit(X_train, y_train)
        y_pred_bag = bagging.predict(X_test)
        acc_bag = accuracy_score(y_test, y_pred_bag)
        print(f"  Accuracy: {acc_bag:.4f}")

        self.models['bagging'] = bagging
        self.results['Bagging'] = {
            'accuracy': acc_bag,
            'predictions': y_pred_bag
        }

        # 2. Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"  Accuracy: {acc_rf:.4f}")

        self.models['random_forest'] = rf
        self.results['Random Forest'] = {
            'accuracy': acc_rf,
            'predictions': y_pred_rf
        }

        # 3. Extra Trees
        print("\nTraining Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        et.fit(X_train, y_train)
        y_pred_et = et.predict(X_test)
        acc_et = accuracy_score(y_test, y_pred_et)
        print(f"  Accuracy: {acc_et:.4f}")

        self.models['extra_trees'] = et
        self.results['Extra Trees'] = {
            'accuracy': acc_et,
            'predictions': y_pred_et
        }

    def train_boosting_models(self, X_train, X_test, y_train, y_test):
        """Train various boosting-based models"""
        print("\n" + "="*60)
        print("Training Boosting Models")
        print("="*60)

        # 1. AdaBoost
        print("\nTraining AdaBoost...")
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            random_state=42
        )
        ada.fit(X_train, y_train)
        y_pred_ada = ada.predict(X_test)
        acc_ada = accuracy_score(y_test, y_pred_ada)
        print(f"  Accuracy: {acc_ada:.4f}")

        self.models['adaboost'] = ada
        self.results['AdaBoost'] = {
            'accuracy': acc_ada,
            'predictions': y_pred_ada
        }

        # 2. Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        acc_gb = accuracy_score(y_test, y_pred_gb)
        print(f"  Accuracy: {acc_gb:.4f}")

        self.models['gradient_boosting'] = gb
        self.results['Gradient Boosting'] = {
            'accuracy': acc_gb,
            'predictions': y_pred_gb
        }

    def analyze_bias_variance(self, X_train, y_train):
        """Analyze bias-variance tradeoff"""
        print("\n" + "="*60)
        print("Analyzing Bias-Variance Tradeoff")
        print("="*60)

        n_estimators_range = [1, 5, 10, 25, 50, 100, 200]

        # Bagging
        print("Analyzing Bagging...")
        bagging_train_scores = []
        bagging_test_scores = []

        for n_est in n_estimators_range:
            model = BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=10),
                n_estimators=n_est,
                random_state=42,
                n_jobs=-1
            )
            # Simple train/test split already done
            model.fit(X_train, y_train)
            bagging_train_scores.append(model.score(X_train, y_train))

        # Boosting
        print("Analyzing AdaBoost...")
        boosting_train_scores = []

        for n_est in n_estimators_range:
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=n_est,
                random_state=42
            )
            model.fit(X_train, y_train)
            boosting_train_scores.append(model.score(X_train, y_train))

        self.results['bias_variance'] = {
            'n_estimators': n_estimators_range,
            'bagging_train': bagging_train_scores,
            'boosting_train': boosting_train_scores
        }

    def analyze_convergence(self, X_train, X_test, y_train, y_test):
        """Analyze how performance improves with number of estimators"""
        print("\n" + "="*60)
        print("Analyzing Convergence")
        print("="*60)

        n_range = range(1, 201, 5)

        # Bagging convergence
        print("Analyzing Bagging convergence...")
        bag_scores = []
        for n in n_range:
            model = BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=10),
                n_estimators=n,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            bag_scores.append(model.score(X_test, y_test))

        # Boosting convergence
        print("Analyzing Boosting convergence...")
        boost_scores = []
        for n in n_range:
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=n,
                random_state=42
            )
            model.fit(X_train, y_train)
            boost_scores.append(model.score(X_test, y_test))

        self.results['convergence'] = {
            'n_estimators': list(n_range),
            'bagging_scores': bag_scores,
            'boosting_scores': boost_scores
        }

    def analyze_learning_curves(self, X, y):
        """Analyze learning curves"""
        print("\n" + "="*60)
        print("Analyzing Learning Curves")
        print("="*60)

        train_sizes = np.linspace(0.1, 1.0, 10)

        # Bagging
        print("Computing Bagging learning curve...")
        bag_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
        bag_train_sizes, bag_train_scores, bag_test_scores = learning_curve(
            bag_model, X, y, train_sizes=train_sizes, cv=5, n_jobs=-1, random_state=42
        )

        # Boosting
        print("Computing Boosting learning curve...")
        boost_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50,
            random_state=42
        )
        boost_train_sizes, boost_train_scores, boost_test_scores = learning_curve(
            boost_model, X, y, train_sizes=train_sizes, cv=5, n_jobs=-1, random_state=42
        )

        self.results['learning_curves'] = {
            'train_sizes': bag_train_sizes,
            'bagging': {
                'train_scores': bag_train_scores,
                'test_scores': bag_test_scores
            },
            'boosting': {
                'train_scores': boost_train_scores,
                'test_scores': boost_test_scores
            }
        }

    def visualize_results(self, y_test):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # 1. Overall Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        all_models = ['Baseline Tree', 'Bagging', 'Random Forest', 'Extra Trees',
                     'AdaBoost', 'Gradient Boosting']
        accuracies = [self.results[m]['accuracy'] for m in all_models]

        colors = ['gray', 'lightblue', 'blue', 'darkblue', 'orange', 'red']
        bars = ax1.barh(range(len(all_models)), accuracies, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(all_models)))
        ax1.set_yticklabels(all_models)
        ax1.set_xlabel('Accuracy', fontsize=10)
        ax1.set_title('Overall Performance Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlim([min(accuracies) - 0.05, 1.0])
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(acc + 0.005, i, f'{acc:.4f}', va='center')
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Bagging Methods Comparison
        ax2 = plt.subplot(3, 3, 2)
        bagging_models = ['Baseline Tree', 'Bagging', 'Random Forest', 'Extra Trees']
        bagging_accs = [self.results[m]['accuracy'] for m in bagging_models]

        colors2 = ['gray', 'lightblue', 'blue', 'darkblue']
        bars = ax2.bar(range(len(bagging_models)), bagging_accs, color=colors2, edgecolor='black')
        ax2.set_xticks(range(len(bagging_models)))
        ax2.set_xticklabels(bagging_models, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Bagging Methods', fontsize=12, fontweight='bold')
        for i, (bar, acc) in enumerate(zip(bars, bagging_accs)):
            ax2.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Boosting Methods Comparison
        ax3 = plt.subplot(3, 3, 3)
        boosting_models = ['Baseline Tree', 'AdaBoost', 'Gradient Boosting']
        boosting_accs = [self.results[m]['accuracy'] for m in boosting_models]

        colors3 = ['gray', 'orange', 'red']
        bars = ax3.bar(range(len(boosting_models)), boosting_accs, color=colors3, edgecolor='black')
        ax3.set_xticks(range(len(boosting_models)))
        ax3.set_xticklabels(boosting_models, rotation=45, ha='right')
        ax3.set_ylabel('Accuracy', fontsize=10)
        ax3.set_title('Boosting Methods', fontsize=12, fontweight='bold')
        for i, (bar, acc) in enumerate(zip(bars, boosting_accs)):
            ax3.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Convergence Analysis
        if 'convergence' in self.results:
            ax4 = plt.subplot(3, 3, 4)
            conv = self.results['convergence']
            ax4.plot(conv['n_estimators'], conv['bagging_scores'],
                    'o-', label='Bagging', linewidth=2, markersize=4)
            ax4.plot(conv['n_estimators'], conv['boosting_scores'],
                    's-', label='Boosting', linewidth=2, markersize=4)
            ax4.set_xlabel('Number of Estimators', fontsize=10)
            ax4.set_ylabel('Test Accuracy', fontsize=10)
            ax4.set_title('Convergence Analysis', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Learning Curves - Bagging
        if 'learning_curves' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            lc = self.results['learning_curves']
            bag_data = lc['bagging']

            train_mean = bag_data['train_scores'].mean(axis=1)
            train_std = bag_data['train_scores'].std(axis=1)
            test_mean = bag_data['test_scores'].mean(axis=1)
            test_std = bag_data['test_scores'].std(axis=1)

            ax5.plot(lc['train_sizes'], train_mean, 'o-', label='Train', linewidth=2)
            ax5.fill_between(lc['train_sizes'], train_mean - train_std,
                            train_mean + train_std, alpha=0.2)
            ax5.plot(lc['train_sizes'], test_mean, 's-', label='CV', linewidth=2)
            ax5.fill_between(lc['train_sizes'], test_mean - test_std,
                            test_mean + test_std, alpha=0.2)
            ax5.set_xlabel('Training Size', fontsize=10)
            ax5.set_ylabel('Accuracy', fontsize=10)
            ax5.set_title('Learning Curve - Bagging', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Learning Curves - Boosting
        if 'learning_curves' in self.results:
            ax6 = plt.subplot(3, 3, 6)
            boost_data = lc['boosting']

            train_mean = boost_data['train_scores'].mean(axis=1)
            train_std = boost_data['train_scores'].std(axis=1)
            test_mean = boost_data['test_scores'].mean(axis=1)
            test_std = boost_data['test_scores'].std(axis=1)

            ax6.plot(lc['train_sizes'], train_mean, 'o-', label='Train', linewidth=2)
            ax6.fill_between(lc['train_sizes'], train_mean - train_std,
                            train_mean + train_std, alpha=0.2)
            ax6.plot(lc['train_sizes'], test_mean, 's-', label='CV', linewidth=2)
            ax6.fill_between(lc['train_sizes'], test_mean - test_std,
                            test_mean + test_std, alpha=0.2)
            ax6.set_xlabel('Training Size', fontsize=10)
            ax6.set_ylabel('Accuracy', fontsize=10)
            ax6.set_title('Learning Curve - Boosting', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. Confusion Matrix Comparison
        ax7 = plt.subplot(3, 3, 7)
        cm_bag = confusion_matrix(y_test, self.results['Bagging']['predictions'])
        cm_boost = confusion_matrix(y_test, self.results['AdaBoost']['predictions'])

        # Combined heatmap
        combined_cm = np.hstack([cm_bag, cm_boost])
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', ax=ax7,
                   xticklabels=['Bag-0', 'Bag-1', 'Boost-0', 'Boost-1'],
                   yticklabels=['True-0', 'True-1'])
        ax7.set_title('Confusion Matrices (Bagging vs Boosting)', fontsize=12, fontweight='bold')
        ax7.axvline(x=2, color='red', linewidth=2)

        # 8. Methodology Comparison
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        comparison_text = """
        Boosting vs Bagging

        BAGGING (Bootstrap Aggregating):
        • Parallel training
        • Random sampling with replacement
        • Reduces variance
        • Works well with high-variance models
        • Less prone to overfitting
        • Examples: Random Forest, Extra Trees

        BOOSTING:
        • Sequential training
        • Focus on misclassified samples
        • Reduces bias
        • Adapts to difficult examples
        • Can overfit if not careful
        • Examples: AdaBoost, Gradient Boosting

        Trade-offs:
        • Bagging: Faster (parallel)
        • Boosting: Often more accurate
        • Bagging: More robust
        • Boosting: More sensitive to outliers
        """
        ax8.text(0.05, 0.95, comparison_text, transform=ax8.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        # 9. Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        baseline_acc = self.results['Baseline Tree']['accuracy']
        best_bagging = max([self.results[m]['accuracy'] for m in
                           ['Bagging', 'Random Forest', 'Extra Trees']])
        best_boosting = max([self.results[m]['accuracy'] for m in
                            ['AdaBoost', 'Gradient Boosting']])

        summary_text = f"""
        Summary Statistics
        {'='*40}

        Baseline (Single Tree): {baseline_acc:.4f}

        Best Bagging:  {best_bagging:.4f}
        Improvement:   +{(best_bagging - baseline_acc)*100:.2f}%

        Best Boosting: {best_boosting:.4f}
        Improvement:   +{(best_boosting - baseline_acc)*100:.2f}%

        Key Findings:
        • Both methods improve over baseline
        • Bagging reduces variance
        • Boosting reduces bias
        • Choose based on:
          - Data characteristics
          - Computational resources
          - Overfitting risk
          - Parallelization needs
        """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/boosting_vs_bagging.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/boosting_vs_bagging.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Boosting vs Bagging Analysis")
    print("="*60)

    # Initialize
    analysis = BoostingVsBagging()

    # Create dataset
    df, feature_names = analysis.create_dataset()

    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = analysis.scaler.fit_transform(X_train)
    X_test_scaled = analysis.scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train models
    analysis.train_baseline(X_train_scaled, X_test_scaled, y_train, y_test)
    analysis.train_bagging_models(X_train_scaled, X_test_scaled, y_train, y_test)
    analysis.train_boosting_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Analyze
    analysis.analyze_bias_variance(X_train_scaled, y_train)
    analysis.analyze_convergence(X_train_scaled, X_test_scaled, y_train, y_test)
    analysis.analyze_learning_curves(X_train_scaled, y_train)

    # Visualize
    analysis.visualize_results(y_test)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
