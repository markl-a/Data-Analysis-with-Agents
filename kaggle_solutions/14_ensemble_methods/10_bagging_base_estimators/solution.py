"""
Bagging with Different Base Estimators
========================================

This solution demonstrates Bootstrap Aggregating (Bagging) with various base
estimators including Decision Trees, SVMs, Neural Networks, and more.

Key Concepts:
- Bagging reduces variance by training multiple models on bootstrap samples
- Different base estimators have different characteristics
- Diversity in base estimators can improve ensemble performance
- Comparison of weak vs strong base learners

Author: Kaggle Solutions Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc
)
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BaggingAnalyzer:
    """Comprehensive analyzer for Bagging with different base estimators."""

    def __init__(self, task='classification', random_state=42):
        """Initialize the analyzer."""
        self.task = task
        self.random_state = random_state
        self.models = {}
        self.base_models = {}
        self.results = {}

    def generate_data(self):
        """Generate synthetic dataset."""
        if self.task == 'classification':
            X, y = make_classification(
                n_samples=2000,
                n_features=20,
                n_informative=15,
                n_redundant=3,
                n_classes=2,
                random_state=self.random_state,
                flip_y=0.05
            )
        else:
            X, y = make_regression(
                n_samples=2000,
                n_features=20,
                n_informative=15,
                noise=10,
                random_state=self.random_state
            )

        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def get_base_estimators(self):
        """Get various base estimators for bagging."""
        if self.task == 'classification':
            return {
                'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
                'Decision Tree (Deep)': DecisionTreeClassifier(max_depth=None, random_state=self.random_state),
                'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=self.random_state),
                'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=self.random_state),
                'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
                'KNN (k=15)': KNeighborsClassifier(n_neighbors=15),
                'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=self.random_state),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
                'Naive Bayes': GaussianNB()
            }
        else:
            return {
                'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=self.random_state),
                'Decision Tree (Deep)': DecisionTreeRegressor(max_depth=None, random_state=self.random_state),
                'SVR (Linear)': SVR(kernel='linear'),
                'SVR (RBF)': SVR(kernel='rbf'),
                'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
                'KNN (k=15)': KNeighborsRegressor(n_neighbors=15),
                'MLP': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=self.random_state),
                'Ridge': Ridge(random_state=self.random_state)
            }

    def train_bagging_models(self, X_train, y_train):
        """Train bagging models with different base estimators."""
        base_estimators = self.get_base_estimators()

        print(f"Training {len(base_estimators)} bagging models...")

        for name, estimator in base_estimators.items():
            print(f"  - Training with {name}...")

            # Train base model
            base_model = estimator.__class__(**estimator.get_params())
            base_model.fit(X_train, y_train)
            self.base_models[name] = base_model

            # Train bagged version
            if self.task == 'classification':
                bagged_model = BaggingClassifier(
                    estimator=estimator,
                    n_estimators=50,
                    max_samples=0.8,
                    max_features=0.8,
                    bootstrap=True,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                bagged_model = BaggingRegressor(
                    estimator=estimator,
                    n_estimators=50,
                    max_samples=0.8,
                    max_features=0.8,
                    bootstrap=True,
                    random_state=self.random_state,
                    n_jobs=-1
                )

            bagged_model.fit(X_train, y_train)
            self.models[name] = bagged_model

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models."""
        print("\nEvaluating models...")

        for name in self.models.keys():
            if self.task == 'classification':
                # Base model
                base_pred = self.base_models[name].predict(X_test)
                base_acc = accuracy_score(y_test, base_pred)

                # Bagged model
                bagged_pred = self.models[name].predict(X_test)
                bagged_acc = accuracy_score(y_test, bagged_pred)

                self.results[name] = {
                    'base_score': base_acc,
                    'bagged_score': bagged_acc,
                    'improvement': bagged_acc - base_acc,
                    'base_pred': base_pred,
                    'bagged_pred': bagged_pred
                }

                print(f"{name}:")
                print(f"  Base: {base_acc:.4f}")
                print(f"  Bagged: {bagged_acc:.4f}")
                print(f"  Improvement: {bagged_acc - base_acc:+.4f}")

            else:
                # Base model
                base_pred = self.base_models[name].predict(X_test)
                base_r2 = r2_score(y_test, base_pred)

                # Bagged model
                bagged_pred = self.models[name].predict(X_test)
                bagged_r2 = r2_score(y_test, bagged_pred)

                self.results[name] = {
                    'base_score': base_r2,
                    'bagged_score': bagged_r2,
                    'improvement': bagged_r2 - base_r2,
                    'base_pred': base_pred,
                    'bagged_pred': bagged_pred
                }

                print(f"{name}:")
                print(f"  Base: {base_r2:.4f}")
                print(f"  Bagged: {bagged_r2:.4f}")
                print(f"  Improvement: {bagged_r2 - base_r2:+.4f}")

    def plot_performance_comparison(self):
        """Plot performance comparison between base and bagged models."""
        names = list(self.results.keys())
        base_scores = [self.results[name]['base_score'] for name in names]
        bagged_scores = [self.results[name]['bagged_score'] for name in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, base_scores, width, label='Base Model', alpha=0.8)
        bars2 = ax.bar(x + width/2, bagged_scores, width, label='Bagged Model', alpha=0.8)

        ax.set_xlabel('Base Estimator')
        ax.set_ylabel('Score')
        ax.set_title(f'Base vs Bagged Performance ({self.task.capitalize()})')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Performance comparison plot saved!")

    def plot_improvement_analysis(self):
        """Plot improvement from bagging."""
        names = list(self.results.keys())
        improvements = [self.results[name]['improvement'] for name in names]

        # Sort by improvement
        sorted_indices = np.argsort(improvements)
        names_sorted = [names[i] for i in sorted_indices]
        improvements_sorted = [improvements[i] for i in sorted_indices]

        colors = ['red' if x < 0 else 'green' for x in improvements_sorted]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(names_sorted)), improvements_sorted, color=colors, alpha=0.7)

        ax.set_yticks(range(len(names_sorted)))
        ax.set_yticklabels(names_sorted)
        ax.set_xlabel('Improvement (Bagged - Base)')
        ax.set_title('Bagging Improvement by Base Estimator')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Improvement analysis plot saved!")

    def plot_learning_curves(self, X_train, y_train):
        """Plot learning curves for selected models."""
        selected_models = ['Decision Tree (Deep)', 'SVM (RBF)', 'MLP']

        fig, axes = plt.subplots(len(selected_models), 2, figsize=(14, 4 * len(selected_models)))

        if len(selected_models) == 1:
            axes = axes.reshape(1, -1)

        for idx, name in enumerate(selected_models):
            if name not in self.models:
                continue

            for col, (model_type, model) in enumerate([
                ('Base', self.base_models[name]),
                ('Bagged', self.models[name])
            ]):
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_train, y_train,
                    cv=5,
                    n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='accuracy' if self.task == 'classification' else 'r2'
                )

                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)

                axes[idx, col].plot(train_sizes, train_mean, label='Training', marker='o')
                axes[idx, col].fill_between(train_sizes, train_mean - train_std,
                                           train_mean + train_std, alpha=0.3)
                axes[idx, col].plot(train_sizes, val_mean, label='Validation', marker='s')
                axes[idx, col].fill_between(train_sizes, val_mean - val_std,
                                           val_mean + val_std, alpha=0.3)

                axes[idx, col].set_xlabel('Training Size')
                axes[idx, col].set_ylabel('Score')
                axes[idx, col].set_title(f'{name} ({model_type})')
                axes[idx, col].legend()
                axes[idx, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Learning curves plot saved!")

    def plot_variance_reduction(self, X_train, y_train, X_test, y_test):
        """Plot variance reduction from bagging."""
        selected_model = 'Decision Tree (Deep)'

        if selected_model not in self.models:
            return

        n_iterations = 30
        base_scores = []
        bagged_scores = []

        print("\nAnalyzing variance reduction...")

        for i in range(n_iterations):
            # Resample training data
            indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_resample = X_train[indices]
            y_resample = y_train.iloc[indices] if isinstance(y_train, pd.Series) else y_train[indices]

            # Train base model
            base_estimator = self.base_models[selected_model].__class__(
                **self.base_models[selected_model].get_params()
            )
            base_estimator.fit(X_resample, y_resample)
            base_pred = base_estimator.predict(X_test)

            # Train bagged model
            if self.task == 'classification':
                bagged_estimator = BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=None, random_state=i),
                    n_estimators=50,
                    random_state=i,
                    n_jobs=-1
                )
            else:
                bagged_estimator = BaggingRegressor(
                    estimator=DecisionTreeRegressor(max_depth=None, random_state=i),
                    n_estimators=50,
                    random_state=i,
                    n_jobs=-1
                )

            bagged_estimator.fit(X_resample, y_resample)
            bagged_pred = bagged_estimator.predict(X_test)

            # Calculate scores
            if self.task == 'classification':
                base_scores.append(accuracy_score(y_test, base_pred))
                bagged_scores.append(accuracy_score(y_test, bagged_pred))
            else:
                base_scores.append(r2_score(y_test, base_pred))
                bagged_scores.append(r2_score(y_test, bagged_pred))

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Score distribution
        axes[0].hist(base_scores, bins=20, alpha=0.5, label='Base Model', edgecolor='black')
        axes[0].hist(bagged_scores, bins=20, alpha=0.5, label='Bagged Model', edgecolor='black')
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Score Distribution Across Resamples')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Variance comparison
        variance_data = [base_scores, bagged_scores]
        bp = axes[1].boxplot(variance_data, labels=['Base Model', 'Bagged Model'])
        axes[1].set_ylabel('Score')
        axes[1].set_title(f'Variance Reduction\nBase Var: {np.var(base_scores):.4f}, Bagged Var: {np.var(bagged_scores):.4f}')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('variance_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Variance reduction plot saved!")

    def plot_ensemble_diversity(self):
        """Plot diversity of predictions in ensemble."""
        selected_model = 'Decision Tree (Deep)'

        if selected_model not in self.models:
            return

        bagged_model = self.models[selected_model]

        # Get predictions from individual estimators
        predictions = np.array([estimator.predict(X_test)
                               for estimator in bagged_model.estimators_])

        # Calculate prediction diversity
        n_estimators = len(bagged_model.estimators_)
        diversity_matrix = np.zeros((n_estimators, n_estimators))

        for i in range(n_estimators):
            for j in range(n_estimators):
                if self.task == 'classification':
                    diversity_matrix[i, j] = np.mean(predictions[i] != predictions[j])
                else:
                    diversity_matrix[i, j] = np.corrcoef(predictions[i], predictions[j])[0, 1]

        plt.figure(figsize=(10, 8))
        sns.heatmap(diversity_matrix, cmap='viridis', center=0 if self.task == 'regression' else None)
        plt.title(f'Ensemble Diversity Matrix ({selected_model})')
        plt.xlabel('Estimator Index')
        plt.ylabel('Estimator Index')

        plt.tight_layout()
        plt.savefig('ensemble_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Ensemble diversity plot saved!")

    def analyze_bootstrap_samples(self, X_train):
        """Analyze bootstrap sampling characteristics."""
        n_samples = len(X_train)
        n_iterations = 1000

        unique_samples = []
        oob_samples = []

        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            unique_samples.append(len(np.unique(indices)))
            oob_samples.append(n_samples - len(np.unique(indices)))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Unique samples distribution
        axes[0].hist(unique_samples, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(unique_samples), color='r', linestyle='--',
                       label=f'Mean: {np.mean(unique_samples):.1f}')
        axes[0].axvline(n_samples * (1 - np.exp(-1)), color='g', linestyle='--',
                       label=f'Theoretical: {n_samples * (1 - np.exp(-1)):.1f}')
        axes[0].set_xlabel('Unique Samples')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Bootstrap Sample Size Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # OOB samples distribution
        axes[1].hist(oob_samples, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(oob_samples), color='r', linestyle='--',
                       label=f'Mean: {np.mean(oob_samples):.1f}')
        axes[1].axvline(n_samples * np.exp(-1), color='g', linestyle='--',
                       label=f'Theoretical: {n_samples * np.exp(-1):.1f}')
        axes[1].set_xlabel('Out-of-Bag Samples')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('OOB Sample Size Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bootstrap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Bootstrap analysis plot saved!")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Bagging with Different Base Estimators")
    print("=" * 80)

    # Classification
    print("\n" + "=" * 80)
    print("CLASSIFICATION TASK")
    print("=" * 80)

    analyzer_clf = BaggingAnalyzer(task='classification')
    X_train, X_test, y_train, y_test = analyzer_clf.generate_data()

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    analyzer_clf.train_bagging_models(X_train, y_train)
    analyzer_clf.evaluate_models(X_test, y_test)

    analyzer_clf.plot_performance_comparison()
    analyzer_clf.plot_improvement_analysis()
    analyzer_clf.plot_learning_curves(X_train, y_train)
    analyzer_clf.plot_variance_reduction(X_train, y_train, X_test, y_test)
    analyzer_clf.plot_ensemble_diversity()
    analyzer_clf.analyze_bootstrap_samples(X_train)

    # Regression
    print("\n" + "=" * 80)
    print("REGRESSION TASK")
    print("=" * 80)

    analyzer_reg = BaggingAnalyzer(task='regression')
    X_train, X_test, y_train, y_test = analyzer_reg.generate_data()

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    analyzer_reg.train_bagging_models(X_train, y_train)
    analyzer_reg.evaluate_models(X_test, y_test)

    print("\n" + "=" * 80)
    print("Analysis complete! All visualizations saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
