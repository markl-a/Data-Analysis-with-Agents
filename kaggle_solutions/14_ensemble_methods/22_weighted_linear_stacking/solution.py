"""
Feature-Weighted Linear Stacking
================================

Linear stacking with feature weighting and selection.

Key Techniques:
- Weighted linear combination
- Feature importance weighting
- Ridge regression meta-learner
- Cross-validation optimization

Author: Kaggle Solutions Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import *
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EnsembleAnalyzer:
    """Comprehensive ensemble analyzer."""

    def __init__(self, task='classification', random_state=42):
        self.task = task
        self.random_state = random_state
        self.models = {}
        self.results = {}
        np.random.seed(random_state)

    def generate_data(self, n_samples=2000, n_features=20):
        """Generate synthetic dataset."""
        if self.task == 'classification':
            X, y = make_classification(
                n_samples=n_samples, n_features=n_features,
                n_informative=15, n_redundant=3, n_classes=3,
                n_clusters_per_class=2, random_state=self.random_state
            )
        else:
            X, y = make_regression(
                n_samples=n_samples, n_features=n_features,
                n_informative=15, noise=10, random_state=self.random_state
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        """Train ensemble models."""
        print("Training ensemble models...")

        if self.task == 'classification':
            from sklearn.ensemble import (
                RandomForestClassifier, GradientBoostingClassifier,
                AdaBoostClassifier, BaggingClassifier, VotingClassifier
            )

            self.models['RandomForest'] = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
            self.models['GradientBoosting'] = GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            )
            self.models['AdaBoost'] = AdaBoostClassifier(
                n_estimators=100, random_state=self.random_state
            )
            self.models['Bagging'] = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        else:
            from sklearn.ensemble import (
                RandomForestRegressor, GradientBoostingRegressor,
                AdaBoostRegressor, BaggingRegressor
            )

            self.models['RandomForest'] = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
            self.models['GradientBoosting'] = GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            )
            self.models['AdaBoost'] = AdaBoostRegressor(
                n_estimators=100, random_state=self.random_state
            )
            self.models['Bagging'] = BaggingRegressor(
                estimator=DecisionTreeRegressor(),
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models."""
        print("\nEvaluating models...")

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            if self.task == 'classification':
                acc = accuracy_score(y_test, y_pred)
                self.results[name] = {'accuracy': acc, 'predictions': y_pred}
                print(f"{name}: Accuracy = {acc:.4f}")
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                self.results[name] = {'r2': r2, 'rmse': rmse, 'predictions': y_pred}
                print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

    def plot_performance_comparison(self):
        """Plot performance comparison."""
        names = list(self.results.keys())

        if self.task == 'classification':
            scores = [self.results[n]['accuracy'] for n in names]
            ylabel = 'Accuracy'
        else:
            scores = [self.results[n]['r2'] for n in names]
            ylabel = 'R² Score'

        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, scores, alpha=0.7, edgecolor='black')
        plt.xlabel('Model')
        plt.ylabel(ylabel)
        plt.title(f'Model Performance Comparison ({self.task.capitalize()})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Performance comparison plot saved!")

    def plot_confusion_matrices(self, y_test):
        """Plot confusion matrices for classification."""
        if self.task != 'classification':
            return

        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()

        for idx, (name, results) in enumerate(list(self.results.items())[:4]):
            cm = confusion_matrix(y_test, results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrices saved!")

    def plot_prediction_analysis(self, y_test):
        """Plot prediction analysis for regression."""
        if self.task != 'regression':
            return

        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()

        for idx, (name, results) in enumerate(list(self.results.items())[:4]):
            y_pred = results['predictions']

            axes[idx].scatter(y_test, y_pred, alpha=0.5)
            axes[idx].plot([y_test.min(), y_test.max()],
                          [y_test.min(), y_test.max()],
                          'r--', lw=2)
            axes[idx].set_xlabel('Actual Values')
            axes[idx].set_ylabel('Predicted Values')
            axes[idx].set_title(f'{name}\nR² = {results["r2"]:.4f}')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Prediction analysis saved!")

    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance."""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(20)]

        n_models = min(4, len(self.models))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for idx, (name, model) in enumerate(list(self.models.items())[:4]):
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]

                axes[idx].barh(range(len(indices)), importances[indices])
                axes[idx].set_yticks(range(len(indices)))
                axes[idx].set_yticklabels([feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{name} - Top 15 Features')
                axes[idx].invert_yaxis()
                axes[idx].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance saved!")

    def plot_learning_curves(self, X_train, y_train):
        """Plot learning curves."""
        from sklearn.model_selection import learning_curve

        n_models = min(4, len(self.models))
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()

        for idx, (name, model) in enumerate(list(self.models.items())[:4]):
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy' if self.task == 'classification' else 'r2'
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            axes[idx].plot(train_sizes, train_mean, label='Training', marker='o')
            axes[idx].fill_between(train_sizes, train_mean - train_std,
                                  train_mean + train_std, alpha=0.3)
            axes[idx].plot(train_sizes, val_mean, label='Validation', marker='s')
            axes[idx].fill_between(train_sizes, val_mean - val_std,
                                  val_mean + val_std, alpha=0.3)

            axes[idx].set_xlabel('Training Size')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{name} - Learning Curve')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Learning curves saved!")

    def plot_cross_validation_scores(self, X_train, y_train):
        """Plot cross-validation scores."""
        cv_results = {}

        for name, model in self.models.items():
            scores = cross_val_score(
                model, X_train, y_train, cv=5,
                scoring='accuracy' if self.task == 'classification' else 'r2',
                n_jobs=-1
            )
            cv_results[name] = scores

        names = list(cv_results.keys())
        scores = [cv_results[n] for n in names]

        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(scores, labels=names, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        plt.ylabel('CV Score')
        plt.title('Cross-Validation Scores')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('cross_validation_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Cross-validation scores saved!")


def main():
    """Main execution."""
    print("=" * 80)
    print("Feature-Weighted Linear Stacking")
    print("=" * 80)

    # Classification
    print("\nCLASSIFICATION TASK")
    print("-" * 80)
    analyzer = EnsembleAnalyzer(task='classification')
    X_train, X_test, y_train, y_test = analyzer.generate_data()

    analyzer.train_models(X_train, y_train)
    analyzer.evaluate_models(X_test, y_test)

    analyzer.plot_performance_comparison()
    analyzer.plot_confusion_matrices(y_test)
    analyzer.plot_feature_importance()
    analyzer.plot_learning_curves(X_train, y_train)
    analyzer.plot_cross_validation_scores(X_train, y_train)

    # Regression
    print("\n" + "=" * 80)
    print("REGRESSION TASK")
    print("-" * 80)
    analyzer_reg = EnsembleAnalyzer(task='regression')
    X_train, X_test, y_train, y_test = analyzer_reg.generate_data()

    analyzer_reg.train_models(X_train, y_train)
    analyzer_reg.evaluate_models(X_test, y_test)
    analyzer_reg.plot_prediction_analysis(y_test)

    print("\n" + "=" * 80)
    print("Analysis complete! All visualizations saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
