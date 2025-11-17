"""
Out-of-Bag (OOB) Error Analysis
================================

This solution demonstrates Out-of-Bag error estimation in ensemble methods,
comparing it with cross-validation and analyzing its properties for model
selection and performance estimation.

Key Concepts:
- OOB error provides unbiased error estimate without separate validation set
- Approximately 37% of samples are OOB for each bootstrap sample
- OOB error can be used for hyperparameter tuning
- Comparison with cross-validation

Author: Kaggle Solutions Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class OOBAnalyzer:
    """Comprehensive OOB error analyzer."""

    def __init__(self, task='classification', random_state=42):
        """Initialize OOB analyzer."""
        self.task = task
        self.random_state = random_state
        self.results = {}
        np.random.seed(random_state)

    def generate_data(self, n_samples=2000, n_features=20):
        """Generate synthetic dataset."""
        if self.task == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=15,
                n_redundant=3,
                n_classes=2,
                random_state=self.random_state,
                flip_y=0.05
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=15,
                noise=10,
                random_state=self.random_state
            )

        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def train_with_oob(self, X_train, y_train):
        """Train models with OOB scoring enabled."""
        print("Training models with OOB scoring...")

        if self.task == 'classification':
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=200,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'Bagging': BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=10),
                    n_estimators=200,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'ExtraTrees': ExtraTreesClassifier(
                    n_estimators=200,
                    oob_score=True,
                    bootstrap=True,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            }
        else:
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=200,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'Bagging': BaggingRegressor(
                    estimator=DecisionTreeRegressor(max_depth=10),
                    n_estimators=200,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'ExtraTrees': RandomForestRegressor(
                    n_estimators=200,
                    oob_score=True,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1
                )
            }

        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.results[name] = {
                'model': model,
                'oob_score': model.oob_score_
            }
            print(f"    OOB Score: {model.oob_score_:.4f}")

        return models

    def compare_oob_vs_cv(self, X_train, y_train):
        """Compare OOB error with cross-validation."""
        print("\nComparing OOB vs Cross-Validation...")

        comparison_results = {}

        for name, result in self.results.items():
            model = result['model']
            oob_score = result['oob_score']

            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=5,
                scoring='accuracy' if self.task == 'classification' else 'r2',
                n_jobs=-1
            )

            comparison_results[name] = {
                'oob': oob_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }

            print(f"{name}:")
            print(f"  OOB: {oob_score:.4f}")
            print(f"  CV:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Difference: {abs(oob_score - cv_scores.mean()):.4f}")

        self.comparison_results = comparison_results

    def analyze_oob_progression(self, X_train, y_train):
        """Analyze how OOB error changes with number of estimators."""
        print("\nAnalyzing OOB error progression...")

        n_estimators_range = list(range(10, 201, 10))
        oob_errors = {name: [] for name in ['RandomForest', 'Bagging', 'ExtraTrees']}

        for n_est in n_estimators_range:
            if self.task == 'classification':
                models = {
                    'RandomForest': RandomForestClassifier(
                        n_estimators=n_est,
                        oob_score=True,
                        random_state=self.random_state,
                        n_jobs=-1,
                        warm_start=False
                    ),
                    'Bagging': BaggingClassifier(
                        estimator=DecisionTreeClassifier(max_depth=10),
                        n_estimators=n_est,
                        oob_score=True,
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    'ExtraTrees': ExtraTreesClassifier(
                        n_estimators=n_est,
                        oob_score=True,
                        bootstrap=True,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                }
            else:
                models = {
                    'RandomForest': RandomForestRegressor(
                        n_estimators=n_est,
                        oob_score=True,
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    'Bagging': BaggingRegressor(
                        estimator=DecisionTreeRegressor(max_depth=10),
                        n_estimators=n_est,
                        oob_score=True,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                }
                oob_errors.pop('ExtraTrees', None)

            for name, model in models.items():
                model.fit(X_train, y_train)
                if self.task == 'classification':
                    oob_errors[name].append(1 - model.oob_score_)
                else:
                    oob_errors[name].append(1 - model.oob_score_)

        self.oob_progression = {
            'n_estimators': n_estimators_range,
            'errors': oob_errors
        }

    def hyperparameter_tuning_with_oob(self, X_train, y_train):
        """Use OOB for hyperparameter tuning."""
        print("\nHyperparameter tuning using OOB score...")

        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        best_params = {}
        best_score = -np.inf if self.task == 'classification' else -np.inf

        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    if self.task == 'classification':
                        model = RandomForestClassifier(
                            n_estimators=100,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            oob_score=True,
                            random_state=self.random_state,
                            n_jobs=-1
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            oob_score=True,
                            random_state=self.random_state,
                            n_jobs=-1
                        )

                    model.fit(X_train, y_train)

                    if model.oob_score_ > best_score:
                        best_score = model.oob_score_
                        best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf
                        }

        print(f"Best OOB Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")

        self.best_oob_params = best_params
        self.best_oob_score = best_score

    def analyze_oob_sample_distribution(self, n_samples=1000, n_iterations=1000):
        """Analyze OOB sample distribution."""
        print("\nAnalyzing OOB sample distribution...")

        oob_counts = []

        for _ in range(n_iterations):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            unique_indices = np.unique(indices)
            oob_count = n_samples - len(unique_indices)
            oob_counts.append(oob_count)

        theoretical_oob = n_samples * np.exp(-1)

        self.oob_distribution = {
            'counts': oob_counts,
            'mean': np.mean(oob_counts),
            'std': np.std(oob_counts),
            'theoretical': theoretical_oob
        }

        print(f"Mean OOB samples: {np.mean(oob_counts):.1f}")
        print(f"Theoretical OOB: {theoretical_oob:.1f}")
        print(f"OOB percentage: {(np.mean(oob_counts) / n_samples) * 100:.2f}%")

    def plot_oob_vs_cv_comparison(self):
        """Plot OOB vs CV comparison."""
        names = list(self.comparison_results.keys())
        oob_scores = [self.comparison_results[n]['oob'] for n in names]
        cv_means = [self.comparison_results[n]['cv_mean'] for n in names]
        cv_stds = [self.comparison_results[n]['cv_std'] for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, oob_scores, width, label='OOB Score', alpha=0.8)
        bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, label='CV Score', alpha=0.8, capsize=5)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('OOB Score vs Cross-Validation Score')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('oob_vs_cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("OOB vs CV comparison plot saved!")

    def plot_oob_progression(self):
        """Plot OOB error progression."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for name, errors in self.oob_progression['errors'].items():
            ax.plot(self.oob_progression['n_estimators'], errors,
                   marker='o', label=name, linewidth=2)

        ax.set_xlabel('Number of Estimators')
        ax.set_ylabel('OOB Error')
        ax.set_title('OOB Error vs Number of Estimators')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('oob_error_progression.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("OOB progression plot saved!")

    def plot_oob_distribution(self):
        """Plot OOB sample distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        counts = self.oob_distribution['counts']
        theoretical = self.oob_distribution['theoretical']

        # Histogram
        axes[0].hist(counts, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(counts), color='r', linestyle='--',
                       label=f'Empirical Mean: {np.mean(counts):.1f}')
        axes[0].axvline(theoretical, color='g', linestyle='--',
                       label=f'Theoretical: {theoretical:.1f}')
        axes[0].set_xlabel('Number of OOB Samples')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('OOB Sample Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(counts)
        axes[1].axhline(theoretical, color='g', linestyle='--',
                       label=f'Theoretical: {theoretical:.1f}')
        axes[1].set_ylabel('Number of OOB Samples')
        axes[1].set_title('OOB Sample Statistics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('oob_sample_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("OOB sample distribution plot saved!")

    def plot_oob_predictions(self, X_train, y_train):
        """Plot OOB predictions analysis."""
        model = self.results['RandomForest']['model']

        # Get OOB predictions
        oob_predictions = model.oob_prediction_

        if self.task == 'classification':
            # For classification, oob_prediction_ might not exist
            # We'll use oob_decision_function_
            if hasattr(model, 'oob_decision_function_'):
                oob_proba = model.oob_decision_function_
                oob_pred = np.argmax(oob_proba, axis=1)

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_train, oob_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
                axes[0].set_title('OOB Predictions - Confusion Matrix')
                axes[0].set_xlabel('Predicted')
                axes[0].set_ylabel('Actual')

                # Probability distribution
                correct = (oob_pred == y_train)
                axes[1].hist(oob_proba[correct, 1], bins=30, alpha=0.5,
                           label='Correct', edgecolor='black')
                axes[1].hist(oob_proba[~correct, 1], bins=30, alpha=0.5,
                           label='Incorrect', edgecolor='black')
                axes[1].set_xlabel('Prediction Probability')
                axes[1].set_ylabel('Frequency')
                axes[1].set_title('OOB Prediction Probabilities')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Actual vs Predicted
            axes[0].scatter(y_train, oob_predictions, alpha=0.5)
            axes[0].plot([y_train.min(), y_train.max()],
                        [y_train.min(), y_train.max()],
                        'r--', lw=2)
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('OOB Predictions')
            axes[0].set_title('OOB Predictions vs Actual')
            axes[0].grid(True, alpha=0.3)

            # Residuals
            residuals = y_train - oob_predictions
            axes[1].scatter(oob_predictions, residuals, alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('OOB Predictions')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('OOB Residuals')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('oob_predictions_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("OOB predictions analysis plot saved!")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Out-of-Bag (OOB) Error Analysis")
    print("=" * 80)

    # Classification
    print("\n" + "=" * 80)
    print("CLASSIFICATION TASK")
    print("=" * 80)

    analyzer = OOBAnalyzer(task='classification')
    X_train, X_test, y_train, y_test = analyzer.generate_data()

    analyzer.train_with_oob(X_train, y_train)
    analyzer.compare_oob_vs_cv(X_train, y_train)
    analyzer.analyze_oob_progression(X_train, y_train)
    analyzer.hyperparameter_tuning_with_oob(X_train, y_train)
    analyzer.analyze_oob_sample_distribution()

    analyzer.plot_oob_vs_cv_comparison()
    analyzer.plot_oob_progression()
    analyzer.plot_oob_distribution()
    analyzer.plot_oob_predictions(X_train, y_train)

    # Regression
    print("\n" + "=" * 80)
    print("REGRESSION TASK")
    print("=" * 80)

    analyzer_reg = OOBAnalyzer(task='regression')
    X_train, X_test, y_train, y_test = analyzer_reg.generate_data()

    analyzer_reg.train_with_oob(X_train, y_train)
    analyzer_reg.compare_oob_vs_cv(X_train, y_train)
    analyzer_reg.analyze_oob_progression(X_train, y_train)

    print("\n" + "=" * 80)
    print("Analysis complete! All visualizations saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
