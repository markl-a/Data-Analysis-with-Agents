"""
Gradient Boosting Machines Analysis
===================================

Comprehensive implementation demonstrating gradient boosting machines analysis.

Author: Kaggle Solutions Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import *
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EnsembleAnalyzer:
    """Comprehensive ensemble analyzer with advanced features."""

    def __init__(self, task='classification', random_state=42):
        """Initialize analyzer."""
        self.task = task
        self.random_state = random_state
        self.models = {}
        self.results = {}
        np.random.seed(random_state)

    def generate_data(self, n_samples=2000, n_features=20, n_informative=15):
        """Generate synthetic dataset with configurable complexity."""
        print(f"Generating {self.task} dataset...")
        
        if self.task == 'classification':
            X, y = make_classification(
                n_samples=n_samples, n_features=n_features,
                n_informative=n_informative, n_redundant=3, n_classes=2,
                n_clusters_per_class=2, random_state=self.random_state,
                flip_y=0.05, class_sep=0.8
            )
        else:
            X, y = make_regression(
                n_samples=n_samples, n_features=n_features,
                n_informative=n_informative, noise=10,
                random_state=self.random_state, bias=100
            )

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y if self.task == 'classification' else None
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    def create_base_models(self):
        """Create diverse base models for ensemble."""
        if self.task == 'classification':
            base_models = {
                'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=self.random_state),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.random_state),
                'SVM': SVC(probability=True, random_state=self.random_state),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=self.random_state)
            }
        else:
            base_models = {
                'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=self.random_state),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
                'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=self.random_state),
                'Ridge': Ridge(random_state=self.random_state),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'MLP': MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=self.random_state)
            }
        
        return base_models

    def train_models(self, X_train, y_train):
        """Train specialized ensemble models."""
        print("Training ensemble models...")

        base_models = self.create_base_models()
        
        # Select top 4 base models for ensemble
        selected_models = list(base_models.items())[:4]
        
        for name, model in selected_models:
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model


    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation."""
        print("\nEvaluating models...")

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            if self.task == 'classification':
                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba.shape[1] == 2:
                        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                except:
                    auc_score = 0.0

                self.results[name] = {
                    'accuracy': acc, 'precision': precision, 'recall': recall,
                    'f1': f1, 'auc': auc_score, 'predictions': y_pred
                }
                
                print(f"{name}:")
                print(f"  Accuracy: {acc:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  AUC: {auc_score:.4f}")
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

                self.results[name] = {
                    'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                    'predictions': y_pred
                }
                
                print(f"{name}:")
                print(f"  R² Score: {r2:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  MAPE: {mape:.2f}%")

    def plot_performance_comparison(self):
        """Create comprehensive performance comparison plot."""
        names = list(self.results.keys())

        if self.task == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()

            for idx, metric in enumerate(metrics):
                scores = [self.results[n][metric] for n in names]
                axes[idx].bar(names, scores, alpha=0.7, edgecolor='black')
                axes[idx].set_ylabel(metric.capitalize())
                axes[idx].set_title(f'{metric.capitalize()} Comparison')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3, axis='y')
                
                for i, (name, score) in enumerate(zip(names, scores)):
                    axes[idx].text(i, score, f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            metrics = ['r2', 'rmse', 'mae', 'mape']
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()

            for idx, metric in enumerate(metrics):
                scores = [self.results[n][metric] for n in names]
                axes[idx].bar(names, scores, alpha=0.7, edgecolor='black')
                axes[idx].set_ylabel(metric.upper())
                axes[idx].set_title(f'{metric.upper()} Comparison')
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Performance comparison plot saved!")

    def plot_confusion_matrices(self, y_test):
        """Plot confusion matrices for classification."""
        if self.task != 'classification':
            return

        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for idx, (name, results) in enumerate(self.results.items()):
            if idx < len(axes):
                cm = confusion_matrix(y_test, results['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Confusion matrices plot saved!")

    def plot_prediction_scatter(self, y_test):
        """Plot prediction scatter plots for regression."""
        if self.task != 'regression':
            return

        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for idx, (name, results) in enumerate(self.results.items()):
            if idx < len(axes):
                y_pred = results['predictions']
                
                axes[idx].scatter(y_test, y_pred, alpha=0.5)
                axes[idx].plot([y_test.min(), y_test.max()],
                              [y_test.min(), y_test.max()],
                              'r--', lw=2)
                axes[idx].set_xlabel('Actual')
                axes[idx].set_ylabel('Predicted')
                axes[idx].set_title(f'{name}\nR²: {results["r2"]:.3f}')
                axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Prediction scatter plot saved!")

    def plot_feature_importance(self, n_features=15):
        """Plot feature importance for models that support it."""
        feature_names = [f'Feature {i}' for i in range(20)]
        
        importances_dict = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances_dict[name] = model.feature_importances_
        
        if not importances_dict:
            return

        n_models = len(importances_dict)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for idx, (name, importances) in enumerate(importances_dict.items()):
            if idx < len(axes):
                indices = np.argsort(importances)[::-1][:n_features]
                
                axes[idx].barh(range(len(indices)), importances[indices])
                axes[idx].set_yticks(range(len(indices)))
                axes[idx].set_yticklabels([feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{name} - Top {n_features} Features')
                axes[idx].invert_yaxis()
                axes[idx].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved!")

    def plot_residuals(self, y_test):
        """Plot residuals for regression models."""
        if self.task != 'regression':
            return

        n_models = len(self.models)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.ravel()

        for idx, (name, results) in enumerate(self.results.items()):
            if idx < len(axes):
                y_pred = results['predictions']
                residuals = y_test - y_pred
                
                axes[idx].scatter(y_pred, residuals, alpha=0.5)
                axes[idx].axhline(y=0, color='r', linestyle='--', linewidth=2)
                axes[idx].set_xlabel('Predicted Values')
                axes[idx].set_ylabel('Residuals')
                axes[idx].set_title(f'{name} - Residuals')
                axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Residuals plot saved!")

    def cross_validate_models(self, X, y, cv=5):
        """Perform cross-validation on all models."""
        print("\nPerforming cross-validation...")
        
        cv_results = {}
        scoring = 'accuracy' if self.task == 'classification' else 'r2'
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                cv_results[name] = scores
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                print(f"{name}: Cross-validation failed - {e}")
                cv_results[name] = np.array([0.0])

        # Plot CV results
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
        print("Cross-validation scores plot saved!")


def main():
    """Main execution."""
    print("=" * 80)
    print("Gradient Boosting Machines Analysis")
    print("=" * 80)

    # Classification Task
    print("\nCLASSIFICATION TASK")
    print("-" * 80)
    analyzer = EnsembleAnalyzer(task='classification')
    X_train, X_test, y_train, y_test = analyzer.generate_data()

    analyzer.train_models(X_train, y_train)
    analyzer.evaluate_models(X_test, y_test)

    analyzer.plot_performance_comparison()
    analyzer.plot_confusion_matrices(y_test)
    analyzer.plot_feature_importance()
    
    # Cross-validation
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    analyzer.cross_validate_models(X_full, y_full)

    # Regression Task
    print("\n" + "=" * 80)
    print("REGRESSION TASK")
    print("-" * 80)
    analyzer_reg = EnsembleAnalyzer(task='regression')
    X_train, X_test, y_train, y_test = analyzer_reg.generate_data()

    analyzer_reg.train_models(X_train, y_train)
    analyzer_reg.evaluate_models(X_test, y_test)
    analyzer_reg.plot_prediction_scatter(y_test)
    analyzer_reg.plot_residuals(y_test)

    print("\n" + "=" * 80)
    print("Analysis complete! All visualizations saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
