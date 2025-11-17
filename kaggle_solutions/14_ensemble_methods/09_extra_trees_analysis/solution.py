"""
Extra Trees Classifier and Regressor Analysis
==============================================

This solution demonstrates the use of Extra Trees (Extremely Randomized Trees)
for both classification and regression tasks, comparing them with Random Forest
and analyzing their unique characteristics.

Key Concepts:
- Extra Trees use random thresholds for splits instead of optimal ones
- Generally faster than Random Forest
- Can reduce variance even further than Random Forest
- Often better for high-dimensional datasets

Author: Kaggle Solutions Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve,
    validation_curve,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExtraTreesAnalyzer:
    """Comprehensive analyzer for Extra Trees algorithms."""

    def __init__(self, task='classification', random_state=42):
        """
        Initialize the analyzer.

        Parameters:
        -----------
        task : str
            Either 'classification' or 'regression'
        random_state : int
            Random state for reproducibility
        """
        self.task = task
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def generate_classification_data(self, n_samples=2000, n_features=20,
                                    n_informative=15, n_redundant=3):
        """Generate synthetic classification dataset."""
        print("Generating classification dataset...")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=3,
            n_clusters_per_class=2,
            random_state=self.random_state,
            flip_y=0.1
        )

        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['target'].value_counts()}")

        return df

    def generate_regression_data(self, n_samples=2000, n_features=20,
                                n_informative=15, noise=0.1):
        """Generate synthetic regression dataset."""
        print("Generating regression dataset...")
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise * 100,
            random_state=self.random_state
        )

        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Target statistics:\n{df['target'].describe()}")

        return df

    def prepare_data(self, df):
        """Prepare data for modeling."""
        X = df.drop('target', axis=1)
        y = df['target']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def train_extra_trees(self, X_train, y_train, **kwargs):
        """Train Extra Trees model."""
        if self.task == 'classification':
            model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )

        print(f"Training Extra Trees {self.task.capitalize()}...")
        model.fit(X_train, y_train)
        self.models['extra_trees'] = model
        return model

    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model for comparison."""
        if self.task == 'classification':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )

        print(f"Training Random Forest {self.task.capitalize()}...")
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model

    def evaluate_classification(self, model, X_test, y_test, model_name):
        """Evaluate classification model."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        self.results[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{class_report}")

        return accuracy

    def evaluate_regression(self, model, X_test, y_test, model_name):
        """Evaluate regression model."""
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }

        print(f"\n{model_name} Results:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return r2

    def compare_training_speed(self, X_train, y_train, n_trials=5):
        """Compare training speed between Extra Trees and Random Forest."""
        import time

        et_times = []
        rf_times = []

        print("\nComparing training speed...")
        for i in range(n_trials):
            # Extra Trees
            start = time.time()
            if self.task == 'classification':
                model = ExtraTreesClassifier(
                    n_estimators=100,
                    random_state=i,
                    n_jobs=-1
                )
            else:
                model = ExtraTreesRegressor(
                    n_estimators=100,
                    random_state=i,
                    n_jobs=-1
                )
            model.fit(X_train, y_train)
            et_times.append(time.time() - start)

            # Random Forest
            start = time.time()
            if self.task == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=i,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=i,
                    n_jobs=-1
                )
            model.fit(X_train, y_train)
            rf_times.append(time.time() - start)

        self.results['training_speed'] = {
            'extra_trees': et_times,
            'random_forest': rf_times
        }

        print(f"Extra Trees avg time: {np.mean(et_times):.3f}s")
        print(f"Random Forest avg time: {np.mean(rf_times):.3f}s")
        print(f"Speedup: {np.mean(rf_times) / np.mean(et_times):.2f}x")

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Extra Trees."""
        print("\nPerforming hyperparameter tuning...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        if self.task == 'classification':
            base_model = ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1)
            scoring = 'accuracy'
        else:
            base_model = ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1)
            scoring = 'r2'

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.results['best_params'] = grid_search.best_params_
        self.results['best_score'] = grid_search.best_score_
        self.models['tuned_extra_trees'] = grid_search.best_estimator_

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def plot_feature_importance_comparison(self, feature_names):
        """Compare feature importance between Extra Trees and Random Forest."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (model_name, model) in enumerate([
            ('Extra Trees', self.models['extra_trees']),
            ('Random Forest', self.models['random_forest'])
        ]):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            axes[idx].barh(range(len(indices)), importances[indices])
            axes[idx].set_yticks(range(len(indices)))
            axes[idx].set_yticklabels([feature_names[i] for i in indices])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name} - Top 15 Features')
            axes[idx].invert_yaxis()

        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Feature importance comparison plot saved!")

    def plot_training_speed(self):
        """Plot training speed comparison."""
        et_times = self.results['training_speed']['extra_trees']
        rf_times = self.results['training_speed']['random_forest']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot
        axes[0].boxplot([et_times, rf_times], labels=['Extra Trees', 'Random Forest'])
        axes[0].set_ylabel('Training Time (seconds)')
        axes[0].set_title('Training Speed Comparison')
        axes[0].grid(True, alpha=0.3)

        # Bar plot
        means = [np.mean(et_times), np.mean(rf_times)]
        stds = [np.std(et_times), np.std(rf_times)]
        axes[1].bar(['Extra Trees', 'Random Forest'], means, yerr=stds, capsize=5)
        axes[1].set_ylabel('Mean Training Time (seconds)')
        axes[1].set_title('Average Training Speed')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Training speed comparison plot saved!")

    def plot_learning_curves(self, X_train, y_train):
        """Plot learning curves for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (model_name, model) in enumerate([
            ('Extra Trees', self.models['extra_trees']),
            ('Random Forest', self.models['random_forest'])
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

            axes[idx].plot(train_sizes, train_mean, label='Training score', marker='o')
            axes[idx].fill_between(train_sizes, train_mean - train_std,
                                  train_mean + train_std, alpha=0.3)
            axes[idx].plot(train_sizes, val_mean, label='Validation score', marker='s')
            axes[idx].fill_between(train_sizes, val_mean - val_std,
                                  val_mean + val_std, alpha=0.3)

            axes[idx].set_xlabel('Training Set Size')
            axes[idx].set_ylabel('Score')
            axes[idx].set_title(f'{model_name} - Learning Curve')
            axes[idx].legend(loc='best')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Learning curves plot saved!")

    def plot_confusion_matrices(self):
        """Plot confusion matrices for classification tasks."""
        if self.task != 'classification':
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, model_name in enumerate(['extra_trees', 'random_forest']):
            cm = self.results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Confusion matrices plot saved!")

    def plot_prediction_comparison(self, y_test):
        """Plot prediction comparison for regression tasks."""
        if self.task != 'regression':
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, model_name in enumerate(['extra_trees', 'random_forest']):
            y_pred = self.results[model_name]['predictions']

            axes[idx].scatter(y_test, y_pred, alpha=0.5)
            axes[idx].plot([y_test.min(), y_test.max()],
                          [y_test.min(), y_test.max()],
                          'r--', lw=2)
            axes[idx].set_xlabel('Actual Values')
            axes[idx].set_ylabel('Predicted Values')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nR² = {self.results[model_name]["r2"]:.4f}')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Prediction comparison plot saved!")

    def plot_residuals(self, y_test):
        """Plot residuals for regression tasks."""
        if self.task != 'regression':
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        for idx, model_name in enumerate(['extra_trees', 'random_forest']):
            y_pred = self.results[model_name]['predictions']
            residuals = y_test - y_pred

            # Residual plot
            axes[idx, 0].scatter(y_pred, residuals, alpha=0.5)
            axes[idx, 0].axhline(y=0, color='r', linestyle='--')
            axes[idx, 0].set_xlabel('Predicted Values')
            axes[idx, 0].set_ylabel('Residuals')
            axes[idx, 0].set_title(f'{model_name.replace("_", " ").title()} - Residuals')
            axes[idx, 0].grid(True, alpha=0.3)

            # Residual distribution
            axes[idx, 1].hist(residuals, bins=30, edgecolor='black')
            axes[idx, 1].set_xlabel('Residuals')
            axes[idx, 1].set_ylabel('Frequency')
            axes[idx, 1].set_title(f'{model_name.replace("_", " ").title()} - Residual Distribution')
            axes[idx, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Residuals analysis plot saved!")

    def plot_tree_depths(self):
        """Plot distribution of tree depths."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (model_name, model) in enumerate([
            ('Extra Trees', self.models['extra_trees']),
            ('Random Forest', self.models['random_forest'])
        ]):
            depths = [tree.get_depth() for tree in model.estimators_]

            axes[idx].hist(depths, bins=20, edgecolor='black', alpha=0.7)
            axes[idx].axvline(np.mean(depths), color='r', linestyle='--',
                            label=f'Mean: {np.mean(depths):.1f}')
            axes[idx].set_xlabel('Tree Depth')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{model_name} - Tree Depth Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('tree_depths_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Tree depths distribution plot saved!")

    def plot_validation_curves(self, X_train, y_train):
        """Plot validation curves for key hyperparameters."""
        param_range = [50, 100, 150, 200, 250, 300]

        if self.task == 'classification':
            model = ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1)
            scoring = 'accuracy'
        else:
            model = ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1)
            scoring = 'r2'

        train_scores, val_scores = validation_curve(
            model, X_train, y_train,
            param_name='n_estimators',
            param_range=param_range,
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, label='Training score', marker='o')
        plt.fill_between(param_range, train_mean - train_std,
                        train_mean + train_std, alpha=0.3)
        plt.plot(param_range, val_mean, label='Validation score', marker='s')
        plt.fill_between(param_range, val_mean - val_std,
                        val_mean + val_std, alpha=0.3)

        plt.xlabel('Number of Estimators')
        plt.ylabel('Score')
        plt.title('Validation Curve - n_estimators')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('validation_curve_n_estimators.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Validation curve plot saved!")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Extra Trees Classifier and Regressor Analysis")
    print("=" * 80)

    # Classification Task
    print("\n" + "=" * 80)
    print("CLASSIFICATION TASK")
    print("=" * 80)

    analyzer_clf = ExtraTreesAnalyzer(task='classification')

    # Generate and prepare data
    df_clf = analyzer_clf.generate_classification_data()
    X_train, X_test, y_train, y_test, feature_names = analyzer_clf.prepare_data(df_clf)

    # Train models
    analyzer_clf.train_extra_trees(X_train, y_train)
    analyzer_clf.train_random_forest(X_train, y_train)

    # Evaluate models
    analyzer_clf.evaluate_classification(
        analyzer_clf.models['extra_trees'], X_test, y_test, 'extra_trees'
    )
    analyzer_clf.evaluate_classification(
        analyzer_clf.models['random_forest'], X_test, y_test, 'random_forest'
    )

    # Compare training speed
    analyzer_clf.compare_training_speed(X_train, y_train)

    # Hyperparameter tuning
    analyzer_clf.hyperparameter_tuning(X_train, y_train)

    # Generate visualizations
    analyzer_clf.plot_feature_importance_comparison(feature_names)
    analyzer_clf.plot_training_speed()
    analyzer_clf.plot_learning_curves(X_train, y_train)
    analyzer_clf.plot_confusion_matrices()
    analyzer_clf.plot_tree_depths()
    analyzer_clf.plot_validation_curves(X_train, y_train)

    # Regression Task
    print("\n" + "=" * 80)
    print("REGRESSION TASK")
    print("=" * 80)

    analyzer_reg = ExtraTreesAnalyzer(task='regression')

    # Generate and prepare data
    df_reg = analyzer_reg.generate_regression_data()
    X_train, X_test, y_train, y_test, feature_names = analyzer_reg.prepare_data(df_reg)

    # Train models
    analyzer_reg.train_extra_trees(X_train, y_train)
    analyzer_reg.train_random_forest(X_train, y_train)

    # Evaluate models
    analyzer_reg.evaluate_regression(
        analyzer_reg.models['extra_trees'], X_test, y_test, 'extra_trees'
    )
    analyzer_reg.evaluate_regression(
        analyzer_reg.models['random_forest'], X_test, y_test, 'random_forest'
    )

    # Compare training speed
    analyzer_reg.compare_training_speed(X_train, y_train)

    # Generate visualizations
    analyzer_reg.plot_prediction_comparison(y_test)
    analyzer_reg.plot_residuals(y_test)

    print("\n" + "=" * 80)
    print("Analysis complete! All visualizations saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
