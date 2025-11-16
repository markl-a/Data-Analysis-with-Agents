"""
Hyperparameter Tuning for Ensembles
Comprehensive hyperparameter optimization for ensemble methods

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐⭐ Expert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, GridSearchCV, RandomizedSearchCV,
                                      cross_val_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
import time
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class HyperparameterTuning:
    """Comprehensive Hyperparameter Tuning Analysis"""

    def __init__(self):
        self.models = {}
        self.tuning_results = {}
        self.best_params = {}
        self.scaler = StandardScaler()

    def create_dataset(self):
        """Create synthetic classification dataset"""
        print("Creating synthetic dataset...")

        X, y = make_classification(
            n_samples=2500,
            n_features=25,
            n_informative=18,
            n_redundant=4,
            n_repeated=3,
            n_classes=2,
            n_clusters_per_class=2,
            weights=[0.6, 0.4],
            flip_y=0.05,
            class_sep=0.75,
            random_state=42
        )

        feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['target'].value_counts()}")

        return df, feature_names

    def baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models with default parameters"""
        print("\n" + "="*60)
        print("Training Baseline Models (Default Parameters)")
        print("="*60)

        models_def = {
            'RF_baseline': RandomForestClassifier(random_state=42, n_jobs=-1),
            'GB_baseline': GradientBoostingClassifier(random_state=42)
        }

        for name, model in models_def.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Training time: {train_time:.2f}s")

            self.models[name] = model
            self.tuning_results[name] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'params': model.get_params()
            }

    def grid_search_rf(self, X_train, y_train):
        """Grid search for Random Forest"""
        print("\n" + "="*60)
        print("Grid Search - Random Forest")
        print("="*60)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        print("Starting grid search...")
        start_time = time.time()

        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1, return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time

        print(f"\nGrid search completed in {search_time:.2f}s")
        print(f"Best score: {grid_search.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        self.models['RF_grid'] = grid_search.best_estimator_
        self.best_params['RF_grid'] = grid_search.best_params_
        self.tuning_results['RF_grid'] = {
            'best_score': grid_search.best_score_,
            'search_time': search_time,
            'cv_results': grid_search.cv_results_
        }

        return grid_search

    def random_search_rf(self, X_train, y_train):
        """Random search for Random Forest"""
        print("\n" + "="*60)
        print("Random Search - Random Forest")
        print("="*60)

        param_distributions = {
            'n_estimators': randint(50, 300),
            'max_depth': [10, 20, 30, 40, None],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        print("Starting random search (100 iterations)...")
        start_time = time.time()

        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=100, cv=5,
            scoring='accuracy', n_jobs=-1, verbose=1,
            random_state=42, return_train_score=True
        )

        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time

        print(f"\nRandom search completed in {search_time:.2f}s")
        print(f"Best score: {random_search.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")

        self.models['RF_random'] = random_search.best_estimator_
        self.best_params['RF_random'] = random_search.best_params_
        self.tuning_results['RF_random'] = {
            'best_score': random_search.best_score_,
            'search_time': search_time,
            'cv_results': random_search.cv_results_
        }

        return random_search

    def grid_search_gb(self, X_train, y_train):
        """Grid search for Gradient Boosting"""
        print("\n" + "="*60)
        print("Grid Search - Gradient Boosting")
        print("="*60)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 1.0]
        }

        print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

        gb = GradientBoostingClassifier(random_state=42)

        print("Starting grid search...")
        start_time = time.time()

        grid_search = GridSearchCV(
            gb, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1, return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time

        print(f"\nGrid search completed in {search_time:.2f}s")
        print(f"Best score: {grid_search.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        self.models['GB_grid'] = grid_search.best_estimator_
        self.best_params['GB_grid'] = grid_search.best_params_
        self.tuning_results['GB_grid'] = {
            'best_score': grid_search.best_score_,
            'search_time': search_time,
            'cv_results': grid_search.cv_results_
        }

        return grid_search

    def analyze_parameter_importance(self):
        """Analyze which parameters matter most"""
        print("\n" + "="*60)
        print("Analyzing Parameter Importance")
        print("="*60)

        param_effects = {}

        # Random Forest analysis
        if 'RF_random' in self.tuning_results:
            cv_results = self.tuning_results['RF_random']['cv_results']
            params_df = pd.DataFrame(cv_results['params'])
            scores = cv_results['mean_test_score']

            print("\nRandom Forest - Parameter Effects:")
            for param in params_df.columns:
                unique_vals = params_df[param].unique()
                if len(unique_vals) > 1:
                    param_scores = {}
                    for val in unique_vals:
                        mask = params_df[param] == val
                        param_scores[str(val)] = scores[mask].mean()

                    best_val = max(param_scores, key=param_scores.get)
                    worst_val = min(param_scores, key=param_scores.get)
                    effect = param_scores[best_val] - param_scores[worst_val]

                    param_effects[f'RF_{param}'] = effect
                    print(f"  {param}: {effect:.4f} range")

        self.param_effects = param_effects

    def compare_all_models(self, X_test, y_test):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("Final Model Comparison")
        print("="*60)

        comparison_results = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            comparison_results[name] = {
                'accuracy': accuracy
            }

            print(f"{name}: {accuracy:.4f}")

        self.comparison_results = comparison_results

        return comparison_results

    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # 1. Model Accuracy Comparison
        ax1 = plt.subplot(3, 3, 1)
        model_names = list(self.comparison_results.keys())
        accuracies = [self.comparison_results[m]['accuracy'] for m in model_names]

        colors = ['gray', 'gray', 'blue', 'green', 'orange'][:len(model_names)]
        bars = ax1.barh(range(len(model_names)), accuracies, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(model_names)))
        ax1.set_yticklabels(model_names, fontsize=9)
        ax1.set_xlabel('Test Accuracy', fontsize=10)
        ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlim([min(accuracies) - 0.02, 1.0])
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Search Time Comparison
        ax2 = plt.subplot(3, 3, 2)
        tuned_models = ['RF_grid', 'RF_random', 'GB_grid']
        search_times = [self.tuning_results[m]['search_time'] for m in tuned_models if m in self.tuning_results]
        tuned_names = [m for m in tuned_models if m in self.tuning_results]

        bars = ax2.bar(range(len(tuned_names)), search_times, color=['blue', 'green', 'orange'][:len(tuned_names)],
                      edgecolor='black')
        ax2.set_xticks(range(len(tuned_names)))
        ax2.set_xticklabels(tuned_names, rotation=45, ha='right')
        ax2.set_ylabel('Time (seconds)', fontsize=10)
        ax2.set_title('Hyperparameter Search Time', fontsize=12, fontweight='bold')
        for i, (bar, t) in enumerate(zip(bars, search_times)):
            ax2.text(i, t + 5, f'{t:.1f}s', ha='center', va='bottom')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Improvement from Baseline
        ax3 = plt.subplot(3, 3, 3)
        baseline_rf = self.comparison_results['RF_baseline']['accuracy']
        baseline_gb = self.comparison_results['GB_baseline']['accuracy']

        improvements = []
        labels = []

        if 'RF_grid' in self.comparison_results:
            imp = (self.comparison_results['RF_grid']['accuracy'] - baseline_rf) * 100
            improvements.append(imp)
            labels.append('RF Grid\nSearch')

        if 'RF_random' in self.comparison_results:
            imp = (self.comparison_results['RF_random']['accuracy'] - baseline_rf) * 100
            improvements.append(imp)
            labels.append('RF Random\nSearch')

        if 'GB_grid' in self.comparison_results:
            imp = (self.comparison_results['GB_grid']['accuracy'] - baseline_gb) * 100
            improvements.append(imp)
            labels.append('GB Grid\nSearch')

        colors3 = ['blue', 'green', 'orange'][:len(improvements)]
        bars = ax3.bar(range(len(improvements)), improvements, color=colors3, edgecolor='black')
        ax3.set_xticks(range(len(improvements)))
        ax3.set_xticklabels(labels, fontsize=9)
        ax3.set_ylabel('Improvement (%)', fontsize=10)
        ax3.set_title('Improvement Over Baseline', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax3.text(i, imp + 0.05 if imp > 0 else imp - 0.05,
                    f'{imp:.2f}%', ha='center',
                    va='bottom' if imp > 0 else 'top')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. RF Grid Search - Top Parameters
        if 'RF_grid' in self.tuning_results:
            ax4 = plt.subplot(3, 3, 4)
            cv_results = self.tuning_results['RF_grid']['cv_results']

            # Get top 10 configurations
            indices = np.argsort(cv_results['mean_test_score'])[-10:][::-1]
            scores = cv_results['mean_test_score'][indices]

            bars = ax4.barh(range(10), scores, color='blue', alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(10))
            ax4.set_yticklabels([f'Config {i+1}' for i in range(10)])
            ax4.set_xlabel('CV Score', fontsize=10)
            ax4.set_title('RF Grid Search - Top 10 Configs', fontsize=12, fontweight='bold')
            ax4.invert_yaxis()
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax4.text(score + 0.002, i, f'{score:.4f}', va='center', fontsize=8)
            ax4.grid(True, alpha=0.3, axis='x')

        # 5. RF Random Search - Score Distribution
        if 'RF_random' in self.tuning_results:
            ax5 = plt.subplot(3, 3, 5)
            cv_results = self.tuning_results['RF_random']['cv_results']
            scores = cv_results['mean_test_score']

            ax5.hist(scores, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax5.axvline(scores.max(), color='red', linestyle='--',
                       linewidth=2, label=f'Best: {scores.max():.4f}')
            ax5.axvline(scores.mean(), color='blue', linestyle='--',
                       linewidth=2, label=f'Mean: {scores.mean():.4f}')
            ax5.set_xlabel('CV Score', fontsize=10)
            ax5.set_ylabel('Frequency', fontsize=10)
            ax5.set_title('RF Random Search - Score Distribution', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        # 6. GB Grid Search - Parameter Effect
        if 'GB_grid' in self.tuning_results:
            ax6 = plt.subplot(3, 3, 6)
            cv_results = self.tuning_results['GB_grid']['cv_results']

            # Analyze learning_rate effect
            params_df = pd.DataFrame(cv_results['params'])
            scores = cv_results['mean_test_score']

            if 'learning_rate' in params_df.columns:
                lr_scores = {}
                for lr in params_df['learning_rate'].unique():
                    mask = params_df['learning_rate'] == lr
                    lr_scores[lr] = scores[mask].mean()

                lrs = sorted(lr_scores.keys())
                lr_means = [lr_scores[lr] for lr in lrs]

                ax6.plot(lrs, lr_means, 'o-', linewidth=2, markersize=10, color='orange')
                ax6.set_xlabel('Learning Rate', fontsize=10)
                ax6.set_ylabel('Mean CV Score', fontsize=10)
                ax6.set_title('GB - Learning Rate Effect', fontsize=12, fontweight='bold')
                ax6.set_xscale('log')
                ax6.grid(True, alpha=0.3)

        # 7. Parameter Importance (if available)
        if hasattr(self, 'param_effects'):
            ax7 = plt.subplot(3, 3, 7)
            sorted_effects = sorted(self.param_effects.items(), key=lambda x: x[1], reverse=True)
            params = [p[0] for p in sorted_effects[:10]]
            effects = [p[1] for p in sorted_effects[:10]]

            bars = ax7.barh(range(len(params)), effects, color='purple', alpha=0.7, edgecolor='black')
            ax7.set_yticks(range(len(params)))
            ax7.set_yticklabels(params, fontsize=8)
            ax7.set_xlabel('Score Range', fontsize=10)
            ax7.set_title('Parameter Importance (Score Range)', fontsize=12, fontweight='bold')
            ax7.invert_yaxis()
            for i, (bar, eff) in enumerate(zip(bars, effects)):
                ax7.text(eff + 0.001, i, f'{eff:.4f}', va='center', fontsize=7)
            ax7.grid(True, alpha=0.3, axis='x')

        # 8. Best Parameters Summary - RF
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')

        rf_params_text = "Random Forest Best Parameters\n" + "="*35 + "\n\n"

        if 'RF_grid' in self.best_params:
            rf_params_text += "Grid Search:\n"
            for param, value in self.best_params['RF_grid'].items():
                rf_params_text += f"  {param}: {value}\n"
            rf_params_text += f"  Score: {self.tuning_results['RF_grid']['best_score']:.4f}\n"

        if 'RF_random' in self.best_params:
            rf_params_text += "\nRandom Search:\n"
            for param, value in self.best_params['RF_random'].items():
                rf_params_text += f"  {param}: {value}\n"
            rf_params_text += f"  Score: {self.tuning_results['RF_random']['best_score']:.4f}\n"

        ax8.text(0.1, 0.95, rf_params_text, transform=ax8.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 9. Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = f"""
        Hyperparameter Tuning Summary
        {'='*40}

        Baseline Models:
        • RF Default:  {self.comparison_results['RF_baseline']['accuracy']:.4f}
        • GB Default:  {self.comparison_results['GB_baseline']['accuracy']:.4f}

        Tuned Models:
        """

        if 'RF_grid' in self.comparison_results:
            summary_text += f"• RF Grid:    {self.comparison_results['RF_grid']['accuracy']:.4f}\n        "
        if 'RF_random' in self.comparison_results:
            summary_text += f"• RF Random:  {self.comparison_results['RF_random']['accuracy']:.4f}\n        "
        if 'GB_grid' in self.comparison_results:
            summary_text += f"• GB Grid:    {self.comparison_results['GB_grid']['accuracy']:.4f}\n        "

        summary_text += f"""

        Search Strategies:
        • Grid Search: Exhaustive
        • Random Search: More efficient
        • Best for large spaces: Random

        Key Takeaways:
        • Tuning improves performance
        • Random search is faster
        • Grid search more thorough
        • Start with random, refine with grid
        """

        ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/hyperparameter_tuning.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Hyperparameter Tuning for Ensemble Methods")
    print("="*60)

    # Initialize
    tuning = HyperparameterTuning()

    # Create dataset
    df, feature_names = tuning.create_dataset()

    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = tuning.scaler.fit_transform(X_train)
    X_test_scaled = tuning.scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train baseline models
    tuning.baseline_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Grid search for RF
    tuning.grid_search_rf(X_train_scaled, y_train)

    # Random search for RF
    tuning.random_search_rf(X_train_scaled, y_train)

    # Grid search for GB
    tuning.grid_search_gb(X_train_scaled, y_train)

    # Analyze parameter importance
    tuning.analyze_parameter_importance()

    # Compare all models
    tuning.compare_all_models(X_test_scaled, y_test)

    # Visualize
    tuning.visualize_results()

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
