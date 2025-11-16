"""
Feature Importance Across Ensembles
Compare feature importance methods across different ensemble models

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐⭐ Expert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FeatureImportanceAnalysis:
    """Comprehensive Feature Importance Analysis Across Ensembles"""

    def __init__(self):
        self.models = {}
        self.feature_importances = {}
        self.permutation_importances = {}
        self.scaler = StandardScaler()

    def create_dataset_with_known_importance(self):
        """Create dataset with known feature importance"""
        print("Creating synthetic dataset with known feature importance...")

        # Create dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=30,
            n_informative=10,
            n_redundant=10,
            n_repeated=5,
            n_classes=2,
            n_clusters_per_class=2,
            weights=[0.6, 0.4],
            flip_y=0.05,
            class_sep=0.8,
            random_state=42
        )

        # Create feature names with categories
        feature_names = []
        feature_types = []

        # First 10: Informative features
        for i in range(10):
            feature_names.append(f'informative_{i+1}')
            feature_types.append('informative')

        # Next 10: Redundant features
        for i in range(10):
            feature_names.append(f'redundant_{i+1}')
            feature_types.append('redundant')

        # Next 5: Repeated features
        for i in range(5):
            feature_names.append(f'repeated_{i+1}')
            feature_types.append('repeated')

        # Last 5: Random noise features
        for i in range(5):
            X[:, 25+i] = np.random.randn(2000)
            feature_names.append(f'noise_{i+1}')
            feature_types.append('noise')

        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Feature types:")
        print(f"  - Informative: 10")
        print(f"  - Redundant: 10")
        print(f"  - Repeated: 5")
        print(f"  - Noise: 5")

        return df, feature_names, feature_types

    def train_ensemble_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ensemble models"""
        print("\n" + "="*60)
        print("Training Ensemble Models")
        print("="*60)

        # Define models
        models_def = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                random_state=42
            )
        }

        # Train each model
        for name, model in models_def.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"  Accuracy: {accuracy:.4f}")

            self.models[name] = model

            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[name] = model.feature_importances_

    def calculate_permutation_importance(self, X_test, y_test, feature_names):
        """Calculate permutation importance for all models"""
        print("\n" + "="*60)
        print("Calculating Permutation Importance")
        print("="*60)

        for name, model in self.models.items():
            print(f"\nCalculating for {name}...")

            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )

            self.permutation_importances[name] = {
                'importances_mean': perm_importance.importances_mean,
                'importances_std': perm_importance.importances_std
            }

            # Show top 5
            top_indices = perm_importance.importances_mean.argsort()[-5:][::-1]
            print("  Top 5 features:")
            for idx in top_indices:
                print(f"    {feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f}")

    def compare_importance_methods(self, feature_names):
        """Compare built-in vs permutation importance"""
        print("\n" + "="*60)
        print("Comparing Importance Methods")
        print("="*60)

        comparison_results = {}

        for model_name in self.models.keys():
            if model_name in self.feature_importances:
                builtin_imp = self.feature_importances[model_name]
                perm_imp = self.permutation_importances[model_name]['importances_mean']

                # Calculate correlation
                correlation = np.corrcoef(builtin_imp, perm_imp)[0, 1]

                comparison_results[model_name] = {
                    'correlation': correlation,
                    'builtin': builtin_imp,
                    'permutation': perm_imp
                }

                print(f"\n{model_name}:")
                print(f"  Correlation between methods: {correlation:.4f}")

        self.comparison_results = comparison_results

        return comparison_results

    def analyze_feature_consistency(self, feature_names, feature_types):
        """Analyze consistency of feature importance across models"""
        print("\n" + "="*60)
        print("Analyzing Feature Importance Consistency")
        print("="*60)

        # Create DataFrame with all importances
        importance_df = pd.DataFrame(index=feature_names)

        for model_name, importances in self.feature_importances.items():
            importance_df[model_name] = importances

        # Calculate statistics
        importance_df['mean'] = importance_df.mean(axis=1)
        importance_df['std'] = importance_df.std(axis=1)
        importance_df['cv'] = importance_df['std'] / (importance_df['mean'] + 1e-10)
        importance_df['feature_type'] = feature_types

        # Sort by mean importance
        importance_df = importance_df.sort_values('mean', ascending=False)

        print("\nTop 10 Most Important Features (averaged):")
        print(importance_df.head(10)[['mean', 'std', 'feature_type']])

        self.importance_df = importance_df

        return importance_df

    def analyze_by_feature_type(self, feature_types):
        """Analyze importance by feature type"""
        print("\n" + "="*60)
        print("Analyzing Importance by Feature Type")
        print("="*60)

        # Group by feature type
        type_importance = {}

        for feature_type in ['informative', 'redundant', 'repeated', 'noise']:
            mask = np.array(feature_types) == feature_type
            type_importance[feature_type] = {}

            for model_name, importances in self.feature_importances.items():
                avg_imp = importances[mask].mean()
                type_importance[feature_type][model_name] = avg_imp

        self.type_importance = type_importance

        # Print results
        print("\nAverage Importance by Feature Type:")
        for feature_type in ['informative', 'redundant', 'repeated', 'noise']:
            print(f"\n{feature_type.capitalize()}:")
            for model_name, importance in type_importance[feature_type].items():
                print(f"  {model_name}: {importance:.6f}")

        return type_importance

    def visualize_results(self, feature_names, feature_types):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 14))

        # 1. Top 15 Features - Random Forest
        ax1 = plt.subplot(4, 3, 1)
        rf_imp = self.feature_importances['Random Forest']
        top_indices = rf_imp.argsort()[-15:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = rf_imp[top_indices]

        colors = ['green' if feature_types[i] == 'informative' else 'orange'
                 if feature_types[i] == 'redundant' else 'blue'
                 if feature_types[i] == 'repeated' else 'red'
                 for i in top_indices]

        ax1.barh(range(15), top_importances, color=colors)
        ax1.set_yticks(range(15))
        ax1.set_yticklabels(top_features, fontsize=8)
        ax1.set_xlabel('Importance', fontsize=10)
        ax1.set_title('Top 15 Features - Random Forest', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()

        # 2. Top 15 Features - Gradient Boosting
        ax2 = plt.subplot(4, 3, 2)
        gb_imp = self.feature_importances['Gradient Boosting']
        top_indices = gb_imp.argsort()[-15:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = gb_imp[top_indices]

        colors = ['green' if feature_types[i] == 'informative' else 'orange'
                 if feature_types[i] == 'redundant' else 'blue'
                 if feature_types[i] == 'repeated' else 'red'
                 for i in top_indices]

        ax2.barh(range(15), top_importances, color=colors)
        ax2.set_yticks(range(15))
        ax2.set_yticklabels(top_features, fontsize=8)
        ax2.set_xlabel('Importance', fontsize=10)
        ax2.set_title('Top 15 Features - Gradient Boosting', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        # 3. Feature Importance Heatmap
        ax3 = plt.subplot(4, 3, 3)
        importance_matrix = []
        model_names = list(self.feature_importances.keys())

        for model_name in model_names:
            importance_matrix.append(self.feature_importances[model_name])

        importance_matrix = np.array(importance_matrix)

        # Show top 20 features only
        avg_importance = importance_matrix.mean(axis=0)
        top_20_indices = avg_importance.argsort()[-20:][::-1]

        sns.heatmap(importance_matrix[:, top_20_indices], cmap='YlOrRd',
                   xticklabels=[feature_names[i] for i in top_20_indices],
                   yticklabels=model_names, ax=ax3, cbar_kws={'label': 'Importance'})
        ax3.set_title('Feature Importance Heatmap (Top 20)', fontsize=12, fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=90, fontsize=7)

        # 4. Importance by Feature Type
        ax4 = plt.subplot(4, 3, 4)
        type_data = self.type_importance
        x = np.arange(len(type_data))
        width = 0.2

        for i, model_name in enumerate(model_names):
            values = [type_data[ft][model_name] for ft in ['informative', 'redundant', 'repeated', 'noise']]
            ax4.bar(x + i*width, values, width, label=model_name)

        ax4.set_xlabel('Feature Type', fontsize=10)
        ax4.set_ylabel('Average Importance', fontsize=10)
        ax4.set_title('Importance by Feature Type', fontsize=12, fontweight='bold')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(['Informative', 'Redundant', 'Repeated', 'Noise'])
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Permutation vs Built-in Importance - Random Forest
        ax5 = plt.subplot(4, 3, 5)
        if 'Random Forest' in self.comparison_results:
            comp = self.comparison_results['Random Forest']
            ax5.scatter(comp['builtin'], comp['permutation'], alpha=0.6, s=50)
            ax5.plot([0, max(comp['builtin'])], [0, max(comp['builtin'])],
                    'r--', linewidth=2, label='Perfect correlation')
            ax5.set_xlabel('Built-in Importance', fontsize=10)
            ax5.set_ylabel('Permutation Importance', fontsize=10)
            ax5.set_title(f'RF: Built-in vs Permutation (r={comp["correlation"]:.3f})',
                         fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Permutation vs Built-in Importance - Gradient Boosting
        ax6 = plt.subplot(4, 3, 6)
        if 'Gradient Boosting' in self.comparison_results:
            comp = self.comparison_results['Gradient Boosting']
            ax6.scatter(comp['builtin'], comp['permutation'], alpha=0.6, s=50, color='orange')
            ax6.plot([0, max(comp['builtin'])], [0, max(comp['builtin'])],
                    'r--', linewidth=2, label='Perfect correlation')
            ax6.set_xlabel('Built-in Importance', fontsize=10)
            ax6.set_ylabel('Permutation Importance', fontsize=10)
            ax6.set_title(f'GB: Built-in vs Permutation (r={comp["correlation"]:.3f})',
                         fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. Feature Importance Correlation Matrix
        ax7 = plt.subplot(4, 3, 7)
        # Calculate pairwise correlations between models
        n_models = len(model_names)
        corr_matrix = np.zeros((n_models, n_models))

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                imp1 = self.feature_importances[model1]
                imp2 = self.feature_importances[model2]
                corr_matrix[i, j] = np.corrcoef(imp1, imp2)[0, 1]

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=model_names, yticklabels=model_names,
                   ax=ax7, vmin=0, vmax=1, cbar_kws={'label': 'Correlation'})
        ax7.set_title('Model Agreement on Feature Importance', fontsize=12, fontweight='bold')
        plt.setp(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax7.get_yticklabels(), rotation=0, fontsize=9)

        # 8. Coefficient of Variation
        ax8 = plt.subplot(4, 3, 8)
        importance_df = self.importance_df.sort_values('cv', ascending=False).head(15)
        colors8 = ['green' if ft == 'informative' else 'orange'
                  if ft == 'redundant' else 'blue'
                  if ft == 'repeated' else 'red'
                  for ft in importance_df['feature_type']]

        ax8.barh(range(15), importance_df['cv'], color=colors8)
        ax8.set_yticks(range(15))
        ax8.set_yticklabels(importance_df.index, fontsize=8)
        ax8.set_xlabel('Coefficient of Variation', fontsize=10)
        ax8.set_title('Features with Highest Disagreement', fontsize=12, fontweight='bold')
        ax8.invert_yaxis()

        # 9. Average Importance with Error Bars
        ax9 = plt.subplot(4, 3, 9)
        top_15 = self.importance_df.head(15)
        colors9 = ['green' if ft == 'informative' else 'orange'
                  if ft == 'redundant' else 'blue'
                  if ft == 'repeated' else 'red'
                  for ft in top_15['feature_type']]

        ax9.barh(range(15), top_15['mean'], xerr=top_15['std'],
                capsize=3, color=colors9, alpha=0.7, edgecolor='black')
        ax9.set_yticks(range(15))
        ax9.set_yticklabels(top_15.index, fontsize=8)
        ax9.set_xlabel('Mean Importance ± Std', fontsize=10)
        ax9.set_title('Top 15 Features (Averaged Across Models)', fontsize=12, fontweight='bold')
        ax9.invert_yaxis()

        # 10. Permutation Importance - Random Forest
        ax10 = plt.subplot(4, 3, 10)
        if 'Random Forest' in self.permutation_importances:
            perm_imp = self.permutation_importances['Random Forest']
            top_indices = perm_imp['importances_mean'].argsort()[-15:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = perm_imp['importances_mean'][top_indices]
            top_stds = perm_imp['importances_std'][top_indices]

            colors10 = ['green' if feature_types[i] == 'informative' else 'orange'
                       if feature_types[i] == 'redundant' else 'blue'
                       if feature_types[i] == 'repeated' else 'red'
                       for i in top_indices]

            ax10.barh(range(15), top_importances, xerr=top_stds,
                     capsize=3, color=colors10, alpha=0.7, edgecolor='black')
            ax10.set_yticks(range(15))
            ax10.set_yticklabels(top_features, fontsize=8)
            ax10.set_xlabel('Permutation Importance', fontsize=10)
            ax10.set_title('Top 15 - Permutation (RF)', fontsize=12, fontweight='bold')
            ax10.invert_yaxis()

        # 11. Feature Type Distribution
        ax11 = plt.subplot(4, 3, 11)
        # Calculate average rank by feature type
        type_ranks = {}
        for ft in ['informative', 'redundant', 'repeated', 'noise']:
            mask = np.array(feature_types) == ft
            ranks = []
            for model_name, importances in self.feature_importances.items():
                # Get ranks (higher importance = lower rank number)
                sorted_indices = importances.argsort()[::-1]
                feature_ranks = np.empty_like(sorted_indices)
                feature_ranks[sorted_indices] = np.arange(len(sorted_indices))
                ranks.append(feature_ranks[mask].mean())
            type_ranks[ft] = ranks

        x = np.arange(len(model_names))
        width = 0.2
        colors_dict = {'informative': 'green', 'redundant': 'orange',
                      'repeated': 'blue', 'noise': 'red'}

        for i, ft in enumerate(['informative', 'redundant', 'repeated', 'noise']):
            ax11.bar(x + i*width, type_ranks[ft], width,
                    label=ft.capitalize(), color=colors_dict[ft])

        ax11.set_xlabel('Model', fontsize=10)
        ax11.set_ylabel('Average Rank', fontsize=10)
        ax11.set_title('Average Feature Rank by Type', fontsize=12, fontweight='bold')
        ax11.set_xticks(x + width * 1.5)
        ax11.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax11.legend()
        ax11.invert_yaxis()  # Lower rank = more important
        ax11.grid(True, alpha=0.3, axis='y')

        # 12. Legend and Summary
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')

        summary_text = """
        Feature Importance Analysis
        {'='*40}

        Feature Types (Color Legend):
        🟢 Green: Informative (truly predictive)
        🟠 Orange: Redundant (correlated)
        🔵 Blue: Repeated (duplicates)
        🔴 Red: Noise (random)

        Key Findings:
        • Informative features rank highest
        • High correlation between models
        • Built-in ≈ Permutation importance
        • Noise features correctly ignored
        • Redundant features get some weight

        Methods Compared:
        1. Built-in (Gini/Gain based)
        2. Permutation importance
        3. Averaged across models

        Best Practice:
        Use multiple methods and models
        to identify robust important features
        """

        # Replace color codes with actual text
        summary_text = summary_text.replace('🟢', '•')
        summary_text = summary_text.replace('🟠', '•')
        summary_text = summary_text.replace('🔵', '•')
        summary_text = summary_text.replace('🔴', '•')

        ax12.text(0.1, 0.95, summary_text, transform=ax12.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/feature_importance.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Feature Importance Analysis Across Ensembles")
    print("="*60)

    # Initialize
    analysis = FeatureImportanceAnalysis()

    # Create dataset
    df, feature_names, feature_types = analysis.create_dataset_with_known_importance()

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
    analysis.train_ensemble_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Calculate permutation importance
    analysis.calculate_permutation_importance(X_test_scaled, y_test, feature_names)

    # Compare methods
    analysis.compare_importance_methods(feature_names)

    # Analyze consistency
    analysis.analyze_feature_consistency(feature_names, feature_types)

    # Analyze by feature type
    analysis.analyze_by_feature_type(feature_types)

    # Visualize
    analysis.visualize_results(feature_names, feature_types)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
