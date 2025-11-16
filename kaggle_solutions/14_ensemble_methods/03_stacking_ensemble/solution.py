"""
Stacking Ensemble - Multi-Level Model Stacking
Building powerful stacked ensemble classifiers

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐⭐ Expert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StackingEnsemble:
    """Stacking Ensemble Analysis and Comparison"""

    def __init__(self):
        self.base_models = {}
        self.stacked_models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def create_dataset(self):
        """Create synthetic classification dataset"""
        print("Creating synthetic dataset...")

        # Create dataset
        X, y = make_classification(
            n_samples=3000,
            n_features=25,
            n_informative=18,
            n_redundant=4,
            n_repeated=3,
            n_classes=3,
            n_clusters_per_class=2,
            weights=[0.5, 0.3, 0.2],
            flip_y=0.05,
            class_sep=0.7,
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

    def train_base_models(self, X_train, X_test, y_train, y_test):
        """Train individual base models"""
        print("\n" + "="*60)
        print("Training Base Models")
        print("="*60)

        # Define base models
        base_models_def = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
        }

        # Train and evaluate each base model
        for name, model in base_models_def.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Store results
            self.base_models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }

            print(f"  Accuracy: {accuracy:.4f}")

        return base_models_def

    def create_stacking_level1(self, X_train, X_test, y_train, y_test):
        """Create Level 1 stacking with 3 base models"""
        print("\n" + "="*60)
        print("Creating Level 1 Stacking Ensemble")
        print("="*60)

        # Define base estimators
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ]

        # Create stacking classifier with Logistic Regression as meta-model
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        )

        print("Training stacking model...")
        stacking_clf.fit(X_train, y_train)

        # Predictions
        y_pred = stacking_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Level 1 Stacking Accuracy: {accuracy:.4f}")

        self.stacked_models['level1'] = stacking_clf
        self.results['Stacking L1'] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }

        return stacking_clf

    def create_stacking_level2(self, X_train, X_test, y_train, y_test):
        """Create Level 2 stacking with 5 diverse base models"""
        print("\n" + "="*60)
        print("Creating Level 2 Stacking Ensemble (5 models)")
        print("="*60)

        # Define base estimators
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]

        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        )

        print("Training stacking model...")
        stacking_clf.fit(X_train, y_train)

        # Predictions
        y_pred = stacking_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Level 2 Stacking Accuracy: {accuracy:.4f}")

        self.stacked_models['level2'] = stacking_clf
        self.results['Stacking L2'] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }

        return stacking_clf

    def create_stacking_custom_meta(self, X_train, X_test, y_train, y_test):
        """Create stacking with different meta-models"""
        print("\n" + "="*60)
        print("Testing Different Meta-Models")
        print("="*60)

        # Define base estimators (same for all)
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ]

        # Test different meta-models
        meta_models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'SVM': SVC(random_state=42)
        }

        meta_results = {}

        for meta_name, meta_model in meta_models.items():
            print(f"\nTesting meta-model: {meta_name}")

            stacking_clf = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )

            stacking_clf.fit(X_train, y_train)
            y_pred = stacking_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            meta_results[meta_name] = accuracy
            print(f"  Accuracy: {accuracy:.4f}")

        self.results['meta_comparison'] = meta_results

        return meta_results

    def analyze_cross_validation(self, X, y):
        """Perform cross-validation analysis"""
        print("\n" + "="*60)
        print("Cross-Validation Analysis")
        print("="*60)

        # Models to compare
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Stacking (3 models)': self.stacked_models['level1'],
            'Stacking (5 models)': self.stacked_models['level2']
        }

        cv_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            print(f"\nCross-validating {name}...")
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

        self.results['cv_results'] = cv_results

        return cv_results

    def analyze_prediction_diversity(self, X_test, y_test):
        """Analyze diversity of base model predictions"""
        print("\n" + "="*60)
        print("Analyzing Prediction Diversity")
        print("="*60)

        # Get predictions from base models
        base_model_names = ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'SVM', 'KNN']
        predictions = {}

        for name in base_model_names:
            if name in self.base_models:
                predictions[name] = self.base_models[name].predict(X_test)

        # Calculate pairwise agreement
        n_models = len(predictions)
        agreement_matrix = np.zeros((n_models, n_models))

        model_list = list(predictions.keys())
        for i, model1 in enumerate(model_list):
            for j, model2 in enumerate(model_list):
                agreement = np.mean(predictions[model1] == predictions[model2])
                agreement_matrix[i, j] = agreement

        self.results['diversity'] = {
            'agreement_matrix': agreement_matrix,
            'model_names': model_list
        }

        print("\nPairwise Agreement Matrix:")
        df_agreement = pd.DataFrame(
            agreement_matrix,
            index=model_list,
            columns=model_list
        )
        print(df_agreement.round(3))

        return agreement_matrix

    def visualize_results(self, X_test, y_test):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # 1. Base Models Comparison
        ax1 = plt.subplot(3, 3, 1)
        base_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting',
                      'Logistic Regression', 'SVM', 'KNN', 'Naive Bayes', 'Decision Tree']
        base_accuracies = [self.results[m]['accuracy'] for m in base_models if m in self.results]
        base_names = [m for m in base_models if m in self.results]

        colors = plt.cm.viridis(np.linspace(0, 1, len(base_names)))
        bars = ax1.barh(range(len(base_names)), base_accuracies, color=colors)
        ax1.set_yticks(range(len(base_names)))
        ax1.set_yticklabels(base_names)
        ax1.set_xlabel('Accuracy', fontsize=10)
        ax1.set_title('Base Models Performance', fontsize=12, fontweight='bold')
        ax1.set_xlim([0, 1])
        for i, (bar, acc) in enumerate(zip(bars, base_accuracies)):
            ax1.text(acc + 0.01, i, f'{acc:.4f}', va='center')
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Stacking vs Best Base Model
        ax2 = plt.subplot(3, 3, 2)
        comparison_models = ['Random Forest', 'Gradient Boosting', 'Stacking L1', 'Stacking L2']
        comparison_accs = [self.results[m]['accuracy'] for m in comparison_models if m in self.results]
        comparison_names = [m for m in comparison_models if m in self.results]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(comparison_names)]
        bars = ax2.bar(range(len(comparison_names)), comparison_accs, color=colors)
        ax2.set_xticks(range(len(comparison_names)))
        ax2.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Stacking vs Base Models', fontsize=12, fontweight='bold')
        ax2.set_ylim([min(comparison_accs) - 0.05, 1.0])
        for i, (bar, acc) in enumerate(zip(bars, comparison_accs)):
            ax2.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Meta-Model Comparison
        if 'meta_comparison' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            meta_results = self.results['meta_comparison']
            meta_names = list(meta_results.keys())
            meta_accs = list(meta_results.values())

            bars = ax3.bar(range(len(meta_names)), meta_accs, color='skyblue', edgecolor='black')
            ax3.set_xticks(range(len(meta_names)))
            ax3.set_xticklabels(meta_names, rotation=45, ha='right')
            ax3.set_ylabel('Accuracy', fontsize=10)
            ax3.set_title('Meta-Model Comparison', fontsize=12, fontweight='bold')
            for i, (bar, acc) in enumerate(zip(bars, meta_accs)):
                ax3.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Cross-Validation Results
        if 'cv_results' in self.results:
            ax4 = plt.subplot(3, 3, 4)
            cv_results = self.results['cv_results']
            cv_names = list(cv_results.keys())
            cv_means = [cv_results[m]['mean'] for m in cv_names]
            cv_stds = [cv_results[m]['std'] for m in cv_names]

            x_pos = np.arange(len(cv_names))
            ax4.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color='coral', edgecolor='black', alpha=0.7)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(cv_names, rotation=45, ha='right')
            ax4.set_ylabel('CV Accuracy', fontsize=10)
            ax4.set_title('Cross-Validation Comparison', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')

        # 5. Prediction Diversity Heatmap
        if 'diversity' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            diversity = self.results['diversity']
            sns.heatmap(diversity['agreement_matrix'],
                       annot=True, fmt='.3f', cmap='RdYlGn_r',
                       xticklabels=diversity['model_names'],
                       yticklabels=diversity['model_names'],
                       ax=ax5, cbar_kws={'label': 'Agreement'}, vmin=0.5, vmax=1.0)
            ax5.set_title('Model Prediction Agreement', fontsize=12, fontweight='bold')
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

        # 6. Confusion Matrix - Best Stacking Model
        ax6 = plt.subplot(3, 3, 6)
        best_stacking = 'Stacking L2' if 'Stacking L2' in self.results else 'Stacking L1'
        cm = confusion_matrix(y_test, self.results[best_stacking]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6)
        ax6.set_xlabel('Predicted', fontsize=10)
        ax6.set_ylabel('Actual', fontsize=10)
        ax6.set_title(f'Confusion Matrix - {best_stacking}', fontsize=12, fontweight='bold')

        # 7. Architecture Diagram
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        architecture_text = """
        Stacking Architecture

        Layer 1 (Base Models):
        ├── Random Forest
        ├── Extra Trees
        ├── Gradient Boosting
        ├── SVM
        └── KNN
               ↓
        Layer 2 (Meta-Model):
        └── Logistic Regression
               ↓
        Final Predictions
        """
        ax7.text(0.1, 0.9, architecture_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 8. Improvement Analysis
        ax8 = plt.subplot(3, 3, 8)
        rf_acc = self.results['Random Forest']['accuracy']
        stack_l1_acc = self.results['Stacking L1']['accuracy']
        stack_l2_acc = self.results['Stacking L2']['accuracy']

        improvements = [
            0,  # baseline
            (stack_l1_acc - rf_acc) * 100,
            (stack_l2_acc - rf_acc) * 100
        ]
        labels = ['RF\n(baseline)', 'Stacking\nL1', 'Stacking\nL2']

        bars = ax8.bar(range(3), improvements, color=['gray', 'green', 'darkgreen'])
        ax8.set_xticks(range(3))
        ax8.set_xticklabels(labels)
        ax8.set_ylabel('Improvement (%)', fontsize=10)
        ax8.set_title('Improvement Over Baseline', fontsize=12, fontweight='bold')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax8.text(i, imp + 0.1, f'{imp:.2f}%', ha='center', va='bottom')
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = f"""
        Stacking Ensemble Summary
        {'='*40}

        Best Base Model:
        • Random Forest: {rf_acc:.4f}

        Stacking Models:
        • Level 1 (3 models): {stack_l1_acc:.4f}
        • Level 2 (5 models): {stack_l2_acc:.4f}

        Improvement:
        • Level 1: +{(stack_l1_acc - rf_acc)*100:.2f}%
        • Level 2: +{(stack_l2_acc - rf_acc)*100:.2f}%

        Key Insights:
        • Stacking combines diverse models
        • Meta-model learns from predictions
        • Achieves better generalization
        • Trade-off: Complexity vs Performance
        """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/stacking_ensemble.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/stacking_ensemble.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Stacking Ensemble Analysis")
    print("="*60)

    # Initialize
    stacking = StackingEnsemble()

    # Create dataset
    df, feature_names = stacking.create_dataset()

    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = stacking.scaler.fit_transform(X_train)
    X_test_scaled = stacking.scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train base models
    stacking.train_base_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Create stacking ensembles
    stacking.create_stacking_level1(X_train_scaled, X_test_scaled, y_train, y_test)
    stacking.create_stacking_level2(X_train_scaled, X_test_scaled, y_train, y_test)

    # Test different meta-models
    stacking.create_stacking_custom_meta(X_train_scaled, X_test_scaled, y_train, y_test)

    # Cross-validation analysis
    stacking.analyze_cross_validation(X_train_scaled, y_train)

    # Diversity analysis
    stacking.analyze_prediction_diversity(X_test_scaled, y_test)

    # Visualize
    stacking.visualize_results(X_test_scaled, y_test)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
