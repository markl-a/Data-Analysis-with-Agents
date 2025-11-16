"""
Blending Models - Holdout-Based Ensemble
Compare blending with stacking and simple averaging

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BlendingEnsemble:
    """Blending Ensemble Implementation and Analysis"""

    def __init__(self):
        self.base_models = {}
        self.blending_models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def create_dataset(self):
        """Create synthetic classification dataset"""
        print("Creating synthetic dataset...")

        X, y = make_classification(
            n_samples=4000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_repeated=2,
            n_classes=2,
            n_clusters_per_class=2,
            weights=[0.6, 0.4],
            flip_y=0.05,
            class_sep=0.8,
            random_state=42
        )

        feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['target'].value_counts()}")

        return df, feature_names

    def split_data_for_blending(self, X, y):
        """Split data into train, blend, and test sets"""
        print("\n" + "="*60)
        print("Splitting Data for Blending")
        print("="*60)

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Second split: separate train and blend sets
        X_train, X_blend, y_train, y_blend = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )

        print(f"Train set: {X_train.shape}")
        print(f"Blend set: {X_blend.shape}")
        print(f"Test set: {X_test.shape}")

        return X_train, X_blend, X_test, y_train, y_blend, y_test

    def train_base_models(self, X_train, y_train):
        """Train base models on training set"""
        print("\n" + "="*60)
        print("Training Base Models")
        print("="*60)

        # Define base models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7)
        }

        # Train each base model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.base_models[name] = model

        return models

    def generate_blend_features(self, X_blend, X_test):
        """Generate meta-features from base model predictions"""
        print("\n" + "="*60)
        print("Generating Blend Features")
        print("="*60)

        # Get predictions on blend set
        blend_features = np.zeros((X_blend.shape[0], len(self.base_models)))
        test_features = np.zeros((X_test.shape[0], len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Getting predictions from {name}...")

            # Predict on blend set
            blend_pred = model.predict_proba(X_blend)[:, 1]
            blend_features[:, i] = blend_pred

            # Predict on test set
            test_pred = model.predict_proba(X_test)[:, 1]
            test_features[:, i] = test_pred

        print(f"Blend features shape: {blend_features.shape}")
        print(f"Test features shape: {test_features.shape}")

        return blend_features, test_features

    def simple_averaging(self, X_test, y_test):
        """Simple average of base model predictions"""
        print("\n" + "="*60)
        print("Simple Averaging Ensemble")
        print("="*60)

        predictions = []
        probabilities = []

        for name, model in self.base_models.items():
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
            probabilities.append(prob)

        # Average predictions
        avg_prob = np.mean(probabilities, axis=0)
        avg_pred = (avg_prob >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, avg_pred)
        logloss = log_loss(y_test, avg_prob)

        print(f"Average Ensemble Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        self.results['Simple Average'] = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'predictions': avg_pred,
            'probabilities': avg_prob
        }

        return avg_pred, avg_prob

    def weighted_averaging(self, X_blend, y_blend, X_test, y_test):
        """Weighted average based on blend set performance"""
        print("\n" + "="*60)
        print("Weighted Averaging Ensemble")
        print("="*60)

        # Calculate weights based on blend set accuracy
        weights = []
        for name, model in self.base_models.items():
            pred = model.predict(X_blend)
            acc = accuracy_score(y_blend, pred)
            weights.append(acc)
            print(f"{name} blend accuracy: {acc:.4f}")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        print("\nNormalized weights:")
        for name, w in zip(self.base_models.keys(), weights):
            print(f"  {name}: {w:.4f}")

        # Weighted average predictions
        probabilities = []
        for name, model in self.base_models.items():
            prob = model.predict_proba(X_test)[:, 1]
            probabilities.append(prob)

        weighted_prob = np.average(probabilities, axis=0, weights=weights)
        weighted_pred = (weighted_prob >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, weighted_pred)
        logloss = log_loss(y_test, weighted_prob)

        print(f"\nWeighted Average Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        self.results['Weighted Average'] = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'predictions': weighted_pred,
            'probabilities': weighted_prob,
            'weights': weights
        }

        return weighted_pred, weighted_prob

    def train_blending_model(self, blend_features, y_blend, test_features, y_test):
        """Train blending meta-model on blend set"""
        print("\n" + "="*60)
        print("Training Blending Meta-Model")
        print("="*60)

        # Train logistic regression on blend features
        blender = LogisticRegression(random_state=42, max_iter=1000)
        blender.fit(blend_features, y_blend)

        print("Blender coefficients:")
        for name, coef in zip(self.base_models.keys(), blender.coef_[0]):
            print(f"  {name}: {coef:.4f}")

        # Predict on test set
        blend_pred = blender.predict(test_features)
        blend_prob = blender.predict_proba(test_features)[:, 1]

        accuracy = accuracy_score(y_test, blend_pred)
        logloss = log_loss(y_test, blend_prob)

        print(f"\nBlending Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        self.blending_models['blender'] = blender
        self.results['Blending'] = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'predictions': blend_pred,
            'probabilities': blend_prob,
            'coefficients': blender.coef_[0]
        }

        return blender

    def compare_individual_models(self, X_test, y_test):
        """Evaluate individual base models"""
        print("\n" + "="*60)
        print("Individual Base Model Performance")
        print("="*60)

        for name, model in self.base_models.items():
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, pred)
            logloss = log_loss(y_test, prob)

            self.results[name] = {
                'accuracy': accuracy,
                'log_loss': logloss,
                'predictions': pred,
                'probabilities': prob
            }

            print(f"{name}: Accuracy={accuracy:.4f}, LogLoss={logloss:.4f}")

    def visualize_results(self, y_test):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # 1. Accuracy Comparison
        ax1 = plt.subplot(3, 3, 1)
        all_models = list(self.base_models.keys()) + ['Simple Average', 'Weighted Average', 'Blending']
        accuracies = [self.results[m]['accuracy'] for m in all_models]

        colors = ['#1f77b4']*5 + ['#ff7f0e', '#2ca02c', '#d62728']
        bars = ax1.barh(range(len(all_models)), accuracies, color=colors)
        ax1.set_yticks(range(len(all_models)))
        ax1.set_yticklabels(all_models)
        ax1.set_xlabel('Accuracy', fontsize=10)
        ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlim([min(accuracies) - 0.05, 1.0])
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(acc + 0.005, i, f'{acc:.4f}', va='center')
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Log Loss Comparison (lower is better)
        ax2 = plt.subplot(3, 3, 2)
        log_losses = [self.results[m]['log_loss'] for m in all_models]

        bars = ax2.bar(range(len(all_models)), log_losses, color=colors)
        ax2.set_xticks(range(len(all_models)))
        ax2.set_xticklabels(all_models, rotation=45, ha='right')
        ax2.set_ylabel('Log Loss', fontsize=10)
        ax2.set_title('Log Loss Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        for i, (bar, loss) in enumerate(zip(bars, log_losses)):
            ax2.text(i, loss + 0.01, f'{loss:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Ensemble Methods Comparison
        ax3 = plt.subplot(3, 3, 3)
        ensemble_methods = ['Simple Average', 'Weighted Average', 'Blending']
        ensemble_accs = [self.results[m]['accuracy'] for m in ensemble_methods]

        colors3 = ['#ff7f0e', '#2ca02c', '#d62728']
        bars = ax3.bar(range(len(ensemble_methods)), ensemble_accs, color=colors3)
        ax3.set_xticks(range(len(ensemble_methods)))
        ax3.set_xticklabels(ensemble_methods, rotation=45, ha='right')
        ax3.set_ylabel('Accuracy', fontsize=10)
        ax3.set_title('Ensemble Methods Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylim([min(ensemble_accs) - 0.01, max(ensemble_accs) + 0.02])
        for i, (bar, acc) in enumerate(zip(bars, ensemble_accs)):
            ax3.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Weighted Average - Weights Visualization
        if 'Weighted Average' in self.results:
            ax4 = plt.subplot(3, 3, 4)
            weights = self.results['Weighted Average']['weights']
            model_names = list(self.base_models.keys())

            bars = ax4.barh(range(len(model_names)), weights, color='skyblue', edgecolor='black')
            ax4.set_yticks(range(len(model_names)))
            ax4.set_yticklabels(model_names)
            ax4.set_xlabel('Weight', fontsize=10)
            ax4.set_title('Weighted Average - Model Weights', fontsize=12, fontweight='bold')
            for i, (bar, w) in enumerate(zip(bars, weights)):
                ax4.text(w + 0.01, i, f'{w:.3f}', va='center')
            ax4.grid(True, alpha=0.3, axis='x')

        # 5. Blending Coefficients
        if 'Blending' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            coefficients = self.results['Blending']['coefficients']
            model_names = list(self.base_models.keys())

            colors5 = ['green' if c > 0 else 'red' for c in coefficients]
            bars = ax5.barh(range(len(model_names)), coefficients, color=colors5, edgecolor='black')
            ax5.set_yticks(range(len(model_names)))
            ax5.set_yticklabels(model_names)
            ax5.set_xlabel('Coefficient', fontsize=10)
            ax5.set_title('Blending Model Coefficients', fontsize=12, fontweight='bold')
            ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            for i, (bar, c) in enumerate(zip(bars, coefficients)):
                ax5.text(c + 0.1 if c > 0 else c - 0.1, i, f'{c:.3f}',
                        va='center', ha='left' if c > 0 else 'right')
            ax5.grid(True, alpha=0.3, axis='x')

        # 6. Confusion Matrix - Blending
        ax6 = plt.subplot(3, 3, 6)
        cm = confusion_matrix(y_test, self.results['Blending']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6)
        ax6.set_xlabel('Predicted', fontsize=10)
        ax6.set_ylabel('Actual', fontsize=10)
        ax6.set_title('Confusion Matrix - Blending', fontsize=12, fontweight='bold')

        # 7. Improvement Over Best Base Model
        ax7 = plt.subplot(3, 3, 7)
        base_accuracies = [self.results[m]['accuracy'] for m in self.base_models.keys()]
        best_base_acc = max(base_accuracies)

        improvements = [
            (self.results['Simple Average']['accuracy'] - best_base_acc) * 100,
            (self.results['Weighted Average']['accuracy'] - best_base_acc) * 100,
            (self.results['Blending']['accuracy'] - best_base_acc) * 100
        ]

        colors7 = ['#ff7f0e', '#2ca02c', '#d62728']
        bars = ax7.bar(range(3), improvements, color=colors7)
        ax7.set_xticks(range(3))
        ax7.set_xticklabels(ensemble_methods, rotation=45, ha='right')
        ax7.set_ylabel('Improvement (%)', fontsize=10)
        ax7.set_title('Improvement Over Best Base Model', fontsize=12, fontweight='bold')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax7.text(i, imp + 0.05 if imp > 0 else imp - 0.05,
                    f'{imp:.2f}%', ha='center',
                    va='bottom' if imp > 0 else 'top')
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. Data Split Visualization
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        split_diagram = """
        Blending Data Split Strategy

        Original Data (4000 samples)
               ↓
        ┌──────┴──────────────────┐
        │                         │
    Test (20%)          Temp (80%)
    800 samples              ↓
                    ┌────────┴────────┐
                    │                 │
               Train (60%)      Blend (20%)
              2400 samples     800 samples

        Process:
        1. Train base models on Train set
        2. Generate predictions on Blend set
        3. Train meta-model on Blend predictions
        4. Evaluate on Test set
        """
        ax8.text(0.1, 0.95, split_diagram, transform=ax8.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        best_base = max(base_accuracies)
        summary_text = f"""
        Blending Ensemble Summary
        {'='*40}

        Best Base Model: {best_base:.4f}

        Ensemble Methods:
        • Simple Average:   {self.results['Simple Average']['accuracy']:.4f}
        • Weighted Average: {self.results['Weighted Average']['accuracy']:.4f}
        • Blending:         {self.results['Blending']['accuracy']:.4f}

        Improvements:
        • Simple:   +{improvements[0]:.2f}%
        • Weighted: +{improvements[1]:.2f}%
        • Blending: +{improvements[2]:.2f}%

        Key Insight:
        Blending uses a holdout set for
        meta-model training, avoiding
        overfitting while achieving strong
        ensemble performance.
        """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/blending_models.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/blending_models.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Blending Models Analysis")
    print("="*60)

    # Initialize
    blending = BlendingEnsemble()

    # Create dataset
    df, feature_names = blending.create_dataset()

    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']

    # Split for blending (train, blend, test)
    X_train, X_blend, X_test, y_train, y_blend, y_test = blending.split_data_for_blending(X, y)

    # Scale features
    X_train_scaled = blending.scaler.fit_transform(X_train)
    X_blend_scaled = blending.scaler.transform(X_blend)
    X_test_scaled = blending.scaler.transform(X_test)

    # Train base models
    blending.train_base_models(X_train_scaled, y_train)

    # Evaluate individual models
    blending.compare_individual_models(X_test_scaled, y_test)

    # Generate blend features
    blend_features, test_features = blending.generate_blend_features(X_blend_scaled, X_test_scaled)

    # Simple averaging
    blending.simple_averaging(X_test_scaled, y_test)

    # Weighted averaging
    blending.weighted_averaging(X_blend_scaled, y_blend, X_test_scaled, y_test)

    # Train blending model
    blending.train_blending_model(blend_features, y_blend, test_features, y_test)

    # Visualize
    blending.visualize_results(y_test)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
