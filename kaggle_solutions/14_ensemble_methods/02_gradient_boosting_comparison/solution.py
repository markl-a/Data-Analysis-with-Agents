"""
Gradient Boosting Comparison - XGBoost vs LightGBM vs CatBoost
Comprehensive comparison of modern boosting algorithms

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐⭐ Expert
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
import time
import warnings
warnings.filterwarnings('ignore')

# Import boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GradientBoostingComparison:
    """Compare different gradient boosting implementations"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_times = {}

    def create_dataset(self):
        """Create synthetic classification dataset"""
        print("Creating synthetic dataset...")

        # Create complex dataset
        X, y = make_classification(
            n_samples=5000,
            n_features=30,
            n_informative=20,
            n_redundant=5,
            n_repeated=5,
            n_classes=2,
            n_clusters_per_class=3,
            weights=[0.6, 0.4],
            flip_y=0.03,
            class_sep=0.6,
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

    def train_sklearn_gb(self, X_train, X_test, y_train, y_test):
        """Train scikit-learn Gradient Boosting"""
        print("\n" + "="*60)
        print("Training Scikit-learn GradientBoostingClassifier")
        print("="*60)

        start_time = time.time()

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        self.models['sklearn_gb'] = model
        self.training_times['sklearn_gb'] = training_time
        self.results['sklearn_gb'] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        return model

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost"""
        if not XGBOOST_AVAILABLE:
            print("\nXGBoost not available - skipping")
            return None

        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)

        start_time = time.time()

        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            verbosity=0,
            use_label_encoder=False
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        training_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        self.models['xgboost'] = model
        self.training_times['xgboost'] = training_time
        self.results['xgboost'] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        return model

    def train_lightgbm(self, X_train, X_test, y_train, y_test):
        """Train LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            print("\nLightGBM not available - skipping")
            return None

        print("\n" + "="*60)
        print("Training LightGBM")
        print("="*60)

        start_time = time.time()

        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(10, verbose=False)])
        training_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        self.models['lightgbm'] = model
        self.training_times['lightgbm'] = training_time
        self.results['lightgbm'] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        return model

    def train_catboost(self, X_train, X_test, y_train, y_test):
        """Train CatBoost"""
        if not CATBOOST_AVAILABLE:
            print("\nCatBoost not available - skipping")
            return None

        print("\n" + "="*60)
        print("Training CatBoost")
        print("="*60)

        start_time = time.time()

        model = cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            random_state=42,
            verbose=False
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        training_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        self.models['catboost'] = model
        self.training_times['catboost'] = training_time
        self.results['catboost'] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        return model

    def compare_learning_curves(self, X_train, y_train):
        """Compare learning curves across models"""
        print("\n" + "="*60)
        print("Generating Learning Curves")
        print("="*60)

        n_estimators_range = range(10, 101, 10)
        learning_curves = {}

        # Sklearn GB
        print("Sklearn GB...")
        sklearn_scores = []
        for n_est in n_estimators_range:
            model = GradientBoostingClassifier(n_estimators=n_est, random_state=42, verbose=0)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            sklearn_scores.append(scores.mean())
        learning_curves['sklearn_gb'] = sklearn_scores

        # XGBoost
        if XGBOOST_AVAILABLE:
            print("XGBoost...")
            xgb_scores = []
            for n_est in n_estimators_range:
                model = xgb.XGBClassifier(n_estimators=n_est, random_state=42, verbosity=0, use_label_encoder=False)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                xgb_scores.append(scores.mean())
            learning_curves['xgboost'] = xgb_scores

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("LightGBM...")
            lgb_scores = []
            for n_est in n_estimators_range:
                model = lgb.LGBMClassifier(n_estimators=n_est, random_state=42, verbose=-1)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                lgb_scores.append(scores.mean())
            learning_curves['lightgbm'] = lgb_scores

        # CatBoost
        if CATBOOST_AVAILABLE:
            print("CatBoost...")
            cb_scores = []
            for n_est in n_estimators_range:
                model = cb.CatBoostClassifier(iterations=n_est, random_state=42, verbose=False)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                cb_scores.append(scores.mean())
            learning_curves['catboost'] = cb_scores

        self.results['learning_curves'] = {
            'n_estimators': list(n_estimators_range),
            'curves': learning_curves
        }

        return learning_curves

    def visualize_comparison(self):
        """Create comprehensive comparison visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # Get available models
        available_models = [m for m in ['sklearn_gb', 'xgboost', 'lightgbm', 'catboost']
                          if m in self.results]

        # 1. Accuracy Comparison
        ax1 = plt.subplot(3, 3, 1)
        accuracies = [self.results[m]['accuracy'] for m in available_models]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(available_models)]
        bars = ax1.bar(range(len(available_models)), accuracies, color=colors)
        ax1.set_xticks(range(len(available_models)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in available_models], rotation=45, ha='right')
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylim([min(accuracies) - 0.02, 1.0])
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. ROC-AUC Comparison
        ax2 = plt.subplot(3, 3, 2)
        roc_aucs = [self.results[m]['roc_auc'] for m in available_models]
        bars = ax2.bar(range(len(available_models)), roc_aucs, color=colors)
        ax2.set_xticks(range(len(available_models)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in available_models], rotation=45, ha='right')
        ax2.set_ylabel('ROC-AUC', fontsize=10)
        ax2.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylim([min(roc_aucs) - 0.02, 1.0])
        for i, (bar, auc) in enumerate(zip(bars, roc_aucs)):
            ax2.text(i, auc + 0.005, f'{auc:.4f}', ha='center', va='bottom')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Log Loss Comparison (lower is better)
        ax3 = plt.subplot(3, 3, 3)
        log_losses = [self.results[m]['log_loss'] for m in available_models]
        bars = ax3.bar(range(len(available_models)), log_losses, color=colors)
        ax3.set_xticks(range(len(available_models)))
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in available_models], rotation=45, ha='right')
        ax3.set_ylabel('Log Loss', fontsize=10)
        ax3.set_title('Log Loss Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        for i, (bar, loss) in enumerate(zip(bars, log_losses)):
            ax3.text(i, loss + 0.01, f'{loss:.4f}', ha='center', va='bottom')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Training Time Comparison
        ax4 = plt.subplot(3, 3, 4)
        times = [self.training_times[m] for m in available_models]
        bars = ax4.bar(range(len(available_models)), times, color=colors)
        ax4.set_xticks(range(len(available_models)))
        ax4.set_xticklabels([m.replace('_', ' ').title() for m in available_models], rotation=45, ha='right')
        ax4.set_ylabel('Time (seconds)', fontsize=10)
        ax4.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        for i, (bar, t) in enumerate(zip(bars, times)):
            ax4.text(i, t + 0.02, f'{t:.2f}s', ha='center', va='bottom')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Learning Curves
        if 'learning_curves' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            lc = self.results['learning_curves']
            for i, (model_name, scores) in enumerate(lc['curves'].items()):
                ax5.plot(lc['n_estimators'], scores, 'o-', linewidth=2,
                        label=model_name.replace('_', ' ').title(), color=colors[i])
            ax5.set_xlabel('Number of Estimators', fontsize=10)
            ax5.set_ylabel('Cross-Validation Accuracy', fontsize=10)
            ax5.set_title('Learning Curves', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Efficiency Plot (Accuracy vs Time)
        ax6 = plt.subplot(3, 3, 6)
        for i, model in enumerate(available_models):
            ax6.scatter(self.training_times[model], self.results[model]['accuracy'],
                       s=200, color=colors[i], alpha=0.7, edgecolors='black', linewidth=2,
                       label=model.replace('_', ' ').title())
        ax6.set_xlabel('Training Time (seconds)', fontsize=10)
        ax6.set_ylabel('Accuracy', fontsize=10)
        ax6.set_title('Efficiency: Accuracy vs Training Time', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Performance Radar Chart
        ax7 = plt.subplot(3, 3, 7, projection='polar')
        categories = ['Accuracy', 'ROC-AUC', 'Speed', 'Log Loss\n(inverted)']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for i, model in enumerate(available_models):
            # Normalize metrics to 0-1 scale
            acc = self.results[model]['accuracy']
            roc = self.results[model]['roc_auc']
            speed = 1 - (self.training_times[model] / max(times))  # Invert so higher is better
            logloss_inv = 1 - (self.results[model]['log_loss'] / max(log_losses))  # Invert

            values = [acc, roc, speed, logloss_inv]
            values += values[:1]

            ax7.plot(angles, values, 'o-', linewidth=2, label=model.replace('_', ' ').title())
            ax7.fill(angles, values, alpha=0.15)

        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(categories)
        ax7.set_ylim(0, 1)
        ax7.set_title('Overall Performance Comparison', fontsize=12, fontweight='bold', pad=20)
        ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax7.grid(True)

        # 8. Feature Importance Comparison (if available)
        ax8 = plt.subplot(3, 3, 8)
        feature_importances = {}
        for model_name in available_models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                feature_importances[model_name] = model.feature_importances_[:10]  # Top 10

        if feature_importances:
            x = np.arange(10)
            width = 0.2
            for i, (model_name, imp) in enumerate(feature_importances.items()):
                ax8.bar(x + i * width, imp, width, label=model_name.replace('_', ' ').title())

            ax8.set_xlabel('Feature Index', fontsize=10)
            ax8.set_ylabel('Importance', fontsize=10)
            ax8.set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold')
            ax8.set_xticks(x + width * 1.5)
            ax8.set_xticklabels([f'F{i}' for i in range(10)])
            ax8.legend()
            ax8.grid(True, alpha=0.3, axis='y')
        else:
            ax8.text(0.5, 0.5, 'Feature importances\nnot available',
                    ha='center', va='center', transform=ax8.transAxes, fontsize=12)
            ax8.axis('off')

        # 9. Summary Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = "Gradient Boosting Comparison\n" + "="*40 + "\n\n"
        summary_text += f"{'Model':<15} {'Acc':<8} {'AUC':<8} {'Time':<8}\n"
        summary_text += "-"*40 + "\n"

        for model in available_models:
            summary_text += f"{model.replace('_', ' ').title():<15} "
            summary_text += f"{self.results[model]['accuracy']:.4f}  "
            summary_text += f"{self.results[model]['roc_auc']:.4f}  "
            summary_text += f"{self.training_times[model]:.2f}s\n"

        summary_text += "\n" + "="*40 + "\n"
        summary_text += "Key Findings:\n"
        summary_text += f"• Best Accuracy: {max(accuracies):.4f}\n"
        summary_text += f"• Best ROC-AUC: {max(roc_aucs):.4f}\n"
        summary_text += f"• Fastest: {min(times):.2f}s\n"

        ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/gradient_boosting_comparison.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/gradient_boosting_comparison.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Gradient Boosting Algorithms Comparison")
    print("="*60)

    # Initialize
    comparison = GradientBoostingComparison()

    # Create dataset
    df, feature_names = comparison.create_dataset()

    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train all models
    comparison.train_sklearn_gb(X_train, X_test, y_train, y_test)
    comparison.train_xgboost(X_train, X_test, y_train, y_test)
    comparison.train_lightgbm(X_train, X_test, y_train, y_test)
    comparison.train_catboost(X_train, X_test, y_train, y_test)

    # Compare learning curves
    comparison.compare_learning_curves(X_train, y_train)

    # Visualize
    comparison.visualize_comparison()

    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
