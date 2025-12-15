"""
Drug Discovery - Molecular Activity Prediction

Predict compound biological activity using molecular descriptors
and machine learning to accelerate drug discovery.

Dataset: https://www.kaggle.com/c/lish-moa
Difficulty: ⭐⭐⭐⭐ Expert Level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    log_loss, roc_auc_score, f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class DrugDiscoveryModel:
    """Drug Discovery Multi-Label Classification Model."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        self.models: Dict[str, Any] = {}
        self.target_names: List[str] = []

    def create_sample_data(self, n_samples: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic drug discovery dataset."""
        np.random.seed(42)

        # Gene expression features (g-features)
        n_genes = 100
        gene_features = np.random.randn(n_samples, n_genes)

        # Add some structure/correlation
        for i in range(0, n_genes, 10):
            base = np.random.randn(n_samples)
            for j in range(min(10, n_genes - i)):
                gene_features[:, i + j] += base * 0.5

        gene_cols = [f'g-{i}' for i in range(n_genes)]

        # Cell viability features (c-features)
        n_cell = 20
        cell_features = np.random.randn(n_samples, n_cell) * 0.5
        cell_cols = [f'c-{i}' for i in range(n_cell)]

        # Treatment metadata
        cp_type = np.random.choice(['trt_cp', 'ctl_vehicle'], n_samples, p=[0.95, 0.05])
        cp_dose = np.random.choice(['D1', 'D2'], n_samples)
        cp_time = np.random.choice([24, 48, 72], n_samples)

        # Create feature DataFrame
        features = pd.DataFrame(gene_features, columns=gene_cols)
        for i, col in enumerate(cell_cols):
            features[col] = cell_features[:, i]
        features['cp_type'] = cp_type
        features['cp_dose'] = cp_dose
        features['cp_time'] = cp_time

        # Create multi-label targets (MoA - Mechanism of Action)
        moa_names = [
            'nfkb_inhibitor', 'proteasome_inhibitor', 'cyclooxygenase_inhibitor',
            'dopamine_receptor_agonist', 'serotonin_receptor_agonist',
            'histamine_receptor_antagonist', 'acetylcholine_receptor_agonist',
            'gaba_receptor_agonist', 'glutamate_receptor_antagonist',
            'calcium_channel_blocker'
        ]
        self.target_names = moa_names

        targets = pd.DataFrame(columns=moa_names)

        for moa in moa_names:
            # Generate labels based on feature patterns
            # Create some structure in the data
            signal = np.zeros(n_samples)
            for j in range(5):
                idx = np.random.randint(0, n_genes)
                signal += gene_features[:, idx] * np.random.uniform(0.3, 0.7)

            # Add noise and threshold
            prob = 1 / (1 + np.exp(-signal))
            prob = np.clip(prob, 0.01, 0.3)  # Keep class imbalance

            # Control vehicles have no activity
            prob[cp_type == 'ctl_vehicle'] = 0

            targets[moa] = (np.random.random(n_samples) < prob).astype(int)

        return features, targets

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Preprocess features for modeling."""
        # Select numeric columns
        numeric_cols = [col for col in df.columns
                       if col.startswith('g-') or col.startswith('c-')]

        # Encode categorical
        df_processed = df.copy()

        # One-hot encode categoricals
        if 'cp_type' in df.columns:
            df_processed['cp_type_encoded'] = (df['cp_type'] == 'trt_cp').astype(int)
        if 'cp_dose' in df.columns:
            df_processed['cp_dose_encoded'] = (df['cp_dose'] == 'D2').astype(int)
        if 'cp_time' in df.columns:
            df_processed['cp_time_24'] = (df['cp_time'] == 24).astype(int)
            df_processed['cp_time_48'] = (df['cp_time'] == 48).astype(int)
            df_processed['cp_time_72'] = (df['cp_time'] == 72).astype(int)

        # Get all feature columns
        feature_cols = numeric_cols + ['cp_type_encoded', 'cp_dose_encoded',
                                       'cp_time_24', 'cp_time_48', 'cp_time_72']
        feature_cols = [c for c in feature_cols if c in df_processed.columns]

        X = df_processed[feature_cols].values

        if fit:
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled[:, :100])  # PCA on gene features
            X_final = np.hstack([X_pca, X_scaled[:, 100:]])
        else:
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled[:, :100])
            X_final = np.hstack([X_pca, X_scaled[:, 100:]])

        return X_final

    def analyze_data(self, features: pd.DataFrame, targets: pd.DataFrame,
                     output_dir: str = '.') -> None:
        """Perform exploratory data analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Drug Discovery Dataset Analysis', fontsize=16)

        # Target distribution
        target_counts = targets.sum().sort_values(ascending=True)
        target_counts.plot(kind='barh', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('MoA Target Distribution')
        axes[0, 0].set_xlabel('Count')

        # Multi-label distribution
        label_counts = targets.sum(axis=1)
        axes[0, 1].hist(label_counts, bins=range(0, 6), edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Labels per Sample')
        axes[0, 1].set_xlabel('Number of MoA Labels')
        axes[0, 1].set_ylabel('Count')

        # Treatment type distribution
        features['cp_type'].value_counts().plot(kind='pie', ax=axes[0, 2],
                                                 autopct='%1.1f%%')
        axes[0, 2].set_title('Treatment Type Distribution')

        # Gene expression heatmap (sample)
        gene_cols = [col for col in features.columns if col.startswith('g-')][:20]
        sample_genes = features[gene_cols].iloc[:50]
        sns.heatmap(sample_genes.T, cmap='RdBu_r', center=0, ax=axes[1, 0],
                   cbar_kws={'label': 'Expression'})
        axes[1, 0].set_title('Gene Expression Heatmap (Sample)')

        # PCA variance explained
        X_scaled = self.scaler.fit_transform(features[gene_cols[:100]].values)
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        axes[1, 1].plot(cumsum, 'b-', linewidth=2)
        axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        axes[1, 1].set_title('PCA Variance Explained')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Target correlation
        target_corr = targets.corr()
        sns.heatmap(target_corr, cmap='coolwarm', center=0, ax=axes[1, 2],
                   annot=False, square=True)
        axes[1, 2].set_title('Target Correlation')

        # Dose effect
        dose_effect = features.groupby('cp_dose')['g-0'].mean()
        dose_effect.plot(kind='bar', ax=axes[2, 0], color=['lightblue', 'steelblue'])
        axes[2, 0].set_title('Gene Expression by Dose')
        axes[2, 0].set_ylabel('Mean Expression (g-0)')

        # Time effect
        time_effect = features.groupby('cp_time')['g-1'].mean()
        time_effect.plot(kind='bar', ax=axes[2, 1], color='green', alpha=0.7)
        axes[2, 1].set_title('Gene Expression by Time')
        axes[2, 1].set_ylabel('Mean Expression (g-1)')

        # Feature importance preview
        axes[2, 2].text(0.5, 0.5, 'Feature Importance\n(After Training)',
                       ha='center', va='center', fontsize=14)
        axes[2, 2].set_title('Feature Importance')
        axes[2, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/drug_discovery_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/drug_discovery_analysis.png")
        plt.close()

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train multi-label classification models."""
        print("\nTraining models...")

        # Logistic Regression (fast baseline)
        lr_base = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        self.models['Logistic Regression'] = MultiOutputClassifier(lr_base)
        self.models['Logistic Regression'].fit(X_train, y_train)
        print("  - Logistic Regression trained")

        # Random Forest
        rf_base = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'] = MultiOutputClassifier(rf_base)
        self.models['Random Forest'].fit(X_train, y_train)
        print("  - Random Forest trained")

        # Gradient Boosting (simplified)
        gb_base = GradientBoostingClassifier(
            n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.models['Gradient Boosting'] = MultiOutputClassifier(gb_base)
        self.models['Gradient Boosting'].fit(X_train, y_train)
        print("  - Gradient Boosting trained")

        print(f"\nTrained {len(self.models)} models!")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models."""
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = np.array([est.predict_proba(X_test)[:, 1]
                                     for est in model.estimators_]).T

            # Calculate metrics
            # Log loss (per target, then average)
            logloss_per_target = []
            auc_per_target = []
            for i in range(y_test.shape[1]):
                if y_test[:, i].sum() > 0 and y_test[:, i].sum() < len(y_test):
                    logloss_per_target.append(log_loss(y_test[:, i], y_pred_proba[:, i]))
                    auc_per_target.append(roc_auc_score(y_test[:, i], y_pred_proba[:, i]))

            results.append({
                'Model': name,
                'Log Loss': np.mean(logloss_per_target),
                'Macro AUC': np.mean(auc_per_target),
                'Macro F1': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'Micro F1': f1_score(y_test, y_pred, average='micro', zero_division=0)
            })

        return pd.DataFrame(results).sort_values('Log Loss')

    def per_target_analysis(self, X_test: np.ndarray, y_test: np.ndarray,
                           output_dir: str = '.') -> pd.DataFrame:
        """Analyze performance per target."""
        best_model = self.models['Random Forest']
        y_pred_proba = np.array([est.predict_proba(X_test)[:, 1]
                                for est in best_model.estimators_]).T

        per_target = []
        for i, target in enumerate(self.target_names):
            if y_test[:, i].sum() > 0 and y_test[:, i].sum() < len(y_test):
                auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
                per_target.append({
                    'Target': target,
                    'Positive Rate': y_test[:, i].mean(),
                    'AUC-ROC': auc
                })

        return pd.DataFrame(per_target).sort_values('AUC-ROC', ascending=False)

    def plot_results(self, results: pd.DataFrame, per_target: pd.DataFrame,
                    output_dir: str = '.') -> None:
        """Visualize results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Drug Discovery Model Results', fontsize=16)

        # Model comparison - Log Loss
        colors = ['steelblue', 'green', 'orange']
        results.set_index('Model')['Log Loss'].plot(
            kind='bar', ax=axes[0, 0], color=colors
        )
        axes[0, 0].set_title('Model Comparison - Log Loss (Lower is Better)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('Log Loss')

        # Model comparison - AUC
        results.set_index('Model')['Macro AUC'].plot(
            kind='bar', ax=axes[0, 1], color=colors
        )
        axes[0, 1].set_title('Model Comparison - Macro AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].set_ylim(0, 1)

        # Per-target AUC
        per_target.set_index('Target')['AUC-ROC'].plot(
            kind='barh', ax=axes[1, 0], color='steelblue'
        )
        axes[1, 0].set_title('Per-Target AUC-ROC')
        axes[1, 0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlim(0, 1)

        # F1 scores
        f1_data = results.set_index('Model')[['Macro F1', 'Micro F1']]
        f1_data.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('F1 Scores by Model')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/drug_discovery_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/drug_discovery_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("DRUG DISCOVERY - MOLECULAR ACTIVITY PREDICTION")
    print("=" * 70)

    model = DrugDiscoveryModel()

    # Create data
    print("\nCreating synthetic dataset...")
    features, targets = model.create_sample_data(n_samples=2000)
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Number of MoA targets: {len(model.target_names)}")

    # Analysis
    model.analyze_data(features, targets)

    # Preprocess
    X = model.preprocess_features(features, fit=True)
    y = targets.values
    print(f"\nProcessed features shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train
    model.train_models(X_train, y_train)

    # Evaluate
    results = model.evaluate_models(X_test, y_test)
    per_target = model.per_target_analysis(X_test, y_test)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    print("\n" + "=" * 70)
    print("PER-TARGET PERFORMANCE (Top 5)")
    print("=" * 70)
    print(per_target.head().to_string(index=False))

    # Visualize
    model.plot_results(results, per_target)

    print("\n" + "=" * 70)
    best = results.iloc[0]
    print(f"Best Model: {best['Model']}")
    print(f"Best Log Loss: {best['Log Loss']:.4f}")
    print(f"Best Macro AUC: {best['Macro AUC']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
