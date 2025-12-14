"""
Credit Risk Assessment - Financial ML Solution

This module predicts loan default probability based on applicant
financial and demographic data.

Dataset: https://www.kaggle.com/c/home-credit-default-risk
Difficulty: ⭐⭐⭐ Advanced Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class CreditRiskModel:
    """Credit Risk Assessment Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names: List[str] = []

    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic credit application dataset."""
        np.random.seed(42)
        n_samples = 10000

        # Age (days before current day, converted to years later)
        days_birth = np.random.uniform(-25000, -7000, n_samples).astype(int)
        age_years = -days_birth / 365

        # Employment days (negative = employed, positive = unemployed)
        days_employed = np.random.choice(
            [np.random.uniform(-15000, -30, 1)[0], 365243],  # Employed or pensioner/unemployed
            n_samples, p=[0.7, 0.3]
        ).astype(int)

        # Income
        amt_income_total = np.random.lognormal(11, 0.5, n_samples).clip(25000, 2000000)

        # Credit amount
        amt_credit = amt_income_total * np.random.uniform(1, 10, n_samples)

        # Annuity
        amt_annuity = amt_credit / np.random.uniform(12, 60, n_samples)

        # External scores (most important features)
        ext_source_1 = np.random.beta(2, 5, n_samples)
        ext_source_2 = np.random.beta(3, 2, n_samples)
        ext_source_3 = np.random.beta(2, 3, n_samples)

        # Goods price
        amt_goods_price = amt_credit * np.random.uniform(0.8, 1.0, n_samples)

        # Region and city ratings
        region_rating = np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.5, 0.2])

        # Document flags
        flag_document_3 = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])

        # Own car and realty
        flag_own_car = np.random.choice([0, 1], n_samples, p=[0.34, 0.66])
        flag_own_realty = np.random.choice([0, 1], n_samples, p=[0.31, 0.69])

        # Gender
        code_gender = np.random.choice(['M', 'F'], n_samples, p=[0.34, 0.66])

        # Calculate default probability based on features
        default_prob = (
            0.08 +  # Base rate
            0.15 * (1 - ext_source_1) +
            0.20 * (1 - ext_source_2) +
            0.15 * (1 - ext_source_3) +
            0.05 * (age_years < 30) +
            0.03 * (days_employed == 365243) +
            0.05 * (amt_credit / amt_income_total > 5) -
            0.02 * flag_own_realty -
            0.01 * flag_own_car
        )
        target = (np.random.random(n_samples) < default_prob.clip(0.02, 0.5)).astype(int)

        return pd.DataFrame({
            'SK_ID_CURR': range(100000, 100000 + n_samples),
            'TARGET': target,
            'AMT_INCOME_TOTAL': amt_income_total.round(2),
            'AMT_CREDIT': amt_credit.round(2),
            'AMT_ANNUITY': amt_annuity.round(2),
            'AMT_GOODS_PRICE': amt_goods_price.round(2),
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'EXT_SOURCE_1': ext_source_1.round(4),
            'EXT_SOURCE_2': ext_source_2.round(4),
            'EXT_SOURCE_3': ext_source_3.round(4),
            'REGION_RATING_CLIENT': region_rating,
            'FLAG_DOCUMENT_3': flag_document_3,
            'FLAG_OWN_CAR': flag_own_car,
            'FLAG_OWN_REALTY': flag_own_realty,
            'CODE_GENDER': code_gender
        })

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for credit risk."""
        df = df.copy()

        # Age in years
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

        # Employment years (handle pensioners)
        df['EMPLOYED_YEARS'] = np.where(
            df['DAYS_EMPLOYED'] == 365243, 0, -df['DAYS_EMPLOYED'] / 365
        )
        df['IS_EMPLOYED'] = (df['DAYS_EMPLOYED'] != 365243).astype(int)

        # Income to credit ratio
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

        # Annuity to income ratio
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

        # Credit to goods ratio
        df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)

        # External source average
        df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)

        # Fill missing external sources
        for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
            df[col].fillna(df[col].median(), inplace=True)

        # Encode gender
        df['CODE_GENDER_ENCODED'] = (df['CODE_GENDER'] == 'M').astype(int)

        # Age groups
        df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], bins=[0, 25, 35, 50, 65, 100],
                                 labels=[0, 1, 2, 3, 4]).astype(int)

        return df

    def plot_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate credit risk analysis visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Credit Risk Analysis', fontsize=16)

        # Default rate
        df['TARGET'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title(f'Default Distribution (Rate: {df["TARGET"].mean():.2%})')
        axes[0, 0].set_xticklabels(['No Default', 'Default'], rotation=0)

        # External sources importance
        for i, col in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
            df.boxplot(column=col, by='TARGET', ax=axes[0, 1] if i == 0 else None)
        axes[0, 1].set_title('External Source 2 by Default Status')
        df.boxplot(column='EXT_SOURCE_2', by='TARGET', ax=axes[0, 1])
        plt.suptitle('')

        # Age distribution by default
        df[df['TARGET'] == 0]['AGE_YEARS'].hist(bins=30, ax=axes[0, 2], alpha=0.5,
                                                 label='No Default', color='green')
        df[df['TARGET'] == 1]['AGE_YEARS'].hist(bins=30, ax=axes[0, 2], alpha=0.5,
                                                 label='Default', color='red')
        axes[0, 2].set_title('Age Distribution by Default')
        axes[0, 2].legend()

        # Credit income ratio
        df.boxplot(column='CREDIT_INCOME_RATIO', by='TARGET', ax=axes[1, 0])
        axes[1, 0].set_title('Credit/Income Ratio by Default')
        plt.suptitle('')

        # Default by gender
        gender_default = df.groupby('CODE_GENDER')['TARGET'].mean()
        gender_default.plot(kind='bar', ax=axes[1, 1], color=['pink', 'lightblue'])
        axes[1, 1].set_title('Default Rate by Gender')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

        # Employment status
        emp_default = df.groupby('IS_EMPLOYED')['TARGET'].mean()
        emp_default.plot(kind='bar', ax=axes[1, 2], color='purple')
        axes[1, 2].set_title('Default Rate by Employment')
        axes[1, 2].set_xticklabels(['Unemployed', 'Employed'], rotation=0)

        # Own assets effect
        assets = df.groupby(['FLAG_OWN_CAR', 'FLAG_OWN_REALTY'])['TARGET'].mean().unstack()
        assets.plot(kind='bar', ax=axes[2, 0])
        axes[2, 0].set_title('Default by Asset Ownership')
        axes[2, 0].legend(title='Own Realty')

        # Income distribution
        df['AMT_INCOME_TOTAL'].hist(bins=50, ax=axes[2, 1], color='steelblue', alpha=0.7)
        axes[2, 1].set_title('Income Distribution')
        axes[2, 1].set_xlabel('Income')

        # Feature correlations
        numeric_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                       'CREDIT_INCOME_RATIO', 'AGE_YEARS', 'TARGET']
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 2])
        axes[2, 2].set_title('Feature Correlations')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/credit_risk_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/credit_risk_analysis.png")
        plt.close()

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train credit risk models."""
        X_scaled = self.scaler.fit_transform(X_train)

        print("\nTraining models...")

        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.models['Logistic Regression'].fit(X_scaled, y_train)

        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'
        )
        self.models['Random Forest'].fit(X_scaled, y_train)

        if XGBOOST_AVAILABLE:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight,
                random_state=42, use_label_encoder=False, eval_metric='auc'
            )
            self.models['XGBoost'].fit(X_scaled, y_train)

        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, is_unbalance=True,
                random_state=42, verbose=-1
            )
            self.models['LightGBM'].fit(X_scaled, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate models with credit risk metrics."""
        X_scaled = self.scaler.transform(X_test)
        results = []

        print("\n=== Model Evaluation ===")

        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            gini = 2 * auc - 1

            # KS statistic
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ks = max(tpr - fpr)

            results.append({
                'Model': name,
                'AUC': auc,
                'Gini': gini,
                'KS': ks
            })

            print(f"{name}: AUC={auc:.4f}, Gini={gini:.4f}, KS={ks:.4f}")

        results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_results(self, results_df: pd.DataFrame, X_test: pd.DataFrame,
                    y_test: pd.Series, output_dir: str = '.') -> None:
        """Visualize model results."""
        X_scaled = self.scaler.transform(X_test)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ROC curves
        for name, model in self.models.items():
            y_proba = model.predict_proba(X_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_proba):.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].legend(loc='lower right')

        # Model comparison
        results_df.set_index('Model')[['AUC', 'Gini', 'KS']].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Model Metrics Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Score distribution
        y_proba_best = self.best_model.predict_proba(X_scaled)[:, 1]
        axes[1, 0].hist(y_proba_best[y_test == 0], bins=50, alpha=0.5, label='No Default', density=True)
        axes[1, 0].hist(y_proba_best[y_test == 1], bins=50, alpha=0.5, label='Default', density=True)
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[-10:]
            axes[1, 1].barh(range(10), importance[indices], color='steelblue')
            axes[1, 1].set_yticks(range(10))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_title('Top 10 Features')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/credit_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/credit_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("CREDIT RISK ASSESSMENT")
    print("=" * 70)

    model = CreditRiskModel()

    # Create and preprocess data
    df = model.create_sample_data()
    print(f"\nDataset: {df.shape}, Default rate: {df['TARGET'].mean():.2%}")

    df = model.preprocess_data(df)
    model.plot_analysis(df)

    # Prepare features
    feature_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                   'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                   'AGE_YEARS', 'EMPLOYED_YEARS', 'IS_EMPLOYED',
                   'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
                   'EXT_SOURCE_MEAN', 'CODE_GENDER_ENCODED',
                   'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AGE_GROUP']
    model.feature_names = feature_cols

    X = df[feature_cols]
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and evaluate
    model.train_models(X_train, y_train)
    results = model.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    model.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best AUC: {results.iloc[0]['AUC']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
