"""
Heart Disease Prediction - Comprehensive Machine Learning Solution

This module provides a complete solution for predicting heart disease risk based on
patient clinical data. It implements multiple ML algorithms with focus on medical
diagnosis requirements.

Dataset: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
Difficulty: ⭐⭐ Intermediate Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, f1_score, recall_score, precision_score
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


class HeartDiseasePredictor:
    """Heart Disease Risk Prediction Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names: List[str] = []

    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic heart disease dataset."""
        np.random.seed(42)
        n_samples = 900

        # Generate features with realistic distributions
        age = np.random.normal(55, 10, n_samples).clip(29, 77).astype(int)
        sex = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])

        # Chest pain type (0: typical angina, 1: atypical, 2: non-anginal, 3: asymptomatic)
        cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.2, 0.3, 0.4])

        # Resting blood pressure
        trestbps = np.random.normal(130, 18, n_samples).clip(94, 200).astype(int)

        # Serum cholesterol
        chol = np.random.normal(245, 52, n_samples).clip(126, 564).astype(int)

        # Fasting blood sugar > 120 mg/dl
        fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])

        # Resting ECG results
        restecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.45, 0.05])

        # Maximum heart rate achieved
        thalach = np.random.normal(150, 23, n_samples).clip(71, 202).astype(int)

        # Exercise induced angina
        exang = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])

        # ST depression induced by exercise
        oldpeak = np.random.exponential(1, n_samples).clip(0, 6.2).round(1)

        # Slope of peak exercise ST segment
        slope = np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.5, 0.4])

        # Number of major vessels colored by flouroscopy
        ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.55, 0.22, 0.13, 0.1])

        # Thalassemia
        thal = np.random.choice([1, 2, 3], n_samples, p=[0.05, 0.6, 0.35])

        # Generate target based on risk factors
        risk_score = (
            0.1 +
            0.2 * (cp == 0) +
            0.15 * (age > 55) +
            0.1 * (sex == 1) +
            0.15 * (oldpeak > 1.5) +
            0.1 * (ca > 0) +
            0.1 * (thal == 3) +
            0.1 * exang -
            0.1 * (thalach > 150)
        )
        target = (np.random.random(n_samples) < risk_score.clip(0.1, 0.9)).astype(int)

        return pd.DataFrame({
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
            'thal': thal, 'target': target
        })

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and engineer features."""
        df = df.copy()

        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100],
                                  labels=['young', 'middle', 'senior', 'elderly'])
        df['age_group_encoded'] = LabelEncoder().fit_transform(df['age_group'])

        # Blood pressure categories
        df['bp_category'] = pd.cut(df['trestbps'],
                                   bins=[0, 120, 130, 140, 300],
                                   labels=['normal', 'elevated', 'high_1', 'high_2'])
        df['bp_cat_encoded'] = LabelEncoder().fit_transform(df['bp_category'])

        # Cholesterol risk
        df['chol_risk'] = pd.cut(df['chol'],
                                 bins=[0, 200, 240, 1000],
                                 labels=['desirable', 'borderline', 'high'])
        df['chol_risk_encoded'] = LabelEncoder().fit_transform(df['chol_risk'])

        # Heart rate reserve (age-predicted max - achieved)
        df['hr_reserve'] = (220 - df['age']) - df['thalach']

        # Cardiovascular risk score
        df['cv_risk_score'] = (
            df['fbs'] +
            df['exang'] +
            (df['ca'] > 0).astype(int) +
            (df['oldpeak'] > 1).astype(int) +
            (df['trestbps'] > 140).astype(int)
        )

        return df

    def plot_exploratory_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate EDA visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Heart Disease Dataset - EDA', fontsize=16)

        # Target distribution
        df['target'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title('Heart Disease Distribution')
        axes[0, 0].set_xticklabels(['No Disease', 'Disease'], rotation=0)

        # Age distribution by target
        df[df['target'] == 0]['age'].hist(bins=20, ax=axes[0, 1], alpha=0.5, label='No Disease', color='green')
        df[df['target'] == 1]['age'].hist(bins=20, ax=axes[0, 1], alpha=0.5, label='Disease', color='red')
        axes[0, 1].set_title('Age Distribution by Target')
        axes[0, 1].legend()

        # Sex vs Target
        pd.crosstab(df['sex'], df['target']).plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Sex vs Heart Disease')
        axes[0, 2].set_xticklabels(['Female', 'Male'], rotation=0)

        # Chest pain type vs Target
        pd.crosstab(df['cp'], df['target']).plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Chest Pain Type vs Target')

        # Max heart rate by target
        df.boxplot(column='thalach', by='target', ax=axes[1, 1])
        axes[1, 1].set_title('Max Heart Rate by Target')
        plt.suptitle('')

        # Oldpeak by target
        df.boxplot(column='oldpeak', by='target', ax=axes[1, 2])
        axes[1, 2].set_title('ST Depression by Target')
        plt.suptitle('')

        # Number of vessels vs Target
        pd.crosstab(df['ca'], df['target']).plot(kind='bar', ax=axes[2, 0])
        axes[2, 0].set_title('Major Vessels vs Target')

        # Cholesterol distribution
        df['chol'].hist(bins=30, ax=axes[2, 1], color='orange', alpha=0.7)
        axes[2, 1].set_title('Cholesterol Distribution')
        axes[2, 1].axvline(x=200, color='green', linestyle='--', label='Normal')
        axes[2, 1].axvline(x=240, color='red', linestyle='--', label='High')
        axes[2, 1].legend()

        # Correlation heatmap
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'target']
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 2])
        axes[2, 2].set_title('Feature Correlations')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/heart_disease_eda.png', dpi=300, bbox_inches='tight')
        print(f"EDA saved to {output_dir}/heart_disease_eda.png")
        plt.close()

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train multiple models."""
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("\nTraining models...")

        self.models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        self.models['Logistic Regression'].fit(X_train_scaled, y_train)

        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train_scaled, y_train)

        self.models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)
        self.models['SVM'].fit(X_train_scaled, y_train)

        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train_scaled, y_train)

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
            self.models['XGBoost'].fit(X_train_scaled, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all models."""
        X_test_scaled = self.scaler.transform(X_test)
        results = []

        print("\n=== Model Evaluation ===")

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
            })

            print(f"\n{name}: Acc={results[-1]['Accuracy']:.4f}, "
                  f"Recall={results[-1]['Recall']:.4f}, AUC={results[-1]['ROC-AUC']:.4f}")

        results_df = pd.DataFrame(results).sort_values('Recall', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_results(self, results_df: pd.DataFrame, X_test: pd.DataFrame,
                    y_test: pd.Series, output_dir: str = '.') -> None:
        """Generate result visualizations."""
        X_test_scaled = self.scaler.transform(X_test)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Model comparison
        results_df.set_index('Model')[['Accuracy', 'Recall', 'F1-Score']].plot(
            kind='bar', ax=axes[0, 0]
        )
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # ROC curves
        for name, model in self.models.items():
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend(loc='lower right')

        # Confusion matrix
        y_pred = self.best_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[1, 0].set_title('Confusion Matrix - Best Model')

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[-10:]
            axes[1, 1].barh(range(10), importance[indices])
            axes[1, 1].set_yticks(range(10))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_title('Top 10 Feature Importance')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/model_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("HEART DISEASE PREDICTION - ML SOLUTION")
    print("=" * 70)

    predictor = HeartDiseasePredictor()

    # Create and preprocess data
    df = predictor.create_sample_data()
    print(f"\nDataset: {df.shape}, Disease rate: {df['target'].mean():.2%}")

    df = predictor.preprocess_data(df)
    predictor.plot_exploratory_analysis(df)

    # Prepare features
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                   'age_group_encoded', 'bp_cat_encoded', 'chol_risk_encoded',
                   'hr_reserve', 'cv_risk_score']
    predictor.feature_names = feature_cols

    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and evaluate
    predictor.train_models(X_train, y_train)
    results = predictor.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    predictor.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best Recall: {results.iloc[0]['Recall']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
