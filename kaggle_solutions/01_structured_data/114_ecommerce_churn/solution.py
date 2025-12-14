"""
E-Commerce Customer Churn Prediction - ML Solution

This module predicts customer churn for e-commerce platforms using
customer behavior data and machine learning algorithms.

Dataset: https://www.kaggle.com/datasets/nabihazahid/e-commerce-customer-insights-and-churn-dataset
Difficulty: ⭐⭐ Intermediate Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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


class EcommerceChurnPredictor:
    """E-Commerce Customer Churn Prediction Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_model = None
        self.feature_names: List[str] = []

    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic e-commerce customer dataset."""
        np.random.seed(42)
        n_samples = 5000

        # Customer tenure (months)
        tenure = np.random.exponential(12, n_samples).clip(0, 60).astype(int)

        # Device preferences
        preferred_login_device = np.random.choice(
            ['Mobile Phone', 'Computer', 'Phone'], n_samples, p=[0.55, 0.35, 0.10]
        )

        # City tier
        city_tier = np.random.choice([1, 2, 3], n_samples, p=[0.35, 0.40, 0.25])

        # Distance to warehouse
        warehouse_to_home = np.random.exponential(15, n_samples).clip(1, 50).round(0)

        # Payment preferences
        preferred_payment_mode = np.random.choice(
            ['Debit Card', 'Credit Card', 'E wallet', 'UPI', 'Cash on Delivery'],
            n_samples, p=[0.25, 0.20, 0.20, 0.20, 0.15]
        )

        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])

        # App usage
        hour_spend_on_app = np.random.exponential(2, n_samples).clip(0.1, 5).round(1)

        # Registered devices
        number_of_device_registered = np.random.choice([1, 2, 3, 4, 5], n_samples,
                                                        p=[0.20, 0.35, 0.25, 0.15, 0.05])

        # Order categories
        preferred_order_cat = np.random.choice(
            ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'],
            n_samples, p=[0.25, 0.30, 0.20, 0.15, 0.10]
        )

        # Satisfaction score (1-5)
        satisfaction_score = np.random.choice([1, 2, 3, 4, 5], n_samples,
                                              p=[0.05, 0.15, 0.35, 0.30, 0.15])

        marital_status = np.random.choice(['Single', 'Married', 'Divorced'],
                                          n_samples, p=[0.40, 0.50, 0.10])

        number_of_address = np.random.choice([1, 2, 3, 4, 5], n_samples,
                                             p=[0.25, 0.35, 0.25, 0.10, 0.05])

        # Complaints (important churn indicator)
        complain = np.random.choice([0, 1], n_samples, p=[0.80, 0.20])

        # Order metrics
        order_amount_hike = np.random.normal(15, 5, n_samples).clip(-10, 30).round(0)
        coupon_used = np.random.poisson(2, n_samples).clip(0, 15)
        order_count = np.random.poisson(3, n_samples).clip(1, 20)
        day_since_last_order = np.random.exponential(10, n_samples).clip(0, 60).round(0)
        cashback_amount = np.random.exponential(150, n_samples).clip(0, 500).round(2)

        # Generate churn based on risk factors
        churn_prob = (
            0.10 +  # base rate
            0.25 * complain +
            0.15 * (satisfaction_score <= 2) +
            0.10 * (day_since_last_order > 30) +
            0.10 * (tenure < 6) +
            0.05 * (hour_spend_on_app < 1) -
            0.05 * (cashback_amount > 200) -
            0.05 * (coupon_used > 3)
        )
        churn = (np.random.random(n_samples) < churn_prob.clip(0.05, 0.8)).astype(int)

        return pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'tenure': tenure,
            'preferred_login_device': preferred_login_device,
            'city_tier': city_tier,
            'warehouse_to_home': warehouse_to_home,
            'preferred_payment_mode': preferred_payment_mode,
            'gender': gender,
            'hour_spend_on_app': hour_spend_on_app,
            'number_of_device_registered': number_of_device_registered,
            'preferred_order_cat': preferred_order_cat,
            'satisfaction_score': satisfaction_score,
            'marital_status': marital_status,
            'number_of_address': number_of_address,
            'complain': complain,
            'order_amount_hike_from_last_year': order_amount_hike,
            'coupon_used': coupon_used,
            'order_count': order_count,
            'day_since_last_order': day_since_last_order,
            'cashback_amount': cashback_amount,
            'churn': churn
        })

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and engineer features."""
        df = df.copy()

        # Encode categorical variables
        cat_cols = ['preferred_login_device', 'preferred_payment_mode', 'gender',
                   'preferred_order_cat', 'marital_status']

        for col in cat_cols:
            self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])

        # Feature engineering
        # Tenure groups
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 3, 12, 24, 100],
                                    labels=['new', 'growing', 'mature', 'loyal'])
        df['tenure_group_encoded'] = LabelEncoder().fit_transform(df['tenure_group'])

        # Activity level
        df['activity_score'] = (
            df['hour_spend_on_app'] * 2 +
            df['order_count'] * 1.5 +
            (30 - df['day_since_last_order'].clip(0, 30)) / 30 * 3
        )

        # Recency (inverse of days since last order)
        df['recency_score'] = 1 / (df['day_since_last_order'] + 1)

        # Customer value score
        df['value_score'] = df['order_count'] * df['cashback_amount'] / 100

        # Engagement level
        df['engagement_level'] = (
            df['hour_spend_on_app'] +
            df['coupon_used'] * 0.5 +
            df['number_of_device_registered'] * 0.3
        )

        # Risk indicators
        df['high_risk_indicator'] = (
            df['complain'] +
            (df['satisfaction_score'] <= 2).astype(int) +
            (df['day_since_last_order'] > 30).astype(int)
        )

        return df

    def plot_exploratory_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate EDA visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('E-Commerce Churn Analysis - EDA', fontsize=16)

        # Churn distribution
        df['churn'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title('Churn Distribution')
        axes[0, 0].set_xticklabels(['Retained', 'Churned'], rotation=0)

        # Churn by complaint
        pd.crosstab(df['complain'], df['churn']).plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Churn by Complaint Status')
        axes[0, 1].set_xticklabels(['No Complaint', 'Complaint'], rotation=0)

        # Churn by satisfaction score
        sat_churn = df.groupby('satisfaction_score')['churn'].mean()
        sat_churn.plot(kind='bar', ax=axes[0, 2], color='coral')
        axes[0, 2].set_title('Churn Rate by Satisfaction Score')
        axes[0, 2].set_ylabel('Churn Rate')

        # Tenure distribution by churn
        df[df['churn'] == 0]['tenure'].hist(bins=20, ax=axes[1, 0], alpha=0.5, label='Retained', color='green')
        df[df['churn'] == 1]['tenure'].hist(bins=20, ax=axes[1, 0], alpha=0.5, label='Churned', color='red')
        axes[1, 0].set_title('Tenure Distribution by Churn')
        axes[1, 0].legend()

        # Days since last order
        df.boxplot(column='day_since_last_order', by='churn', ax=axes[1, 1])
        axes[1, 1].set_title('Days Since Last Order by Churn')
        plt.suptitle('')

        # App usage by churn
        df.boxplot(column='hour_spend_on_app', by='churn', ax=axes[1, 2])
        axes[1, 2].set_title('App Usage by Churn')
        plt.suptitle('')

        # Churn by preferred category
        cat_churn = df.groupby('preferred_order_cat')['churn'].mean().sort_values(ascending=False)
        cat_churn.plot(kind='bar', ax=axes[2, 0], color='steelblue')
        axes[2, 0].set_title('Churn Rate by Order Category')
        axes[2, 0].tick_params(axis='x', rotation=45)

        # Cashback vs Churn
        df.boxplot(column='cashback_amount', by='churn', ax=axes[2, 1])
        axes[2, 1].set_title('Cashback Amount by Churn')
        plt.suptitle('')

        # Feature correlations
        numeric_cols = ['tenure', 'hour_spend_on_app', 'satisfaction_score',
                       'complain', 'day_since_last_order', 'cashback_amount', 'churn']
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 2])
        axes[2, 2].set_title('Feature Correlations')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/churn_eda.png', dpi=300, bbox_inches='tight')
        print(f"EDA saved to {output_dir}/churn_eda.png")
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

        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1
            )
            self.models['LightGBM'].fit(X_train_scaled, y_train)

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

        results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def identify_at_risk_customers(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Identify customers at risk of churning."""
        feature_cols = self.feature_names
        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)

        churn_prob = self.best_model.predict_proba(X_scaled)[:, 1]

        at_risk = df.copy()
        at_risk['churn_probability'] = churn_prob
        at_risk['risk_level'] = pd.cut(churn_prob, bins=[0, 0.3, 0.6, 1.0],
                                       labels=['Low', 'Medium', 'High'])

        return at_risk[at_risk['churn_probability'] >= threshold].sort_values(
            'churn_probability', ascending=False
        )

    def plot_results(self, results_df: pd.DataFrame, X_test: pd.DataFrame,
                    y_test: pd.Series, output_dir: str = '.') -> None:
        """Generate result visualizations."""
        X_test_scaled = self.scaler.transform(X_test)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Model comparison
        results_df.set_index('Model')[['Accuracy', 'Recall', 'F1-Score', 'ROC-AUC']].plot(
            kind='bar', ax=axes[0, 0]
        )
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend(loc='lower right')

        # ROC curves
        for name, model in self.models.items():
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend(loc='lower right')

        # Confusion matrix
        y_pred = self.best_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['Retained', 'Churned'],
                   yticklabels=['Retained', 'Churned'])
        axes[1, 0].set_title('Confusion Matrix - Best Model')

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[-10:]
            axes[1, 1].barh(range(10), importance[indices], color='steelblue')
            axes[1, 1].set_yticks(range(10))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_title('Top 10 Features')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/churn_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/churn_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("E-COMMERCE CUSTOMER CHURN PREDICTION")
    print("=" * 70)

    predictor = EcommerceChurnPredictor()

    # Create and preprocess data
    df = predictor.create_sample_data()
    print(f"\nDataset: {df.shape}, Churn rate: {df['churn'].mean():.2%}")

    df = predictor.preprocess_data(df)
    predictor.plot_exploratory_analysis(df)

    # Prepare features
    feature_cols = ['tenure', 'city_tier', 'warehouse_to_home', 'hour_spend_on_app',
                   'number_of_device_registered', 'satisfaction_score', 'number_of_address',
                   'complain', 'order_amount_hike_from_last_year', 'coupon_used',
                   'order_count', 'day_since_last_order', 'cashback_amount',
                   'preferred_login_device_encoded', 'preferred_payment_mode_encoded',
                   'gender_encoded', 'preferred_order_cat_encoded', 'marital_status_encoded',
                   'tenure_group_encoded', 'activity_score', 'recency_score',
                   'value_score', 'engagement_level', 'high_risk_indicator']

    predictor.feature_names = feature_cols
    X = df[feature_cols]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and evaluate
    predictor.train_models(X_train, y_train)
    results = predictor.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    predictor.plot_results(results, X_test, y_test)

    # Identify at-risk customers
    at_risk = predictor.identify_at_risk_customers(df, threshold=0.6)
    print(f"\nIdentified {len(at_risk)} high-risk customers")

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best ROC-AUC: {results.iloc[0]['ROC-AUC']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
