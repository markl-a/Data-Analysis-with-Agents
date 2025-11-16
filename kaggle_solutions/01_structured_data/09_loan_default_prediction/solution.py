"""
Loan Default Prediction
========================

Problem: Predict whether a borrower will default on their loan based on financial and demographic features

Kaggle-style competition: https://www.kaggle.com/datasets/yasserh/loan-default-dataset
Difficulty: ⭐⭐

This solution demonstrates:
- Handling imbalanced classification
- Feature engineering for financial data
- Multiple algorithm comparison
- SMOTE for class balancing
- Risk score calibration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class LoanDefaultPredictor:
    """Predicts loan default probability using ensemble methods"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def create_sample_data(self, n_samples=5000):
        """Generate realistic loan application data"""
        np.random.seed(42)

        # Generate features
        data = {
            'age': np.random.randint(21, 70, n_samples),
            'income': np.random.lognormal(10.5, 0.8, n_samples),
            'loan_amount': np.random.lognormal(10, 1.2, n_samples),
            'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
            'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
            'num_credit_lines': np.random.poisson(4, n_samples),
            'debt_to_income': np.random.uniform(0, 0.8, n_samples),
            'previous_defaults': np.random.poisson(0.3, n_samples),
            'loan_purpose': np.random.choice(['home', 'business', 'education', 'auto', 'personal'], n_samples),
            'employment_type': np.random.choice(['full_time', 'part_time', 'self_employed', 'unemployed'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'home_ownership': np.random.choice(['own', 'rent', 'mortgage'], n_samples, p=[0.3, 0.4, 0.3])
        }

        df = pd.DataFrame(data)

        # Calculate loan-to-income ratio
        df['loan_to_income'] = df['loan_amount'] / df['income']

        # Generate target with realistic dependencies
        default_prob = (
            -0.003 * df['credit_score'] +
            0.5 * df['debt_to_income'] +
            0.3 * df['previous_defaults'] +
            0.15 * df['loan_to_income'] +
            -0.01 * df['employment_years'] +
            (df['employment_type'] == 'unemployed').astype(int) * 2.0 +
            np.random.normal(0, 0.5, n_samples)
        )

        # Convert to probability
        default_prob = 1 / (1 + np.exp(-default_prob))
        df['default'] = (default_prob > 0.5).astype(int)

        return df

    def engineer_features(self, df):
        """Create advanced financial features"""
        df = df.copy()

        # Debt burden features
        df['total_debt_estimate'] = df['loan_amount'] + (df['income'] * df['debt_to_income'])
        df['credit_utilization'] = df['debt_to_income'] * 100

        # Risk indicators
        df['high_risk'] = ((df['credit_score'] < 600) | (df['previous_defaults'] > 0)).astype(int)
        df['stable_employment'] = ((df['employment_years'] > 2) & (df['employment_type'] == 'full_time')).astype(int)

        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['young', 'mid', 'mature', 'senior'])

        # Income categories
        df['income_category'] = pd.qcut(df['income'], q=4, labels=['low', 'medium', 'high', 'very_high'])

        return df

    def prepare_data(self, df):
        """Encode and scale features"""
        df = self.engineer_features(df)

        # Separate features and target
        X = df.drop('default', axis=1)
        y = df['default']

        # Encode categorical variables
        categorical_cols = ['loan_purpose', 'employment_type', 'home_ownership', 'age_group', 'income_category']

        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        return X, y

    def train_models(self, X, y):
        """Train multiple models and compare"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiple models
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }

        results = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train_balanced)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': roc_auc_score(y_test, y_pred_proba)
            }

        return results, X_test_scaled, y_test

    def plot_results(self, results, y_test):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC Curves
        ax = axes[0, 0]
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={result['auc_score']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # AUC Comparison
        ax = axes[0, 1]
        models = list(results.keys())
        aucs = [results[m]['auc_score'] for m in models]
        bars = ax.barh(models, aucs, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_xlabel('AUC Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xlim(0, 1)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{aucs[i]:.3f}',
                   ha='left', va='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_result = results[best_model_name]

        ax = axes[1, 0]
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {best_model_name}')

        # Feature Importance (for Random Forest)
        if 'Random Forest' in results:
            ax = axes[1, 1]
            rf_model = results['Random Forest']['model']

            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))],
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            ax.barh(range(len(feature_importance)), feature_importance['importance'], color='#9b59b6')
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importances')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('loan_default_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'loan_default_analysis.png'")
        plt.show()

    def print_results(self, results, y_test):
        """Print detailed results"""
        print("\n" + "="*80)
        print("LOAN DEFAULT PREDICTION RESULTS")
        print("="*80)

        for name, result in results.items():
            print(f"\n{'='*40}")
            print(f"Model: {name}")
            print(f"{'='*40}")
            print(f"AUC Score: {result['auc_score']:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, result['predictions'],
                                       target_names=['No Default', 'Default']))

        # Best model summary
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_auc = results[best_model_name]['auc_score']
        print(f"\n{'='*80}")
        print(f"🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    print("🏦 Loan Default Prediction System")
    print("=" * 80)

    # Initialize predictor
    predictor = LoanDefaultPredictor()

    # Generate data
    print("\n📊 Generating sample loan data...")
    df = predictor.create_sample_data(n_samples=5000)
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")

    # Prepare data
    print("\n🔧 Preparing features...")
    X, y = predictor.prepare_data(df)
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training models...")
    results, X_test, y_test = predictor.train_models(X, y)

    # Print results
    predictor.print_results(results, y_test)

    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_results(results, y_test)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
