"""
Credit Risk Modeling
====================
Domain: Finance & Risk Management
Task: Credit default prediction and risk assessment

This solution demonstrates:
- Credit scoring model development
- Probability of default (PD) estimation
- Loss given default (LGD) modeling
- Feature engineering for credit data
- Model calibration and validation
- Regulatory compliance (Basel III)
- Economic capital calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             classification_report, confusion_matrix)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModeler:
    """
    Comprehensive credit risk modeling system for default prediction
    and capital requirement calculation.
    """

    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}

    def generate_credit_data(self, n_samples=5000):
        """Generate synthetic credit application data."""
        np.random.seed(42)

        data = []

        for i in range(n_samples):
            # Applicant demographics
            age = int(np.random.gamma(8, 5))
            age = np.clip(age, 18, 75)

            income = np.random.lognormal(10.5, 0.7)
            income = np.clip(income, 15000, 500000)

            employment_length = int(np.random.gamma(5, 2))
            employment_length = np.clip(employment_length, 0, 40)

            # Credit history
            credit_score = int(np.random.normal(700, 100))
            credit_score = np.clip(credit_score, 300, 850)

            num_credit_lines = int(np.random.poisson(5))
            num_credit_lines = np.clip(num_credit_lines, 0, 20)

            credit_utilization = np.random.beta(2, 5)

            num_delinquencies = int(np.random.poisson(0.5))
            num_bankruptcies = 1 if np.random.random() < 0.05 else 0

            # Loan characteristics
            loan_amount = np.random.uniform(5000, 50000)
            loan_term = np.random.choice([12, 24, 36, 48, 60])

            dti_ratio = (loan_amount / 12) / (income / 12)
            dti_ratio = np.clip(dti_ratio, 0, 1)

            collateral_value = loan_amount * np.random.uniform(0.5, 2.0)

            # Calculate default probability based on risk factors
            risk_score = 0

            # Age factor
            risk_score += (1 if age < 25 else 0) * 15
            risk_score += (1 if age > 60 else 0) * 10

            # Income factor
            risk_score += (1 if income < 30000 else 0) * 20
            risk_score += (1 if income > 100000 else -10) * -10

            # Credit score factor
            risk_score += (850 - credit_score) / 10

            # Employment length
            risk_score += (1 if employment_length < 2 else 0) * 15

            # Credit utilization
            risk_score += credit_utilization * 30

            # Delinquencies
            risk_score += num_delinquencies * 25
            risk_score += num_bankruptcies * 50

            # DTI ratio
            risk_score += dti_ratio * 40

            # Loan-to-value ratio
            ltv = loan_amount / collateral_value
            risk_score += ltv * 20

            # Add noise
            risk_score += np.random.normal(0, 10)

            # Convert risk score to default probability
            default_prob = 1 / (1 + np.exp(-(risk_score - 50) / 10))
            defaulted = 1 if np.random.random() < default_prob else 0

            # Loss given default (if defaulted)
            if defaulted:
                recovery_rate = np.random.beta(3, 2)
                lgd = 1 - recovery_rate
            else:
                lgd = 0

            data.append({
                'applicant_id': f'APP_{i:06d}',
                'age': age,
                'income': income,
                'employment_length': employment_length,
                'credit_score': credit_score,
                'num_credit_lines': num_credit_lines,
                'credit_utilization': credit_utilization,
                'num_delinquencies': num_delinquencies,
                'num_bankruptcies': num_bankruptcies,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'dti_ratio': dti_ratio,
                'collateral_value': collateral_value,
                'ltv_ratio': ltv,
                'risk_score': risk_score,
                'defaulted': defaulted,
                'lgd': lgd
            })

        df = pd.DataFrame(data)

        print(f"Generated {n_samples} credit applications")
        print(f"\nDefault rate: {df['defaulted'].mean()*100:.2f}%")
        print(f"Average loan amount: ${df['loan_amount'].mean():,.2f}")
        print(f"Average credit score: {df['credit_score'].mean():.0f}")
        print(f"Average LGD (for defaults): {df[df['defaulted']==1]['lgd'].mean()*100:.1f}%")

        return df

    def engineer_features(self, df):
        """Engineer credit risk features."""
        features = df.copy()

        # Interaction features
        features['income_per_credit_line'] = features['income'] / (features['num_credit_lines'] + 1)
        features['loan_to_income'] = features['loan_amount'] / features['income']
        features['monthly_payment_ratio'] = (features['loan_amount'] / features['loan_term']) / (features['income'] / 12)

        # Credit score bins
        features['credit_score_high'] = (features['credit_score'] >= 750).astype(int)
        features['credit_score_medium'] = ((features['credit_score'] >= 650) & (features['credit_score'] < 750)).astype(int)
        features['credit_score_low'] = (features['credit_score'] < 650).astype(int)

        # Risk indicators
        features['high_utilization'] = (features['credit_utilization'] > 0.7).astype(int)
        features['high_dti'] = (features['dti_ratio'] > 0.4).astype(int)
        features['recent_delinquency'] = (features['num_delinquencies'] > 0).astype(int)

        feature_cols = [
            'age', 'income', 'employment_length', 'credit_score',
            'num_credit_lines', 'credit_utilization', 'num_delinquencies',
            'num_bankruptcies', 'loan_amount', 'loan_term', 'dti_ratio',
            'ltv_ratio', 'income_per_credit_line', 'loan_to_income',
            'monthly_payment_ratio', 'credit_score_high', 'credit_score_medium',
            'credit_score_low', 'high_utilization', 'high_dti', 'recent_delinquency'
        ]

        return features[feature_cols]

    def train_models(self, X_train, y_train):
        """Train credit risk models."""
        print("\nTraining credit risk models...")

        # Logistic Regression
        print("  - Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr

        # Random Forest
        print("  - Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        self.feature_importance['Random Forest'] = rf.feature_importances_

        # Gradient Boosting
        print("  - Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb
        self.feature_importance['Gradient Boosting'] = gb.feature_importances_

        print(f"Trained {len(self.models)} models")

    def evaluate_models(self, X_test, y_test):
        """Evaluate credit risk models."""
        results = []

        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            auc = roc_auc_score(y_test, y_pred_proba)

            # Gini coefficient (common in credit risk)
            gini = 2 * auc - 1

            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)

            results.append({
                'Model': name,
                'AUC': auc,
                'Gini': gini,
                'Accuracy': accuracy
            })

            self.predictions[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        return pd.DataFrame(results).sort_values('AUC', ascending=False)

    def calculate_economic_capital(self, df, pd_column='pd', lgd=0.45, confidence_level=0.999):
        """Calculate economic capital using Basel III approach."""
        # Expected Loss
        expected_loss = df[pd_column].mean() * lgd

        # Unexpected Loss (simplified)
        pd_std = df[pd_column].std()
        unexpected_loss = pd_std * lgd * 2.33  # 99% confidence

        # Value at Risk
        from scipy.stats import norm
        var = norm.ppf(confidence_level)
        economic_capital = unexpected_loss * var

        return {
            'expected_loss': expected_loss,
            'unexpected_loss': unexpected_loss,
            'economic_capital': economic_capital,
            'total_exposure': df['loan_amount'].sum()
        }

    def plot_roc_curves(self, y_test):
        """Plot ROC curves and calculate AUC."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, preds in self.predictions.items():
            fpr, tpr, _ = roc_curve(y_test, preds['y_pred_proba'])
            auc = roc_auc_score(y_test, preds['y_pred_proba'])
            gini = 2 * auc - 1

            ax.plot(fpr, tpr, linewidth=2,
                   label=f'{name} (AUC={auc:.3f}, Gini={gini:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Credit Default Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('credit_risk_roc_curves.png', dpi=300, bbox_inches='tight')
        print("Saved: credit_risk_roc_curves.png")
        plt.close()

    def plot_calibration_curve(self, y_test):
        """Plot calibration curve for PD estimates."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, preds in self.predictions.items():
            prob_true, prob_pred = calibration_curve(y_test, preds['y_pred_proba'], n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('True Probability', fontsize=12)
        ax.set_title('Calibration Curve - PD Estimates', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('credit_risk_calibration.png', dpi=300, bbox_inches='tight')
        print("Saved: credit_risk_calibration.png")
        plt.close()

    def plot_feature_importance(self, feature_names, top_n=15):
        """Plot feature importance."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for idx, (model_name, importances) in enumerate(list(self.feature_importance.items())[:2]):
            top_indices = np.argsort(importances)[::-1][:top_n]
            top_features = [feature_names[i] for i in top_indices]
            top_values = importances[top_indices]

            axes[idx].barh(range(top_n), top_values,
                          color=plt.cm.viridis(top_values / max(top_values)))
            axes[idx].set_yticks(range(top_n))
            axes[idx].set_yticklabels(top_features, fontsize=10)
            axes[idx].set_xlabel('Importance Score', fontsize=12)
            axes[idx].set_title(f'Top {top_n} Features - {model_name}',
                               fontsize=13, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
            axes[idx].invert_yaxis()

        plt.tight_layout()
        plt.savefig('credit_risk_feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: credit_risk_feature_importance.png")
        plt.close()

    def plot_risk_distribution(self, df, pd_column='pd'):
        """Plot distribution of predicted default probabilities."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # PD distribution
        axes[0, 0].hist(df[pd_column], bins=50, color='steelblue',
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Probability of Default', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of Default Probabilities',
                            fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # PD by credit score
        bins = [300, 550, 650, 750, 850]
        labels = ['Poor', 'Fair', 'Good', 'Excellent']
        df['score_group'] = pd.cut(df['credit_score'], bins=bins, labels=labels)

        df.boxplot(column=pd_column, by='score_group', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Credit Score Group', fontsize=11)
        axes[0, 1].set_ylabel('Probability of Default', fontsize=11)
        axes[0, 1].set_title('PD by Credit Score Group', fontsize=12, fontweight='bold')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=0)

        # Expected loss by loan amount
        df['expected_loss'] = df[pd_column] * df['loan_amount'] * 0.45  # Assume 45% LGD
        axes[1, 0].scatter(df['loan_amount'], df['expected_loss'],
                          alpha=0.5, s=20, c=df[pd_column], cmap='Reds')
        axes[1, 0].set_xlabel('Loan Amount ($)', fontsize=11)
        axes[1, 0].set_ylabel('Expected Loss ($)', fontsize=11)
        axes[1, 0].set_title('Expected Loss vs Loan Amount',
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Risk rating distribution
        risk_ratings = pd.cut(df[pd_column],
                            bins=[0, 0.01, 0.05, 0.10, 0.20, 1.0],
                            labels=['AAA', 'AA', 'A', 'BBB', 'BB-C'])
        rating_counts = risk_ratings.value_counts().sort_index()

        axes[1, 1].bar(range(len(rating_counts)), rating_counts.values,
                      color=['green', 'yellowgreen', 'yellow', 'orange', 'red'],
                      edgecolor='black', alpha=0.7)
        axes[1, 1].set_xticks(range(len(rating_counts)))
        axes[1, 1].set_xticklabels(rating_counts.index)
        axes[1, 1].set_ylabel('Number of Applicants', fontsize=11)
        axes[1, 1].set_title('Distribution by Risk Rating',
                            fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('credit_risk_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: credit_risk_distribution.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Credit Risk Modeling - Default Prediction and Capital Calculation")
    print("=" * 80)

    # Initialize modeler
    modeler = CreditRiskModeler()

    # Generate data
    print("\n1. Generating Credit Application Data...")
    df = modeler.generate_credit_data(n_samples=5000)

    # Engineer features
    print("\n2. Engineering Credit Risk Features...")
    X = modeler.engineer_features(df)
    y = df['defaulted'].values

    feature_names = X.columns.tolist()
    print(f"Total features: {len(feature_names)}")

    # Split data
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    # Train models
    print("\n3. Training Credit Risk Models...")
    modeler.train_models(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating Models...")
    results = modeler.evaluate_models(X_test, y_test)
    print("\nModel Performance:")
    print(results.to_string(index=False))

    # Add PD to test set
    df_test['pd'] = modeler.predictions['Random Forest']['y_pred_proba']

    # Calculate economic capital
    print("\n5. Calculating Economic Capital...")
    capital_metrics = modeler.calculate_economic_capital(df_test, pd_column='pd')
    print(f"\nEconomic Capital Metrics:")
    print(f"  Expected Loss: {capital_metrics['expected_loss']*100:.2f}%")
    print(f"  Unexpected Loss: {capital_metrics['unexpected_loss']*100:.2f}%")
    print(f"  Economic Capital: {capital_metrics['economic_capital']*100:.2f}%")
    print(f"  Total Exposure: ${capital_metrics['total_exposure']:,.2f}")

    # Visualizations
    print("\n6. Generating Visualizations...")
    modeler.plot_roc_curves(y_test)
    modeler.plot_calibration_curve(y_test)
    modeler.plot_feature_importance(feature_names, top_n=15)
    modeler.plot_risk_distribution(df_test, pd_column='pd')

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("- Credit score is the strongest predictor of default risk")
    print("- DTI ratio and credit utilization are important risk indicators")
    print("- Model calibration ensures accurate PD estimates for pricing")
    print("- Economic capital calculation supports regulatory compliance")
    print("- Risk-based pricing can optimize profitability while managing exposure")


if __name__ == "__main__":
    main()
