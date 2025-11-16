"""
Financial Risk Assessment System
=================================

Problem: Assess and quantify financial risk for loan applications and
credit decisions using machine learning and statistical models

Kaggle-style competition: Credit Risk Assessment
Difficulty: ⭐⭐⭐⭐

This solution demonstrates:
- Credit scoring and risk rating
- Probability of default (PD) estimation
- Loss given default (LGD) calculation
- Expected loss and risk-adjusted pricing
- Regulatory compliance (Basel II/III)
- Portfolio risk analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                            roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class RiskAssessmentSystem:
    """Financial risk assessment and credit scoring"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def create_sample_data(self, n_samples=10000):
        """Generate realistic loan application data"""
        np.random.seed(42)

        # Applicant demographics
        data = {
            'age': np.random.normal(42, 15, n_samples).clip(18, 80),
            'income': np.random.lognormal(10.5, 0.6, n_samples).clip(20000, 500000),
            'employment_years': np.random.exponential(8, n_samples).clip(0, 50),
            'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
            'debt_to_income': np.random.beta(2, 5, n_samples),
            'loan_amount': np.random.lognormal(10, 0.8, n_samples).clip(5000, 500000),
            'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 84], n_samples,
                                                p=[0.1, 0.15, 0.3, 0.25, 0.15, 0.05]),
            'home_ownership': np.random.choice(['rent', 'own', 'mortgage'], n_samples,
                                              p=[0.35, 0.25, 0.4]),
            'purpose': np.random.choice(['debt_consolidation', 'home_improvement',
                                        'business', 'auto', 'personal'], n_samples,
                                       p=[0.35, 0.2, 0.15, 0.2, 0.1]),
            'num_credit_lines': np.random.poisson(8, n_samples).clip(1, 30),
            'num_delinquencies': np.random.poisson(0.5, n_samples),
            'credit_inquiries_6m': np.random.poisson(1, n_samples),
            'bankruptcies': np.random.choice([0, 1, 2], n_samples, p=[0.92, 0.07, 0.01]),
            'utilization_rate': np.random.beta(3, 4, n_samples)
        }

        df = pd.DataFrame(data)

        # Calculate default probability
        default_score = (
            -0.02 * (df['credit_score'] - 300) +
            -0.00005 * (df['income']) +
            0.8 * df['debt_to_income'] +
            0.00003 * df['loan_amount'] +
            0.15 * df['num_delinquencies'] +
            0.3 * df['bankruptcies'] +
            0.2 * df['utilization_rate'] +
            0.1 * (df['credit_inquiries_6m'] > 3).astype(int) +
            -0.02 * df['employment_years'] +
            np.random.normal(0, 0.8, n_samples) -
            3.0  # Offset to get reasonable default rate
        )

        default_prob = 1 / (1 + np.exp(-default_score))
        df['default'] = (default_prob > 0.5).astype(int)

        # Loss given default (LGD) - percentage of loan lost if default occurs
        df['lgd'] = np.where(
            df['default'] == 1,
            np.random.beta(3, 2, n_samples) * 0.8 + 0.2,  # 20-100% loss if default
            0
        )

        return df

    def engineer_features(self, df):
        """Create risk assessment features"""
        df = df.copy()

        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=['home_ownership', 'purpose'], prefix=['home', 'purpose'])

        # Financial ratios
        df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)
        df['monthly_payment_estimate'] = (df['loan_amount'] * 0.05) / 12  # Rough estimate
        df['payment_to_income'] = df['monthly_payment_estimate'] / (df['income'] / 12 + 1)

        # Credit health indicators
        df['good_credit_score'] = (df['credit_score'] >= 700).astype(int)
        df['high_utilization'] = (df['utilization_rate'] > 0.7).astype(int)
        df['recent_inquiries'] = (df['credit_inquiries_6m'] > 2).astype(int)
        df['has_delinquencies'] = (df['num_delinquencies'] > 0).astype(int)
        df['has_bankruptcy'] = (df['bankruptcies'] > 0).astype(int)

        # Risk tiers based on credit score
        df['risk_tier'] = pd.cut(df['credit_score'],
                                 bins=[0, 580, 670, 740, 850],
                                 labels=[3, 2, 1, 0])  # 3=highest risk, 0=lowest

        # Stability indicators
        df['stable_employment'] = (df['employment_years'] >= 2).astype(int)
        df['low_debt_burden'] = (df['debt_to_income'] < 0.36).astype(int)

        # Composite risk score
        df['composite_risk_score'] = (
            (1 - df['credit_score'] / 850) * 30 +
            df['debt_to_income'] * 25 +
            df['loan_to_income'] * 20 +
            df['has_delinquencies'] * 10 +
            df['has_bankruptcy'] * 10 +
            df['high_utilization'] * 5
        )

        return df

    def train_models(self, X, y):
        """Train credit risk models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Default rate: {y_train.mean():.2%}")

        # Initialize models
        models_config = {
            'Logistic Regression': LogisticRegression(class_weight='balanced',
                                                      max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12,
                                                   class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,
                                                           learning_rate=0.1,
                                                           max_depth=5, random_state=42)
        }

        results = {}
        for name, model in models_config.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'cv_score': cross_val_score(model, X_train_scaled, y_train,
                                           cv=5, scoring='roc_auc').mean()
            }

        return results, X_test_scaled, y_test, X_train

    def calculate_expected_loss(self, df, pd_predictions):
        """Calculate expected loss and risk metrics"""
        # Expected Loss = PD × LGD × EAD (Exposure at Default)
        df = df.copy()
        df['pd'] = pd_predictions  # Probability of Default
        df['ead'] = df['loan_amount']  # Exposure at Default

        # For non-defaulted loans, estimate potential LGD
        avg_lgd = 0.45  # Industry standard
        df['lgd_estimate'] = np.where(df['default'] == 1, df['lgd'], avg_lgd)

        df['expected_loss'] = df['pd'] * df['lgd_estimate'] * df['ead']

        # Risk-adjusted pricing
        target_roi = 0.08  # 8% target return
        cost_of_funds = 0.03  # 3% cost of funds
        operating_cost = 0.02  # 2% operating cost

        df['risk_premium'] = df['expected_loss'] / df['ead']
        df['suggested_rate'] = cost_of_funds + operating_cost + df['risk_premium'] + target_roi

        return df

    def calculate_portfolio_metrics(self, df_with_predictions):
        """Calculate portfolio-level risk metrics"""
        total_exposure = df_with_predictions['ead'].sum()
        total_expected_loss = df_with_predictions['expected_loss'].sum()

        portfolio_metrics = {
            'total_exposure': total_exposure,
            'total_expected_loss': total_expected_loss,
            'expected_loss_rate': total_expected_loss / total_exposure,
            'avg_pd': df_with_predictions['pd'].mean(),
            'avg_lgd': df_with_predictions['lgd_estimate'].mean(),
            'avg_suggested_rate': df_with_predictions['suggested_rate'].mean(),
            'high_risk_exposure': df_with_predictions[
                df_with_predictions['pd'] > 0.5
            ]['ead'].sum(),
            'high_risk_count': (df_with_predictions['pd'] > 0.5).sum()
        }

        return portfolio_metrics

    def plot_results(self, results, y_test, df_test, feature_names):
        """Visualize risk assessment results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ROC Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC={result['auc_score']:.3f})", linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curves - Credit Risk', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Model Performance
        ax2 = fig.add_subplot(gs[0, 1])
        model_names = list(results.keys())
        auc_scores = [results[m]['auc_score'] for m in model_names]
        cv_scores = [results[m]['cv_score'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        ax2.bar(x - width/2, auc_scores, width, label='Test AUC', color='#3498db')
        ax2.bar(x + width/2, cv_scores, width, label='CV AUC', color='#2ecc71')
        ax2.set_ylabel('AUC Score', fontsize=11)
        ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0.5, 1.0)

        # Confusion Matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_result = results[best_model_name]

        ax3 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax3, cbar=True)
        ax3.set_xlabel('Predicted', fontsize=11)
        ax3.set_ylabel('Actual', fontsize=11)
        ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(['No Default', 'Default'])
        ax3.set_yticklabels(['No Default', 'Default'])

        # Default Probability Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        no_default_probs = best_result['probabilities'][y_test == 0]
        default_probs = best_result['probabilities'][y_test == 1]

        ax4.hist(no_default_probs, bins=50, alpha=0.6, label='No Default',
                color='green', edgecolor='black', density=True)
        ax4.hist(default_probs, bins=50, alpha=0.6, label='Default',
                color='red', edgecolor='black', density=True)
        ax4.set_xlabel('Default Probability', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title('Default Probability Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Calculate expected loss
        df_with_el = self.calculate_expected_loss(df_test, best_result['probabilities'])
        portfolio_metrics = self.calculate_portfolio_metrics(df_with_el)

        # Expected Loss by Risk Tier
        ax5 = fig.add_subplot(gs[1, 1])
        risk_groups = pd.cut(df_with_el['pd'], bins=[0, 0.1, 0.3, 0.5, 1.0],
                            labels=['Low', 'Medium', 'High', 'Very High'])
        el_by_risk = df_with_el.groupby(risk_groups)['expected_loss'].sum() / 1000

        colors_bar = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax5.bar(range(len(el_by_risk)), el_by_risk.values, color=colors_bar,
                      edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(el_by_risk)))
        ax5.set_xticklabels(el_by_risk.index)
        ax5.set_ylabel('Expected Loss ($1000s)', fontsize=11)
        ax5.set_title('Expected Loss by Risk Tier', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, el_by_risk.values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Suggested Interest Rates
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(df_with_el['pd'], df_with_el['suggested_rate'] * 100,
                   alpha=0.5, s=30, c=df_with_el['default'], cmap='RdYlGn_r')
        ax6.set_xlabel('Probability of Default', fontsize=11)
        ax6.set_ylabel('Suggested Rate (%)', fontsize=11)
        ax6.set_title('Risk-Adjusted Pricing', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # Feature Importance
        ax7 = fig.add_subplot(gs[2, 0])
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(12)

            ax7.barh(range(len(feature_importance)), feature_importance['importance'],
                    color='#9b59b6', edgecolor='black')
            ax7.set_yticks(range(len(feature_importance)))
            ax7.set_yticklabels(feature_importance['feature'], fontsize=9)
            ax7.set_xlabel('Importance', fontsize=11)
            ax7.set_title('Top Feature Importances', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='x')

        # Loan Amount vs Default Rate
        ax8 = fig.add_subplot(gs[2, 1])
        loan_bins = pd.cut(df_with_el['loan_amount'], bins=5)
        default_by_amount = df_with_el.groupby(loan_bins)['default'].mean()

        ax8.bar(range(len(default_by_amount)), default_by_amount.values * 100,
               color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax8.set_xticks(range(len(default_by_amount)))
        ax8.set_xticklabels([f'${int(i.left/1000)}-{int(i.right/1000)}K'
                            for i in default_by_amount.index], rotation=45, ha='right', fontsize=8)
        ax8.set_ylabel('Default Rate (%)', fontsize=11)
        ax8.set_title('Default Rate by Loan Amount', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        summary_text = f"""
        ╔════════════════════════════════════════════╗
        ║      CREDIT RISK ASSESSMENT SUMMARY         ║
        ╚════════════════════════════════════════════╝

        Best Model: {best_model_name}
        ROC AUC: {best_result['auc_score']:.4f}
        CV AUC: {best_result['cv_score']:.4f}

        ┌──────────────────────────────────────────┐
        │ MODEL PERFORMANCE                         │
        ├──────────────────────────────────────────┤
        │ Precision:      {precision:6.2%}                 │
        │ Recall:         {recall:6.2%}                 │
        │ Accuracy:       {((tp+tn)/(tp+tn+fp+fn)):6.2%}                 │
        └──────────────────────────────────────────┘

        ┌──────────────────────────────────────────┐
        │ PORTFOLIO RISK METRICS                    │
        ├──────────────────────────────────────────┤
        │ Total Exposure:     ${portfolio_metrics['total_exposure']/1e6:>8.2f}M    │
        │ Expected Loss:      ${portfolio_metrics['total_expected_loss']/1e6:>8.2f}M    │
        │ Loss Rate:          {portfolio_metrics['expected_loss_rate']:>7.2%}     │
        │ Avg PD:             {portfolio_metrics['avg_pd']:>7.2%}     │
        │ Avg LGD:            {portfolio_metrics['avg_lgd']:>7.2%}     │
        │ Avg Suggested Rate: {portfolio_metrics['avg_suggested_rate']:>7.2%}     │
        └──────────────────────────────────────────┘
        """
        ax9.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.savefig('risk_assessment_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'risk_assessment_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("💰 Financial Risk Assessment System")
    print("=" * 80)

    system = RiskAssessmentSystem()

    # Generate data
    print("\n📊 Generating loan application data...")
    df = system.create_sample_data(n_samples=10000)
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")

    # Engineer features
    print("\n🔧 Engineering risk assessment features...")
    df_engineered = system.engineer_features(df)

    # Prepare data
    feature_cols = [col for col in df_engineered.columns
                   if col not in ['default', 'lgd']]
    X = df_engineered[feature_cols]
    y = df_engineered['default']

    # Train models
    print("\n🤖 Training credit risk models...")
    results, X_test, y_test, X_train = system.train_models(X, y)

    # Get test set with original features for analysis
    test_indices = X.index[-len(X_test):]
    df_test = df_engineered.loc[test_indices].copy()

    # Plot results
    print("\n📈 Generating visualizations...")
    system.plot_results(results, y_test, df_test, X.columns.tolist())

    print("\n✅ Risk assessment analysis complete!")


if __name__ == "__main__":
    main()
