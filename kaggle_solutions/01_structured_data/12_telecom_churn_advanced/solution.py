"""
Advanced Telecom Customer Churn Prediction with CLV
===================================================

Problem: Predict customer churn and calculate Customer Lifetime Value (CLV)
to prioritize retention efforts

Kaggle-style competition: Telecom Churn Prediction with Business Metrics
Difficulty: ⭐⭐⭐

This solution demonstrates:
- Advanced churn prediction with cost-sensitive learning
- Customer Lifetime Value (CLV) calculation
- Retention strategy optimization
- Segment-specific churn models
- Economic impact analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class TelecomChurnPredictor:
    """Advanced churn prediction with CLV and retention economics"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.retention_cost = 100  # Cost to retain a customer
        self.acquisition_cost = 200  # Cost to acquire new customer

    def create_sample_data(self, n_samples=5000):
        """Generate realistic telecom customer data"""
        np.random.seed(42)

        # Customer demographics
        data = {
            'customer_id': range(1, n_samples + 1),
            'tenure_months': np.random.exponential(24, n_samples).clip(1, 120),
            'age': np.random.normal(42, 15, n_samples).clip(18, 80),
            'monthly_charges': np.random.lognormal(4.2, 0.4, n_samples).clip(20, 200),
            'total_charges': None,  # Will calculate based on tenure
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                             n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check',
                                               'Bank transfer', 'Credit card'],
                                              n_samples, p=[0.3, 0.2, 0.25, 0.25]),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'],
                                                n_samples, p=[0.35, 0.45, 0.2]),
            'phone_service': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'multiple_lines': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'online_security': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'online_backup': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'device_protection': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'tech_support': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'streaming_tv': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'streaming_movies': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'paperless_billing': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'num_support_calls': np.random.poisson(2, n_samples),
            'num_late_payments': np.random.poisson(1, n_samples),
            'avg_call_duration': np.random.gamma(2, 30, n_samples),
            'data_usage_gb': np.random.lognormal(3.5, 1.2, n_samples).clip(0, 500)
        }

        df = pd.DataFrame(data)

        # Calculate total charges
        df['total_charges'] = df['monthly_charges'] * df['tenure_months']

        # Add variation to charges
        df['total_charges'] *= np.random.uniform(0.9, 1.1, n_samples)

        # Generate churn with realistic dependencies
        churn_score = (
            -0.05 * df['tenure_months'] +
            -0.5 * (df['contract_type'] == 'Two year').astype(int) +
            -0.3 * (df['contract_type'] == 'One year').astype(int) +
            0.4 * (df['payment_method'] == 'Electronic check').astype(int) +
            0.3 * (df['internet_service'] == 'Fiber optic').astype(int) +
            0.01 * df['monthly_charges'] +
            -0.15 * df['online_security'] +
            -0.15 * df['tech_support'] +
            0.2 * df['num_support_calls'] +
            0.3 * df['num_late_payments'] +
            -0.1 * df['paperless_billing'] +
            np.random.normal(0, 1, n_samples)
        )

        churn_prob = 1 / (1 + np.exp(-churn_score))
        df['churn'] = (churn_prob > 0.55).astype(int)

        return df

    def calculate_clv(self, df):
        """Calculate Customer Lifetime Value"""
        df = df.copy()

        # Average customer lifespan in months (based on churn probability)
        avg_retention_rate = 0.85  # 85% monthly retention
        discount_rate = 0.01  # 1% monthly discount rate

        # Expected lifetime = 1 / churn_rate (in months)
        expected_lifetime_months = df['tenure_months'] + (
            (1 - df['churn']) * 24  # Expected additional months if not churning
        )

        # CLV = (Average Monthly Revenue × Customer Lifespan) / (1 + Discount Rate)
        df['clv'] = (
            df['monthly_charges'] * expected_lifetime_months /
            (1 + discount_rate * expected_lifetime_months)
        )

        # Add profit margin (assume 30% margin)
        df['clv'] = df['clv'] * 0.3

        return df

    def engineer_features(self, df):
        """Create advanced customer features"""
        df = df.copy()

        # Tenure categories
        df['tenure_category'] = pd.cut(df['tenure_months'],
                                      bins=[0, 12, 24, 48, 120],
                                      labels=['new', 'medium', 'long', 'very_long'])

        # Service adoption score
        df['services_count'] = (
            df['phone_service'] + df['multiple_lines'] +
            df['online_security'] + df['online_backup'] +
            df['device_protection'] + df['tech_support'] +
            df['streaming_tv'] + df['streaming_movies']
        )

        # Revenue metrics
        df['avg_monthly_charge'] = df['total_charges'] / df['tenure_months']
        df['revenue_per_service'] = df['monthly_charges'] / (df['services_count'] + 1)

        # Engagement metrics
        df['support_intensity'] = df['num_support_calls'] / (df['tenure_months'] / 12)
        df['payment_reliability'] = 1 - (df['num_late_payments'] / (df['tenure_months'] / 3))
        df['data_usage_per_dollar'] = df['data_usage_gb'] / df['monthly_charges']

        # Risk indicators
        df['high_support_flag'] = (df['num_support_calls'] > 5).astype(int)
        df['late_payment_flag'] = (df['num_late_payments'] > 2).astype(int)
        df['month_to_month_flag'] = (df['contract_type'] == 'Month-to-month').astype(int)

        # Value segments
        df['high_value'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
        df['long_tenure'] = (df['tenure_months'] > 24).astype(int)

        return df

    def train_models(self, X, y, clv):
        """Train models with cost-sensitive learning"""
        # Split data
        X_train, X_test, y_train, y_test, clv_train, clv_test = train_test_split(
            X, y, clv, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Calculate sample weights based on CLV
        sample_weights = clv_train / clv_train.mean()

        # Train models
        models_config = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, min_samples_split=20, class_weight='balanced', random_state=42
            )
        }

        results = {}
        for name, model in models_config.items():
            # Train with sample weights for some models
            if name in ['Random Forest', 'Gradient Boosting']:
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate economic impact
            economic_impact = self.calculate_economic_impact(
                y_test, y_pred, clv_test
            )

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'economic_impact': economic_impact
            }

        return results, X_test_scaled, y_test, clv_test, X_train

    def calculate_economic_impact(self, y_true, y_pred, clv):
        """Calculate the economic value of predictions"""
        # True Positives: Correctly identified churners (saved with retention cost)
        tp_value = np.sum((y_true == 1) & (y_pred == 1) * (clv - self.retention_cost))

        # False Positives: Incorrectly flagged as churners (wasted retention cost)
        fp_value = -np.sum((y_true == 0) & (y_pred == 1) * self.retention_cost)

        # False Negatives: Missed churners (lost CLV)
        fn_value = -np.sum((y_true == 1) & (y_pred == 0) * clv)

        # True Negatives: Correctly identified non-churners (no cost, no gain)
        tn_value = 0

        total_impact = tp_value + fp_value + fn_value + tn_value

        return {
            'tp_value': tp_value,
            'fp_value': fp_value,
            'fn_value': fn_value,
            'total_impact': total_impact,
            'roi': total_impact / (np.sum(y_pred) * self.retention_cost + 1)
        }

    def plot_results(self, results, y_test, clv_test, feature_names):
        """Visualize comprehensive results"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ROC Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC={result['auc_score']:.3f})", linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curves - Churn Prediction', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Economic Impact Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        models = list(results.keys())
        impacts = [results[m]['economic_impact']['total_impact'] for m in models]
        colors = ['green' if i > 0 else 'red' for i in impacts]
        bars = ax2.barh(models, impacts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Economic Impact ($)', fontsize=11)
        ax2.set_title('Model Economic Value Comparison', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        for i, (bar, impact) in enumerate(zip(bars, impacts)):
            ax2.text(impact, bar.get_y() + bar.get_height()/2,
                    f'${impact:,.0f}', ha='left' if impact > 0 else 'right',
                    va='center', fontweight='bold', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')

        # ROI Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        rois = [results[m]['economic_impact']['roi'] * 100 for m in models]
        ax3.bar(range(len(models)), rois, color='#3498db', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('ROI (%)', fontsize=11)
        ax3.set_title('Return on Investment by Model', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.grid(True, alpha=0.3, axis='y')

        # Confusion Matrix - Best Model
        best_model_name = max(results.keys(),
                             key=lambda x: results[x]['economic_impact']['total_impact'])
        best_result = results[best_model_name]

        ax4 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax4,
                   annot_kws={'size': 14})
        ax4.set_xlabel('Predicted', fontsize=11)
        ax4.set_ylabel('Actual', fontsize=11)
        ax4.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
        ax4.set_xticklabels(['Stay', 'Churn'])
        ax4.set_yticklabels(['Stay', 'Churn'])

        # CLV Distribution by Churn Status
        ax5 = fig.add_subplot(gs[1, 1])
        churned_clv = clv_test[y_test == 1]
        stayed_clv = clv_test[y_test == 0]
        ax5.hist(stayed_clv, bins=30, alpha=0.6, label='Stayed', color='green', edgecolor='black')
        ax5.hist(churned_clv, bins=30, alpha=0.6, label='Churned', color='red', edgecolor='black')
        ax5.set_xlabel('Customer Lifetime Value ($)', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('CLV Distribution by Churn Status', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # Feature Importance
        if 'Random Forest' in results:
            ax6 = fig.add_subplot(gs[1, 2])
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(12)

            ax6.barh(range(len(feature_importance)), feature_importance['importance'],
                    color='#9b59b6', edgecolor='black')
            ax6.set_yticks(range(len(feature_importance)))
            ax6.set_yticklabels(feature_importance['feature'], fontsize=9)
            ax6.set_xlabel('Importance', fontsize=11)
            ax6.set_title('Top Feature Importances', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='x')

        # Churn Probability Distribution
        ax7 = fig.add_subplot(gs[2, 0])
        churned_probs = best_result['probabilities'][y_test == 1]
        stayed_probs = best_result['probabilities'][y_test == 0]
        ax7.hist(stayed_probs, bins=30, alpha=0.6, label='Stayed', color='green', edgecolor='black')
        ax7.hist(churned_probs, bins=30, alpha=0.6, label='Churned', color='red', edgecolor='black')
        ax7.set_xlabel('Predicted Churn Probability', fontsize=11)
        ax7.set_ylabel('Frequency', fontsize=11)
        ax7.set_title('Probability Distribution', fontsize=12, fontweight='bold')
        ax7.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')

        # Economic Breakdown
        ax8 = fig.add_subplot(gs[2, 1:])
        impact = best_result['economic_impact']
        categories = ['TP: Saved\nCustomers', 'FP: Wasted\nRetention',
                     'FN: Lost\nCustomers', 'Total\nImpact']
        values = [impact['tp_value'], impact['fp_value'],
                 impact['fn_value'], impact['total_impact']]
        colors_bar = ['green', 'orange', 'red', 'blue']

        bars = ax8.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        ax8.set_ylabel('Economic Value ($)', fontsize=11)
        ax8.set_title(f'Economic Impact Breakdown - {best_model_name}',
                     fontsize=12, fontweight='bold')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax8.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:,.0f}', ha='center',
                    va='bottom' if value > 0 else 'top',
                    fontweight='bold', fontsize=10)

        plt.savefig('telecom_churn_clv_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'telecom_churn_clv_analysis.png'")
        plt.show()

    def print_results(self, results, y_test):
        """Print detailed results"""
        print("\n" + "="*80)
        print("TELECOM CHURN PREDICTION WITH CLV ANALYSIS")
        print("="*80)

        for name, result in results.items():
            print(f"\n{'='*40}")
            print(f"Model: {name}")
            print(f"{'='*40}")
            print(f"AUC Score: {result['auc_score']:.4f}")
            print(f"\nEconomic Impact:")
            impact = result['economic_impact']
            print(f"  TP Value (Saved): ${impact['tp_value']:,.2f}")
            print(f"  FP Value (Wasted): ${impact['fp_value']:,.2f}")
            print(f"  FN Value (Lost): ${impact['fn_value']:,.2f}")
            print(f"  Total Impact: ${impact['total_impact']:,.2f}")
            print(f"  ROI: {impact['roi']*100:.2f}%")
            print(f"\nClassification Report:")
            print(classification_report(y_test, result['predictions'],
                                       target_names=['Stay', 'Churn']))

        best_model_name = max(results.keys(),
                             key=lambda x: results[x]['economic_impact']['total_impact'])
        best_impact = results[best_model_name]['economic_impact']['total_impact']
        print(f"\n{'='*80}")
        print(f"🏆 Best Model: {best_model_name} (Economic Impact: ${best_impact:,.2f})")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    print("📱 Advanced Telecom Churn Prediction with CLV")
    print("=" * 80)

    predictor = TelecomChurnPredictor()

    # Generate data
    print("\n📊 Generating telecom customer data...")
    df = predictor.create_sample_data(n_samples=5000)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")

    # Calculate CLV
    print("\n💰 Calculating Customer Lifetime Value...")
    df = predictor.calculate_clv(df)
    print(f"Average CLV: ${df['clv'].mean():,.2f}")
    print(f"Total at-risk value: ${df[df['churn']==1]['clv'].sum():,.2f}")

    # Engineer features
    print("\n🔧 Engineering features...")
    df_engineered = predictor.engineer_features(df)

    # Prepare data
    exclude_cols = ['customer_id', 'churn', 'clv', 'total_charges']
    X = df_engineered.drop(exclude_cols, axis=1)

    # Encode categorical variables
    categorical_cols = ['contract_type', 'payment_method', 'internet_service', 'tenure_category']
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    y = df_engineered['churn']
    clv = df_engineered['clv']
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training models with CLV-weighted optimization...")
    results, X_test, y_test, clv_test, X_train = predictor.train_models(X, y, clv)

    # Print results
    predictor.print_results(results, y_test)

    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_results(results, y_test, clv_test, X.columns.tolist())

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
