"""
Banking Fraud Detection System
===============================

Problem: Detect fraudulent transactions in real-time banking operations using
machine learning techniques with highly imbalanced datasets

Kaggle-style competition: Credit Card Fraud Detection
Difficulty: ⭐⭐⭐⭐

This solution demonstrates:
- Handling severely imbalanced datasets (fraud rate ~0.2%)
- SMOTE and under-sampling techniques
- Anomaly detection algorithms
- Cost-sensitive learning
- Real-time fraud scoring
- Business impact analysis (false positives vs fraud caught)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                            roc_curve, precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionSystem:
    """Advanced fraud detection with imbalanced data handling"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def create_sample_data(self, n_samples=100000):
        """Generate realistic banking transaction data with fraud patterns"""
        np.random.seed(42)

        # Normal transaction features
        normal_transactions = int(n_samples * 0.998)  # 99.8% legitimate
        fraud_transactions = n_samples - normal_transactions  # 0.2% fraud

        # Legitimate transactions
        legitimate = pd.DataFrame({
            'amount': np.random.lognormal(3.5, 1.2, normal_transactions).clip(1, 10000),
            'hour': np.random.choice(range(24), normal_transactions,
                                   p=[0.01]*6 + [0.03]*3 + [0.08]*9 + [0.05]*3 + [0.02]*3),
            'day_of_week': np.random.randint(0, 7, normal_transactions),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant',
                                                  'retail', 'online', 'travel'],
                                                 normal_transactions,
                                                 p=[0.25, 0.15, 0.20, 0.20, 0.15, 0.05]),
            'card_present': np.random.choice([0, 1], normal_transactions, p=[0.3, 0.7]),
            'foreign_transaction': np.random.choice([0, 1], normal_transactions, p=[0.95, 0.05]),
            'transaction_velocity_1h': np.random.poisson(0.5, normal_transactions),
            'transaction_velocity_24h': np.random.poisson(3, normal_transactions),
            'days_since_last_transaction': np.random.exponential(2, normal_transactions).clip(0, 30),
            'avg_transaction_amount_30d': np.random.lognormal(3.5, 0.8, normal_transactions),
            'customer_age_days': np.random.uniform(30, 3650, normal_transactions),
            'distance_from_home': np.random.exponential(5, normal_transactions).clip(0, 100),
            'is_fraud': 0
        })

        # Fraudulent transactions (different patterns)
        fraudulent = pd.DataFrame({
            'amount': np.random.lognormal(5.0, 1.5, fraud_transactions).clip(100, 5000),
            'hour': np.random.choice(range(24), fraud_transactions,
                                   p=[0.08]*6 + [0.02]*6 + [0.05]*6 + [0.08]*6),
            'day_of_week': np.random.randint(0, 7, fraud_transactions),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant',
                                                  'retail', 'online', 'travel'],
                                                 fraud_transactions,
                                                 p=[0.05, 0.05, 0.05, 0.10, 0.60, 0.15]),
            'card_present': np.random.choice([0, 1], fraud_transactions, p=[0.85, 0.15]),
            'foreign_transaction': np.random.choice([0, 1], fraud_transactions, p=[0.4, 0.6]),
            'transaction_velocity_1h': np.random.poisson(3, fraud_transactions),
            'transaction_velocity_24h': np.random.poisson(8, fraud_transactions),
            'days_since_last_transaction': np.random.exponential(0.5, fraud_transactions).clip(0, 30),
            'avg_transaction_amount_30d': np.random.lognormal(3.0, 0.6, fraud_transactions),
            'customer_age_days': np.random.uniform(1, 180, fraud_transactions),
            'distance_from_home': np.random.exponential(50, fraud_transactions).clip(10, 500),
            'is_fraud': 1
        })

        df = pd.concat([legitimate, fraudulent], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    def engineer_features(self, df):
        """Create fraud-specific features"""
        df = df.copy()

        # One-hot encode merchant category
        df = pd.get_dummies(df, columns=['merchant_category'], prefix='merchant')

        # Risk indicators
        df['high_amount'] = (df['amount'] > df['avg_transaction_amount_30d'] * 3).astype(int)
        df['unusual_hour'] = ((df['hour'] < 6) | (df['hour'] > 23)).astype(int)
        df['weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['high_velocity'] = (df['transaction_velocity_1h'] > 2).astype(int)
        df['new_customer'] = (df['customer_age_days'] < 90).astype(int)
        df['far_from_home'] = (df['distance_from_home'] > 50).astype(int)

        # Composite risk score
        df['risk_score'] = (
            df['high_amount'] * 2 +
            df['unusual_hour'] +
            df['foreign_transaction'] * 2 +
            df['high_velocity'] * 3 +
            df['new_customer'] +
            df['far_from_home'] * 2 +
            (1 - df['card_present']) * 2
        )

        # Amount ratios
        df['amount_ratio_to_avg'] = df['amount'] / (df['avg_transaction_amount_30d'] + 1)
        df['velocity_ratio'] = df['transaction_velocity_1h'] / (df['transaction_velocity_24h'] + 1)

        return df

    def train_models(self, X, y):
        """Train models with imbalanced data techniques"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"Training set fraud rate: {y_train.mean():.4%}")
        print(f"Test set fraud rate: {y_test.mean():.4%}")

        results = {}

        # 1. Logistic Regression with class weights
        print("\n Training Logistic Regression with class weights...")
        lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        lr_model.fit(X_train_scaled, y_train)

        results['Logistic (Weighted)'] = self._evaluate_model(
            lr_model, X_test_scaled, y_test, 'Logistic (Weighted)'
        )

        # 2. Random Forest with SMOTE
        print("\n Training Random Forest with SMOTE...")
        smote_pipeline = ImbPipeline([
            ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                         class_weight='balanced', random_state=42))
        ])
        smote_pipeline.fit(X_train, y_train)

        results['RF + SMOTE'] = self._evaluate_model(
            smote_pipeline, X_test, y_test, 'RF + SMOTE'
        )

        # 3. Isolation Forest (Unsupervised)
        print("\n Training Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.002, random_state=42)
        iso_forest.fit(X_train)

        # Convert to binary predictions
        anomaly_scores = iso_forest.score_samples(X_test)
        threshold = np.percentile(anomaly_scores, 0.2)  # Bottom 0.2% as fraud
        y_pred_iso = (anomaly_scores < threshold).astype(int)

        results['Isolation Forest'] = {
            'predictions': y_pred_iso,
            'probabilities': -anomaly_scores,  # Negative score as probability
            'auc_score': roc_auc_score(y_test, -anomaly_scores)
        }

        return results, X_test, y_test, X_train

    def _evaluate_model(self, model, X_test, y_test, name):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)

        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = y_pred

        return {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba)
        }

    def calculate_business_impact(self, y_test, y_pred, y_proba, avg_fraud_amount=500):
        """Calculate financial impact of fraud detection"""
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Business metrics
        fraud_caught = tp
        fraud_missed = fn
        false_alarms = fp

        # Financial calculations (example values)
        investigation_cost = 25  # Cost to investigate each alert
        fraud_loss = avg_fraud_amount  # Average fraud amount

        money_saved = fraud_caught * fraud_loss
        investigation_costs = (fraud_caught + false_alarms) * investigation_cost
        fraud_losses = fraud_missed * fraud_loss
        net_benefit = money_saved - investigation_costs - fraud_losses

        return {
            'fraud_caught': fraud_caught,
            'fraud_missed': fraud_missed,
            'false_alarms': false_alarms,
            'money_saved': money_saved,
            'investigation_costs': investigation_costs,
            'fraud_losses': fraud_losses,
            'net_benefit': net_benefit,
            'catch_rate': fraud_caught / (fraud_caught + fraud_missed) if (fraud_caught + fraud_missed) > 0 else 0
        }

    def plot_results(self, results, y_test, feature_names):
        """Visualize comprehensive fraud detection results"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ROC Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC={result['auc_score']:.3f})", linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curves - Fraud Detection', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curves
        ax2 = fig.add_subplot(gs[0, 1])
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            ax2.plot(recall, precision,
                    label=f"{name} (AP={result['avg_precision']:.3f})", linewidth=2)
        ax2.set_xlabel('Recall', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_result = results[best_model_name]

        ax3 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax3, cbar=True)
        ax3.set_xlabel('Predicted', fontsize=11)
        ax3.set_ylabel('Actual', fontsize=11)
        ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(['Legitimate', 'Fraud'])
        ax3.set_yticklabels(['Legitimate', 'Fraud'])

        # Business Impact Analysis
        ax4 = fig.add_subplot(gs[1, 0])
        impact = self.calculate_business_impact(y_test, best_result['predictions'],
                                               best_result['probabilities'])

        metrics = ['Fraud\nCaught', 'Fraud\nMissed', 'False\nAlarms']
        values = [impact['fraud_caught'], impact['fraud_missed'], impact['false_alarms']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']

        bars = ax4.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Fraud Detection Performance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Financial Impact
        ax5 = fig.add_subplot(gs[1, 1])
        financial_metrics = ['Money\nSaved', 'Investigation\nCosts', 'Fraud\nLosses', 'Net\nBenefit']
        financial_values = [impact['money_saved']/1000, impact['investigation_costs']/1000,
                          impact['fraud_losses']/1000, impact['net_benefit']/1000]
        fin_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']

        bars = ax5.bar(financial_metrics, financial_values, color=fin_colors,
                      edgecolor='black', linewidth=1.5)
        ax5.set_ylabel('Amount ($1000s)', fontsize=11)
        ax5.set_title('Financial Impact Analysis', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)

        for bar, value in zip(bars, financial_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:.0f}K', ha='center',
                    va='bottom' if value >= 0 else 'top', fontsize=9, fontweight='bold')

        # Score Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        legitimate_scores = best_result['probabilities'][y_test == 0]
        fraud_scores = best_result['probabilities'][y_test == 1]

        ax6.hist(legitimate_scores, bins=50, alpha=0.6, label='Legitimate',
                color='green', edgecolor='black', density=True)
        ax6.hist(fraud_scores, bins=50, alpha=0.6, label='Fraud',
                color='red', edgecolor='black', density=True)
        ax6.set_xlabel('Fraud Score', fontsize=11)
        ax6.set_ylabel('Density', fontsize=11)
        ax6.set_title('Fraud Score Distribution', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        # Model Comparison
        ax7 = fig.add_subplot(gs[2, 0])
        model_names = list(results.keys())
        auc_scores = [results[m]['auc_score'] for m in model_names]
        ap_scores = [results[m]['avg_precision'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        ax7.bar(x - width/2, auc_scores, width, label='ROC AUC', color='#3498db')
        ax7.bar(x + width/2, ap_scores, width, label='Avg Precision', color='#e74c3c')
        ax7.set_ylabel('Score', fontsize=11)
        ax7.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_ylim(0, 1.0)

        # Summary Statistics
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        summary_text = f"""
        ╔═══════════════════════════════════════════════════════════════════╗
        ║              FRAUD DETECTION SYSTEM - PERFORMANCE SUMMARY          ║
        ╚═══════════════════════════════════════════════════════════════════╝

        Best Model: {best_model_name}
        ROC AUC: {best_result['auc_score']:.4f}  |  Avg Precision: {best_result['avg_precision']:.4f}

        ┌─────────────────────────────────────────────────────────────────┐
        │ DETECTION METRICS                                                │
        ├─────────────────────────────────────────────────────────────────┤
        │ Precision:        {precision:6.2%}   │  Recall:        {recall:6.2%} │
        │ F1-Score:         {f1:6.2%}   │  Accuracy:      {((tp+tn)/(tp+tn+fp+fn)):6.2%} │
        │                                                                  │
        │ True Positives:   {tp:6d}   │  False Negatives: {fn:6d}       │
        │ False Positives:  {fp:6d}   │  True Negatives:  {tn:6d}       │
        └─────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────┐
        │ BUSINESS IMPACT                                                  │
        ├─────────────────────────────────────────────────────────────────┤
        │ Fraud Catch Rate:        {impact['catch_rate']:6.2%}                       │
        │ Money Saved:             ${impact['money_saved']:>12,.0f}                │
        │ Investigation Costs:     ${impact['investigation_costs']:>12,.0f}                │
        │ Fraud Losses:            ${impact['fraud_losses']:>12,.0f}                │
        │ NET BENEFIT:             ${impact['net_benefit']:>12,.0f}                │
        └─────────────────────────────────────────────────────────────────┘
        """
        ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.savefig('fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'fraud_detection_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("🔒 Banking Fraud Detection System")
    print("=" * 80)

    detector = FraudDetectionSystem()

    # Generate data
    print("\n📊 Generating transaction data...")
    df = detector.create_sample_data(n_samples=100000)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    print(f"Total fraud cases: {df['is_fraud'].sum()}")

    # Engineer features
    print("\n🔧 Engineering fraud detection features...")
    df_engineered = detector.engineer_features(df)

    # Prepare data
    X = df_engineered.drop('is_fraud', axis=1)
    y = df_engineered['is_fraud']
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training fraud detection models...")
    results, X_test, y_test, X_train = detector.train_models(X, y)

    # Plot results
    print("\n📈 Generating visualizations...")
    detector.plot_results(results, y_test, X.columns.tolist())

    print("\n✅ Fraud detection analysis complete!")


if __name__ == "__main__":
    main()
