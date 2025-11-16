"""
Subscription Renewal Prediction
================================

Problem: Predict whether subscribers will renew their subscription and identify
drivers of retention for SaaS and subscription businesses

Kaggle-style competition: Subscription Retention Optimization
Difficulty: ⭐⭐

This solution demonstrates:
- Subscription lifecycle modeling
- Cohort analysis
- Feature usage tracking
- Customer health scoring
- Renewal likelihood prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SubscriptionRetentionPredictor:
    """Predicts subscription renewal and customer health"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def create_sample_data(self, n_samples=7000):
        """Generate realistic subscription customer data"""
        np.random.seed(42)

        # Customer characteristics
        data = {
            'customer_id': range(1, n_samples + 1),
            'subscription_tier': np.random.choice(['Basic', 'Professional', 'Enterprise'],
                                                 n_samples, p=[0.5, 0.35, 0.15]),
            'tenure_months': np.random.exponential(18, n_samples).clip(1, 60),
            'monthly_price': None,  # Will be set based on tier
            'total_paid': None,  # Will be calculated
            'payment_method': np.random.choice(['credit_card', 'paypal', 'bank_transfer', 'invoice'],
                                              n_samples, p=[0.5, 0.25, 0.15, 0.1]),
            'contract_type': np.random.choice(['monthly', 'annual', 'multi_year'],
                                             n_samples, p=[0.6, 0.35, 0.05]),
            'auto_renew': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),

            # Usage metrics
            'monthly_active_days': np.random.binomial(30, 0.4, n_samples),
            'avg_session_duration_min': np.random.lognormal(2.5, 1, n_samples).clip(1, 120),
            'features_used': np.random.poisson(8, n_samples) + 1,
            'total_features_available': None,  # Based on tier
            'api_calls_per_month': np.random.lognormal(6, 2, n_samples).clip(0, 100000),
            'storage_used_gb': np.random.lognormal(2, 1.5, n_samples).clip(0, 1000),
            'storage_limit_gb': None,  # Based on tier
            'users_count': np.random.poisson(5, n_samples) + 1,
            'seats_purchased': None,  # Will be calculated

            # Engagement metrics
            'logins_per_month': np.random.poisson(15, n_samples),
            'support_tickets': np.random.poisson(2, n_samples),
            'feature_requests': np.random.poisson(0.5, n_samples),
            'nps_score': np.random.choice(range(0, 11), n_samples),
            'training_completed': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'integration_count': np.random.poisson(3, n_samples),
            'custom_workflows': np.random.poisson(2, n_samples),

            # Interaction history
            'support_response_time_hours': np.random.exponential(12, n_samples).clip(0.5, 72),
            'num_upgrades': np.random.poisson(0.3, n_samples),
            'num_downgrades': np.random.poisson(0.1, n_samples),
            'payment_failures': np.random.poisson(0.5, n_samples),
            'late_payments': np.random.poisson(0.3, n_samples),
            'discount_percentage': np.random.choice([0, 10, 20, 30], n_samples, p=[0.6, 0.2, 0.15, 0.05]),

            # Communication
            'email_opens_rate': np.random.beta(2, 3, n_samples),
            'email_click_rate': np.random.beta(1.5, 5, n_samples),
            'webinar_attendance': np.random.poisson(1, n_samples),
            'community_posts': np.random.poisson(0.5, n_samples),
            'referrals_made': np.random.poisson(0.4, n_samples),

            # Time-based
            'days_until_renewal': np.random.randint(1, 90, n_samples),
            'last_login_days_ago': np.random.exponential(10, n_samples).clip(0, 90)
        }

        df = pd.DataFrame(data)

        # Set tier-based attributes
        tier_config = {
            'Basic': {'price': 29, 'features': 15, 'storage': 10},
            'Professional': {'price': 99, 'features': 30, 'storage': 100},
            'Enterprise': {'price': 299, 'features': 50, 'storage': 1000}
        }

        df['monthly_price'] = df['subscription_tier'].map(lambda x: tier_config[x]['price'])
        df['total_features_available'] = df['subscription_tier'].map(lambda x: tier_config[x]['features'])
        df['storage_limit_gb'] = df['subscription_tier'].map(lambda x: tier_config[x]['storage'])

        # Calculate seats and total paid
        df['seats_purchased'] = np.maximum(df['users_count'], 1)
        df['total_paid'] = df['monthly_price'] * df['tenure_months'] * (1 - df['discount_percentage']/100)

        # Generate renewal with realistic dependencies
        renewal_score = (
            0.3 * (df['subscription_tier'] == 'Enterprise').astype(int) +
            0.15 * (df['subscription_tier'] == 'Professional').astype(int) +
            0.02 * df['tenure_months'] +
            0.4 * (df['contract_type'] == 'multi_year').astype(int) +
            0.25 * (df['contract_type'] == 'annual').astype(int) +
            0.5 * df['auto_renew'] +
            0.03 * df['monthly_active_days'] +
            0.01 * np.log1p(df['api_calls_per_month']) +
            0.02 * df['features_used'] +
            0.02 * df['logins_per_month'] +
            -0.15 * df['support_tickets'] +
            0.1 * df['feature_requests'] +
            0.08 * df['nps_score'] +
            0.3 * df['training_completed'] +
            0.1 * df['integration_count'] +
            0.15 * df['custom_workflows'] +
            0.2 * df['num_upgrades'] +
            -0.4 * df['num_downgrades'] +
            -0.3 * df['payment_failures'] +
            -0.2 * df['late_payments'] +
            -0.02 * df['discount_percentage'] +
            0.3 * df['email_opens_rate'] +
            0.2 * df['webinar_attendance'] +
            0.15 * df['referrals_made'] +
            -0.02 * df['last_login_days_ago'] +
            -0.01 * df['days_until_renewal'] +
            np.random.normal(0, 0.8, n_samples)
        )

        # Convert to probability
        renewal_prob = 1 / (1 + np.exp(-renewal_score))
        df['renewed'] = (renewal_prob > 0.6).astype(int)

        return df

    def engineer_features(self, df):
        """Create advanced subscription and engagement features"""
        df = df.copy()

        # Engagement intensity
        df['engagement_score'] = (
            (df['monthly_active_days'] / 30) * 0.3 +
            (df['logins_per_month'] / 30) * 0.3 +
            (df['features_used'] / df['total_features_available']) * 0.4
        ) * 100

        df['power_user'] = (
            (df['monthly_active_days'] > 20) &
            (df['logins_per_month'] > 20) &
            (df['features_used'] > df['total_features_available'] * 0.6)
        ).astype(int)

        # Feature adoption
        df['feature_adoption_rate'] = df['features_used'] / df['total_features_available']
        df['low_adoption'] = (df['feature_adoption_rate'] < 0.3).astype(int)
        df['high_adoption'] = (df['feature_adoption_rate'] > 0.7).astype(int)

        # Storage utilization
        df['storage_utilization'] = df['storage_used_gb'] / df['storage_limit_gb']
        df['approaching_limit'] = (df['storage_utilization'] > 0.8).astype(int)
        df['underutilized'] = (df['storage_utilization'] < 0.2).astype(int)

        # Usage per seat
        df['api_calls_per_user'] = df['api_calls_per_month'] / df['users_count']
        df['logins_per_user'] = df['logins_per_month'] / df['users_count']
        df['usage_per_seat'] = (df['api_calls_per_user'] + df['logins_per_user']) / 2

        # Customer health score
        df['health_score'] = (
            (df['engagement_score'] / 100) * 0.25 +
            df['feature_adoption_rate'] * 0.2 +
            (df['nps_score'] / 10) * 0.2 +
            (1 - df['support_tickets'] / 10).clip(0, 1) * 0.15 +
            df['auto_renew'] * 0.1 +
            df['training_completed'] * 0.1
        ) * 100

        # Risk indicators
        df['at_risk'] = (
            (df['health_score'] < 50) |
            (df['payment_failures'] > 0) |
            (df['num_downgrades'] > 0) |
            (df['last_login_days_ago'] > 30)
        ).astype(int)

        df['inactive_user'] = (
            (df['monthly_active_days'] < 5) |
            (df['last_login_days_ago'] > 14)
        ).astype(int)

        # Tenure categories
        df['tenure_category'] = pd.cut(df['tenure_months'],
                                       bins=[0, 6, 12, 24, 100],
                                       labels=['new', 'established', 'mature', 'veteran'])

        # Payment reliability
        df['payment_score'] = (
            (1 - (df['payment_failures'] > 0).astype(int)) * 0.5 +
            (1 - (df['late_payments'] > 0).astype(int)) * 0.3 +
            (df['payment_method'] == 'credit_card').astype(int) * 0.2
        ) * 100

        # Communication engagement
        df['communication_score'] = (
            df['email_opens_rate'] * 0.3 +
            df['email_click_rate'] * 0.3 +
            (df['webinar_attendance'] > 0).astype(int) * 0.2 +
            (df['community_posts'] > 0).astype(int) * 0.2
        ) * 100

        # Product stickiness
        df['stickiness_score'] = (
            df['integration_count'] * 10 +
            df['custom_workflows'] * 15 +
            (df['users_count'] > 1).astype(int) * 20 +
            df['training_completed'] * 25
        )

        # Value perception
        df['value_score'] = (
            np.log1p(df['api_calls_per_month']) / np.log1p(df['monthly_price']) +
            df['feature_adoption_rate'] * 2 +
            (df['nps_score'] / 10)
        ) * 100

        # Contract commitment
        df['committed_customer'] = (
            (df['contract_type'] == 'annual') |
            (df['contract_type'] == 'multi_year')
        ).astype(int)

        df['long_term_value'] = (
            df['committed_customer'] * 0.3 +
            (df['tenure_months'] > 12).astype(int) * 0.3 +
            (df['subscription_tier'] != 'Basic').astype(int) * 0.4
        ) * 100

        # Recency and urgency
        df['renewal_urgency'] = (df['days_until_renewal'] < 30).astype(int)
        df['recently_active'] = (df['last_login_days_ago'] < 7).astype(int)

        return df

    def train_models(self, X, y):
        """Train multiple retention models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize models
        models_config = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced', C=0.4, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=22, min_samples_split=10,
                class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=8,
                subsample=0.8, random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=12, min_samples_split=20,
                class_weight='balanced', random_state=42
            )
        }

        results = {}
        for name, model in models_config.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='roc_auc'
            )

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

        return results, X_test_scaled, y_test, X_train

    def plot_results(self, results, y_test, feature_names):
        """Visualize comprehensive retention analysis"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # ROC Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f"{name}\n(AUC={result['auc_score']:.3f})",
                    linewidth=2.5)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curves - Subscription Renewal', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Model Performance
        ax2 = fig.add_subplot(gs[0, 1])
        models = list(results.keys())
        auc_scores = [results[m]['auc_score'] for m in models]
        cv_means = [results[m]['cv_mean'] for m in models]

        x = np.arange(len(models))
        width = 0.35
        bars1 = ax2.bar(x - width/2, auc_scores, width, label='Test AUC',
                       color='#3498db', edgecolor='black', alpha=0.8)
        bars2 = ax2.bar(x + width/2, cv_means, width, label='CV Mean',
                       color='#2ecc71', edgecolor='black', alpha=0.8)

        ax2.set_ylabel('AUC Score', fontsize=12)
        ax2.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0.5, 1.0)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Confusion Matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_result = results[best_model_name]

        ax3 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax3,
                   annot_kws={'size': 14}, cbar_kws={'label': 'Count'})
        ax3.set_xlabel('Predicted', fontsize=12)
        ax3.set_ylabel('Actual', fontsize=12)
        ax3.set_title(f'Confusion Matrix - {best_model_name}',
                     fontsize=13, fontweight='bold')
        ax3.set_xticklabels(['Churn', 'Renew'])
        ax3.set_yticklabels(['Churn', 'Renew'])

        # Renewal Probability Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        renewed_probs = best_result['probabilities'][y_test == 1]
        churned_probs = best_result['probabilities'][y_test == 0]

        ax4.hist(churned_probs, bins=40, alpha=0.65, label='Churned',
                color='red', edgecolor='black')
        ax4.hist(renewed_probs, bins=40, alpha=0.65, label='Renewed',
                color='green', edgecolor='black')
        ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                   label='Decision Threshold')
        ax4.set_xlabel('Predicted Renewal Probability', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Probability Distribution by Outcome', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        # Feature Importance
        if 'Random Forest' in results:
            ax5 = fig.add_subplot(gs[1, 1])
            rf_model = results['Random Forest']['model']

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
            ax5.barh(range(len(feature_importance)), feature_importance['importance'],
                    color=colors_feat, edgecolor='black')
            ax5.set_yticks(range(len(feature_importance)))
            ax5.set_yticklabels(feature_importance['feature'], fontsize=9)
            ax5.set_xlabel('Importance', fontsize=12)
            ax5.set_title('Top 15 Retention Drivers', fontsize=13, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')

        # Customer Health Score Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        # This would need access to original data, so we'll create a simplified version
        ax6.text(0.5, 0.5, 'Customer Segmentation\n\n' +
                '● High Health (>70): Retention Focus\n' +
                '● Medium Health (40-70): Engagement\n' +
                '● Low Health (<40): Intervention\n\n' +
                'Risk Factors:\n' +
                '- Low engagement score\n' +
                '- Payment issues\n' +
                '- Inactive usage\n' +
                '- No integrations',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Customer Health Framework', fontsize=13, fontweight='bold')

        # Cross-Validation Scores
        ax7 = fig.add_subplot(gs[2, 0])
        cv_means_plot = [results[m]['cv_mean'] for m in models]
        cv_stds = [results[m]['cv_std'] for m in models]

        ax7.bar(range(len(models)), cv_means_plot, yerr=cv_stds, capsize=5,
               color='#e74c3c', alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(len(models)))
        ax7.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax7.set_ylabel('Cross-Validation AUC', fontsize=12)
        ax7.set_title('Model Stability (5-Fold CV)', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_ylim(0, 1)

        # Churn Risk Segments
        ax8 = fig.add_subplot(gs[2, 1])
        risk_bins = [0, 0.3, 0.5, 0.7, 1.0]
        risk_labels = ['High Risk\n(Churn)', 'Medium-High\nRisk', 'Medium-Low\nRisk', 'Low Risk\n(Renew)']

        risk_categories = pd.cut(best_result['probabilities'], bins=risk_bins, labels=risk_labels)
        risk_counts = risk_categories.value_counts().sort_index()

        colors_risk = ['darkred', 'orange', 'lightgreen', 'darkgreen']
        bars_risk = ax8.bar(range(len(risk_counts)), risk_counts.values,
                           color=colors_risk, alpha=0.7, edgecolor='black')
        ax8.set_xticks(range(len(risk_counts)))
        ax8.set_xticklabels(risk_counts.index, rotation=0, fontsize=10)
        ax8.set_ylabel('Number of Customers', fontsize=12)
        ax8.set_title('Customer Risk Distribution', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars_risk, risk_counts.values):
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count}', ha='center', va='bottom', fontweight='bold')

        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        summary_text = f"""
╔════════════════════════════════════════╗
║   SUBSCRIPTION RETENTION SUMMARY       ║
╚════════════════════════════════════════╝

Best Model: {best_model_name}

Performance:
  AUC:        {best_result['auc_score']:.4f}
  CV Mean:    {best_result['cv_mean']:.4f}
  CV Std:     {best_result['cv_std']:.4f}

Classification:
  Accuracy:   {accuracy:.4f}
  Precision:  {precision:.4f}
  Recall:     {recall:.4f}
  F1-Score:   {f1:.4f}

Confusion Matrix:
  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}
  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}

Renewal Rate: {y_test.mean():.1%}
        """

        ax9.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='lightcyan', alpha=0.3))

        plt.savefig('subscription_retention_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'subscription_retention_analysis.png'")
        plt.show()

    def print_results(self, results, y_test):
        """Print detailed retention results"""
        print("\n" + "="*80)
        print("SUBSCRIPTION RENEWAL PREDICTION RESULTS")
        print("="*80)

        for name, result in results.items():
            print(f"\n{'='*40}")
            print(f"Model: {name}")
            print(f"{'='*40}")
            print(f"AUC Score: {result['auc_score']:.4f}")
            print(f"CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
            print(f"\nClassification Report:")
            print(classification_report(y_test, result['predictions'],
                                       target_names=['Churn', 'Renew']))

        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_auc = results[best_model_name]['auc_score']
        print(f"\n{'='*80}")
        print(f"🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    print("📊 Subscription Renewal Prediction System")
    print("=" * 80)

    predictor = SubscriptionRetentionPredictor()

    # Generate data
    print("\n📊 Generating subscription customer data...")
    df = predictor.create_sample_data(n_samples=7000)
    print(f"Dataset shape: {df.shape}")
    print(f"Renewal rate: {df['renewed'].mean():.2%}")
    print(f"Average tenure: {df['tenure_months'].mean():.1f} months")

    # Engineer features
    print("\n🔧 Engineering retention features...")
    df_engineered = predictor.engineer_features(df)
    print(f"Average health score: {df_engineered['health_score'].mean():.1f}")
    print(f"At-risk customers: {df_engineered['at_risk'].mean():.1%}")

    # Prepare data
    exclude_cols = ['customer_id', 'renewed', 'tenure_category', 'monthly_price',
                   'total_paid', 'total_features_available', 'storage_limit_gb', 'seats_purchased']
    X = df_engineered.drop(exclude_cols, axis=1)

    # Encode categorical variables
    categorical_cols = ['subscription_tier', 'payment_method', 'contract_type']
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            predictor.label_encoders[col] = le

    y = df_engineered['renewed']
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training renewal prediction models...")
    results, X_test, y_test, X_train = predictor.train_models(X, y)

    # Print results
    predictor.print_results(results, y_test)

    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_results(results, y_test, X.columns.tolist())

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
