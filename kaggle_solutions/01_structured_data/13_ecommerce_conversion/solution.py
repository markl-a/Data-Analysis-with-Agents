"""
E-Commerce Purchase Conversion Prediction
=========================================

Problem: Predict whether a website visitor will complete a purchase based on
browsing behavior, session characteristics, and user attributes

Kaggle-style competition: E-Commerce Conversion Optimization
Difficulty: ⭐⭐

This solution demonstrates:
- Conversion funnel analysis
- Behavioral feature engineering
- Time-series session features
- A/B test simulation
- Conversion rate optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class EcommerceConversionPredictor:
    """Predicts purchase conversion from browsing sessions"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def create_sample_data(self, n_samples=8000):
        """Generate realistic e-commerce session data"""
        np.random.seed(42)

        # Session characteristics
        data = {
            'session_id': range(1, n_samples + 1),
            'user_type': np.random.choice(['new', 'returning', 'loyal'],
                                         n_samples, p=[0.4, 0.4, 0.2]),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'],
                                           n_samples, p=[0.6, 0.3, 0.1]),
            'traffic_source': np.random.choice(['organic', 'paid_search', 'social',
                                               'direct', 'referral', 'email'],
                                              n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05]),
            'page_views': np.random.poisson(5, n_samples) + 1,
            'time_on_site_minutes': np.random.exponential(8, n_samples).clip(0.5, 120),
            'product_views': np.random.poisson(3, n_samples),
            'add_to_cart_count': np.random.poisson(0.8, n_samples),
            'cart_abandonment': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'wishlist_adds': np.random.poisson(0.3, n_samples),
            'search_count': np.random.poisson(1.5, n_samples),
            'filter_uses': np.random.poisson(2, n_samples),
            'coupon_viewed': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'reviews_read': np.random.poisson(1.2, n_samples),
            'product_comparisons': np.random.poisson(0.5, n_samples),
            'avg_product_price': np.random.lognormal(4, 1, n_samples).clip(10, 500),
            'session_hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.71, 0.29]),
            'previous_purchases': np.random.poisson(2, n_samples),
            'days_since_last_visit': np.random.exponential(15, n_samples).clip(0, 365),
            'email_subscriber': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'mobile_app_user': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'loyalty_member': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'bounce_rate': np.random.uniform(0, 1, n_samples),
            'exit_rate': np.random.uniform(0, 1, n_samples)
        }

        df = pd.DataFrame(data)

        # Generate conversion with realistic dependencies
        conversion_score = (
            0.5 * (df['user_type'] == 'loyal').astype(int) +
            0.3 * (df['user_type'] == 'returning').astype(int) +
            -0.2 * (df['device_type'] == 'mobile').astype(int) +
            0.3 * (df['traffic_source'] == 'email').astype(int) +
            0.2 * (df['traffic_source'] == 'direct').astype(int) +
            -0.3 * (df['traffic_source'] == 'social').astype(int) +
            0.05 * np.log1p(df['page_views']) +
            0.08 * np.log1p(df['time_on_site_minutes']) +
            0.4 * df['add_to_cart_count'] +
            -0.6 * df['cart_abandonment'] +
            0.3 * df['wishlist_adds'] +
            0.2 * df['coupon_viewed'] +
            0.15 * df['reviews_read'] +
            0.1 * df['product_comparisons'] +
            0.1 * np.log1p(df['previous_purchases']) +
            -0.01 * df['days_since_last_visit'] +
            0.3 * df['email_subscriber'] +
            0.2 * df['mobile_app_user'] +
            0.4 * df['loyalty_member'] +
            -0.5 * df['bounce_rate'] +
            -0.4 * df['exit_rate'] +
            np.random.normal(0, 0.5, n_samples)
        )

        # Convert to probability
        conversion_prob = 1 / (1 + np.exp(-conversion_score))
        df['converted'] = (conversion_prob > 0.7).astype(int)

        # Add revenue for converted sessions
        df['revenue'] = 0.0
        df.loc[df['converted'] == 1, 'revenue'] = (
            df.loc[df['converted'] == 1, 'avg_product_price'] *
            np.random.uniform(1, 3, (df['converted'] == 1).sum())
        )

        return df

    def engineer_features(self, df):
        """Create advanced session and behavioral features"""
        df = df.copy()

        # Engagement metrics
        df['engagement_score'] = (
            df['page_views'] * 0.2 +
            df['time_on_site_minutes'] * 0.3 +
            df['product_views'] * 0.3 +
            df['reviews_read'] * 0.2
        )

        df['pages_per_minute'] = df['page_views'] / (df['time_on_site_minutes'] + 0.1)
        df['avg_time_per_page'] = df['time_on_site_minutes'] / (df['page_views'] + 0.1)

        # Purchase intent signals
        df['purchase_intent'] = (
            df['add_to_cart_count'] * 3 +
            df['wishlist_adds'] * 2 +
            df['product_comparisons'] * 1.5 +
            df['coupon_viewed'] * 1
        )

        df['cart_completion_rate'] = np.where(
            df['add_to_cart_count'] > 0,
            1 - df['cart_abandonment'],
            0
        )

        # Search and discovery
        df['search_intensity'] = df['search_count'] / (df['time_on_site_minutes'] + 0.1)
        df['filter_usage_rate'] = df['filter_uses'] / (df['product_views'] + 1)

        # User quality metrics
        df['user_quality_score'] = (
            (df['user_type'] == 'loyal').astype(int) * 3 +
            (df['user_type'] == 'returning').astype(int) * 2 +
            df['previous_purchases'] * 0.5 +
            df['email_subscriber'] * 1 +
            df['loyalty_member'] * 2
        )

        df['recency_score'] = 1 / (df['days_since_last_visit'] + 1)

        # Device and channel quality
        df['high_intent_device'] = (df['device_type'] == 'desktop').astype(int)
        df['high_quality_traffic'] = (
            (df['traffic_source'].isin(['direct', 'email', 'organic'])).astype(int)
        )

        # Time-based features
        df['is_business_hours'] = (
            (df['session_hour'] >= 9) & (df['session_hour'] <= 17)
        ).astype(int)

        df['is_evening'] = (
            (df['session_hour'] >= 18) & (df['session_hour'] <= 22)
        ).astype(int)

        # Funnel progression
        df['reached_cart'] = (df['add_to_cart_count'] > 0).astype(int)
        df['deep_engagement'] = (
            (df['page_views'] > 5) & (df['time_on_site_minutes'] > 5)
        ).astype(int)

        df['product_exploration'] = (
            df['product_views'] > df['page_views'] * 0.5
        ).astype(int)

        # Interaction quality
        df['quality_session'] = (
            (df['bounce_rate'] < 0.3) &
            (df['time_on_site_minutes'] > 2) &
            (df['page_views'] > 3)
        ).astype(int)

        return df

    def train_models(self, X, y):
        """Train multiple conversion models"""
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
                max_iter=1000, class_weight='balanced', C=0.5, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=20,
                class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=6,
                subsample=0.8, random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5, random_state=42
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
                model, X_train_scaled, y_train,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring='roc_auc'
            )

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'avg_precision': average_precision_score(y_test, y_pred_proba),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

        return results, X_test_scaled, y_test, X_train

    def plot_results(self, results, y_test, feature_names):
        """Visualize comprehensive conversion analysis"""
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
        ax1.set_title('ROC Curves - Conversion Prediction', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curves
        ax2 = fig.add_subplot(gs[0, 1])
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            ax2.plot(recall, precision,
                    label=f"{name}\n(AP={result['avg_precision']:.3f})",
                    linewidth=2.5)
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Model Performance Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        models = list(results.keys())
        auc_scores = [results[m]['auc_score'] for m in models]
        ap_scores = [results[m]['avg_precision'] for m in models]

        x = np.arange(len(models))
        width = 0.35
        bars1 = ax3.bar(x - width/2, auc_scores, width, label='ROC-AUC',
                       color='#3498db', edgecolor='black', alpha=0.8)
        bars2 = ax3.bar(x + width/2, ap_scores, width, label='Avg Precision',
                       color='#2ecc71', edgecolor='black', alpha=0.8)

        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Model Performance Metrics', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Confusion Matrix - Best Model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_result = results[best_model_name]

        ax4 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   annot_kws={'size': 14}, cbar_kws={'label': 'Count'})
        ax4.set_xlabel('Predicted', fontsize=12)
        ax4.set_ylabel('Actual', fontsize=12)
        ax4.set_title(f'Confusion Matrix - {best_model_name}',
                     fontsize=13, fontweight='bold')
        ax4.set_xticklabels(['No Purchase', 'Purchase'])
        ax4.set_yticklabels(['No Purchase', 'Purchase'])

        # Conversion Probability Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        converted_probs = best_result['probabilities'][y_test == 1]
        not_converted_probs = best_result['probabilities'][y_test == 0]

        ax5.hist(not_converted_probs, bins=40, alpha=0.65, label='No Purchase',
                color='red', edgecolor='black')
        ax5.hist(converted_probs, bins=40, alpha=0.65, label='Purchase',
                color='green', edgecolor='black')
        ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                   label='Decision Threshold')
        ax5.set_xlabel('Predicted Conversion Probability', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('Probability Distribution by Outcome', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')

        # Feature Importance
        if 'Random Forest' in results:
            ax6 = fig.add_subplot(gs[1, 2])
            rf_model = results['Random Forest']['model']

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
            ax6.barh(range(len(feature_importance)), feature_importance['importance'],
                    color=colors_feat, edgecolor='black')
            ax6.set_yticks(range(len(feature_importance)))
            ax6.set_yticklabels(feature_importance['feature'], fontsize=9)
            ax6.set_xlabel('Importance', fontsize=12)
            ax6.set_title('Top 15 Feature Importances', fontsize=13, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='x')

        # Cross-Validation Scores
        ax7 = fig.add_subplot(gs[2, 0])
        cv_means = [results[m]['cv_mean'] for m in models]
        cv_stds = [results[m]['cv_std'] for m in models]

        ax7.bar(range(len(models)), cv_means, yerr=cv_stds, capsize=5,
               color='#e74c3c', alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(len(models)))
        ax7.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax7.set_ylabel('Cross-Validation AUC', fontsize=12)
        ax7.set_title('Model Stability (5-Fold CV)', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_ylim(0, 1)

        # Conversion Rate by Predicted Probability Bins
        ax8 = fig.add_subplot(gs[2, 1])
        prob_bins = np.linspace(0, 1, 11)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
        actual_rates = []

        for i in range(len(prob_bins) - 1):
            mask = (best_result['probabilities'] >= prob_bins[i]) & \
                   (best_result['probabilities'] < prob_bins[i+1])
            if mask.sum() > 0:
                actual_rates.append(y_test[mask].mean())
            else:
                actual_rates.append(0)

        ax8.plot(bin_centers, actual_rates, 'o-', linewidth=2.5, markersize=8,
                color='#9b59b6', label='Actual Rate')
        ax8.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Perfect Calibration')
        ax8.set_xlabel('Predicted Probability', fontsize=12)
        ax8.set_ylabel('Actual Conversion Rate', fontsize=12)
        ax8.set_title('Calibration Plot', fontsize=13, fontweight='bold')
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)

        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        summary_text = f"""
╔════════════════════════════════════════╗
║   CONVERSION PREDICTION SUMMARY        ║
╚════════════════════════════════════════╝

Best Model: {best_model_name}

Performance Metrics:
  ROC-AUC:       {best_result['auc_score']:.4f}
  Avg Precision: {best_result['avg_precision']:.4f}
  CV Mean:       {best_result['cv_mean']:.4f}
  CV Std:        {best_result['cv_std']:.4f}

Classification Metrics:
  Accuracy:      {accuracy:.4f}
  Precision:     {precision:.4f}
  Recall:        {recall:.4f}
  F1-Score:      {f1:.4f}

Confusion Matrix:
  True Neg:  {cm[0,0]:6d}  False Pos: {cm[0,1]:6d}
  False Neg: {cm[1,0]:6d}  True Pos:  {cm[1,1]:6d}

Conversion Insights:
  Baseline Rate: {y_test.mean():.2%}
  Predicted High-Risk: {(best_result['probabilities'] > 0.7).mean():.2%}
        """

        ax9.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.3))

        plt.savefig('ecommerce_conversion_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'ecommerce_conversion_analysis.png'")
        plt.show()

    def print_results(self, results, y_test):
        """Print detailed results"""
        print("\n" + "="*80)
        print("E-COMMERCE CONVERSION PREDICTION RESULTS")
        print("="*80)

        for name, result in results.items():
            print(f"\n{'='*40}")
            print(f"Model: {name}")
            print(f"{'='*40}")
            print(f"ROC-AUC Score: {result['auc_score']:.4f}")
            print(f"Average Precision: {result['avg_precision']:.4f}")
            print(f"CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
            print(f"\nClassification Report:")
            print(classification_report(y_test, result['predictions'],
                                       target_names=['No Purchase', 'Purchase']))

        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_auc = results[best_model_name]['auc_score']
        print(f"\n{'='*80}")
        print(f"🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    print("🛒 E-Commerce Purchase Conversion Prediction")
    print("=" * 80)

    predictor = EcommerceConversionPredictor()

    # Generate data
    print("\n📊 Generating e-commerce session data...")
    df = predictor.create_sample_data(n_samples=8000)
    print(f"Dataset shape: {df.shape}")
    print(f"Conversion rate: {df['converted'].mean():.2%}")
    print(f"Total revenue: ${df['revenue'].sum():,.2f}")

    # Engineer features
    print("\n🔧 Engineering behavioral features...")
    df_engineered = predictor.engineer_features(df)

    # Prepare data
    exclude_cols = ['session_id', 'converted', 'revenue']
    X = df_engineered.drop(exclude_cols, axis=1)

    # Encode categorical variables
    categorical_cols = ['user_type', 'device_type', 'traffic_source']
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            predictor.label_encoders[col] = le

    y = df_engineered['converted']
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training conversion models...")
    results, X_test, y_test, X_train = predictor.train_models(X, y)

    # Print results
    predictor.print_results(results, y_test)

    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_results(results, y_test, X.columns.tolist())

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
