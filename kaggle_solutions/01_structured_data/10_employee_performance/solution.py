"""
Employee Performance Classification
=====================================

Problem: Predict employee performance ratings based on work metrics and engagement data

Difficulty: ⭐⭐

This solution demonstrates:
- Multi-class classification
- Feature engineering for HR analytics
- Handling ordinal targets
- Performance prediction modeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class EmployeePerformancePredictor:
    """Predicts employee performance ratings"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def create_sample_data(self, n_samples=3000):
        """Generate realistic employee performance data"""
        np.random.seed(42)

        data = {
            'age': np.random.randint(22, 65, n_samples),
            'years_at_company': np.random.exponential(4, n_samples).clip(0, 40),
            'years_in_role': np.random.exponential(2.5, n_samples).clip(0, 30),
            'projects_completed': np.random.poisson(8, n_samples),
            'avg_monthly_hours': np.random.normal(180, 30, n_samples).clip(120, 300),
            'training_hours': np.random.gamma(2, 10, n_samples).clip(0, 100),
            'satisfaction_score': np.random.beta(5, 2, n_samples),
            'last_evaluation_score': np.random.beta(4, 2, n_samples),
            'num_promotions': np.random.poisson(0.5, n_samples),
            'salary_level': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.25, 0.3, 0.2, 0.1]),
            'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Operations', 'Marketing'], n_samples),
            'remote_work_pct': np.random.uniform(0, 100, n_samples),
            'overtime_hours': np.random.gamma(1.5, 5, n_samples).clip(0, 50)
        }

        df = pd.DataFrame(data)

        # Calculate performance score
        performance_score = (
            0.3 * df['last_evaluation_score'] +
            0.2 * df['satisfaction_score'] +
            0.15 * (df['projects_completed'] / 15) +
            0.15 * (df['training_hours'] / 50) +
            0.1 * (df['num_promotions'] / 3) +
            0.1 * (df['years_at_company'] / 20) +
            np.random.normal(0, 0.1, n_samples)
        ).clip(0, 1)

        # Convert to performance rating (1-5)
        df['performance_rating'] = pd.cut(
            performance_score,
            bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        return df

    def engineer_features(self, df):
        """Create HR analytics features"""
        df = df.copy()

        # Career progression metrics
        df['promotion_rate'] = df['num_promotions'] / (df['years_at_company'] + 1)
        df['role_tenure_ratio'] = df['years_in_role'] / (df['years_at_company'] + 1)

        # Productivity metrics
        df['projects_per_year'] = df['projects_completed'] / (df['years_at_company'] + 1)
        df['hours_per_project'] = df['avg_monthly_hours'] * 12 / (df['projects_completed'] + 1)

        # Engagement indicators
        df['engagement_score'] = (df['satisfaction_score'] + df['last_evaluation_score']) / 2
        df['development_focus'] = df['training_hours'] / (df['years_at_company'] + 1)

        # Work-life balance
        df['work_intensity'] = (df['avg_monthly_hours'] + df['overtime_hours']) / 200

        # One-hot encode department
        df = pd.get_dummies(df, columns=['department'], prefix='dept')

        return df

    def train_model(self, X, y):
        """Train performance prediction model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        return y_test, y_pred, y_pred_proba, X_test.columns

    def plot_results(self, y_test, y_pred, feature_names):
        """Visualize performance prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        ax = axes[0, 0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_xlabel('Predicted Rating')
        ax.set_ylabel('Actual Rating')
        ax.set_title('Confusion Matrix - Performance Ratings')

        # Performance distribution
        ax = axes[0, 1]
        rating_counts = pd.DataFrame({
            'Actual': y_test.value_counts().sort_index(),
            'Predicted': pd.Series(y_pred).value_counts().sort_index()
        })
        rating_counts.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
        ax.set_xlabel('Performance Rating')
        ax.set_ylabel('Count')
        ax.set_title('Actual vs Predicted Distribution')
        ax.legend(['Actual', 'Predicted'])
        ax.grid(True, alpha=0.3, axis='y')

        # Feature Importance
        ax = axes[1, 0]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(12)

        ax.barh(range(len(feature_importance)), feature_importance['importance'], color='#9b59b6')
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 12 Feature Importances')
        ax.grid(True, alpha=0.3, axis='x')

        # Accuracy by rating
        ax = axes[1, 1]
        ratings = sorted(y_test.unique())
        accuracies = []
        for rating in ratings:
            mask = y_test == rating
            acc = accuracy_score(y_test[mask], y_pred[mask])
            accuracies.append(acc)

        bars = ax.bar(ratings, accuracies, color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60'])
        ax.set_xlabel('Performance Rating')
        ax.set_ylabel('Accuracy')
        ax.set_title('Prediction Accuracy by Rating')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('employee_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'employee_performance_analysis.png'")
        plt.show()

    def print_results(self, y_test, y_pred):
        """Print detailed results"""
        print("\n" + "="*80)
        print("EMPLOYEE PERFORMANCE PREDICTION RESULTS")
        print("="*80)

        print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=[f'Rating {i}' for i in sorted(y_test.unique())]))

        # Performance insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        print("\n✓ Performance ratings: 1 (Poor) to 5 (Excellent)")
        print("✓ Model predicts based on work metrics, satisfaction, and career progression")
        print("✓ Can be used for early identification of high performers and underperformers")
        print("✓ Helps in talent management and succession planning")


def main():
    """Main execution function"""
    print("👔 Employee Performance Classification System")
    print("=" * 80)

    predictor = EmployeePerformancePredictor()

    # Generate data
    print("\n📊 Generating employee performance data...")
    df = predictor.create_sample_data(n_samples=3000)
    print(f"Dataset shape: {df.shape}")

    # Display rating distribution
    print("\nPerformance Rating Distribution:")
    print(df['performance_rating'].value_counts().sort_index())

    # Engineer features
    print("\n🔧 Engineering features...")
    df_processed = predictor.engineer_features(df)

    # Prepare data
    X = df_processed.drop('performance_rating', axis=1)
    y = df_processed['performance_rating']

    # Train model
    print("\n🤖 Training model...")
    y_test, y_pred, y_pred_proba, feature_names = predictor.train_model(X, y)

    # Print results
    predictor.print_results(y_test, y_pred)

    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_results(y_test, y_pred, feature_names)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
