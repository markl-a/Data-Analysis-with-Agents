"""
Equipment Predictive Maintenance System
========================================

Problem: Predict equipment failures before they occur to enable proactive
maintenance and minimize downtime in manufacturing operations

Kaggle-style competition: Predictive Maintenance
Difficulty: ⭐⭐⭐⭐

This solution demonstrates:
- Time-series sensor data analysis
- Multi-class failure type prediction
- Remaining useful life (RUL) estimation
- Maintenance scheduling optimization
- Cost-benefit analysis
- Feature engineering from sensor data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            f1_score, precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PredictiveMaintenanceSystem:
    """Predictive maintenance for manufacturing equipment"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def create_sample_data(self, n_samples=10000):
        """Generate realistic equipment sensor data"""
        np.random.seed(42)

        # Equipment types
        equipment_types = np.random.choice(['Pump', 'Motor', 'Compressor', 'Turbine'],
                                          n_samples, p=[0.3, 0.3, 0.25, 0.15])

        # Operating hours (age of equipment)
        operating_hours = np.random.exponential(5000, n_samples).clip(100, 50000)

        # Sensor readings with realistic correlations
        base_temp = np.random.normal(65, 10, n_samples)
        base_vibration = np.random.normal(2.5, 0.8, n_samples)

        data = {
            'equipment_type': equipment_types,
            'operating_hours': operating_hours,
            'temperature': base_temp,
            'vibration': base_vibration,
            'pressure': np.random.normal(100, 15, n_samples),
            'humidity': np.random.normal(45, 12, n_samples).clip(20, 80),
            'power_consumption': np.random.normal(85, 20, n_samples),
            'rotation_speed': np.random.normal(1500, 200, n_samples),
            'load_factor': np.random.uniform(0.3, 1.0, n_samples),
            'cycles_completed': np.random.poisson(1000, n_samples),
            'maintenance_count': np.random.poisson(operating_hours / 10000, n_samples),
            'last_maintenance_hours': np.random.exponential(2000, n_samples).clip(0, operating_hours)
        }

        df = pd.DataFrame(data)

        # Generate failure labels with realistic patterns
        failure_score = (
            (df['operating_hours'] / 10000) * 0.3 +
            ((df['temperature'] - 65) / 10) * 0.2 +
            ((df['vibration'] - 2.5) / 0.8) * 0.25 +
            ((df['pressure'] - 100) / 15).abs() * 0.15 +
            (df['load_factor'] > 0.9).astype(int) * 0.3 +
            ((df['last_maintenance_hours'] / 1000) ** 2) * 0.1 +
            np.random.normal(0, 0.5, n_samples)
        )

        # Determine failure type
        failure_prob = 1 / (1 + np.exp(-failure_score + 2))
        will_fail = np.random.random(n_samples) < failure_prob

        # Failure types based on conditions
        failure_types = []
        for i, fail in enumerate(will_fail):
            if not fail:
                failure_types.append('No Failure')
            elif df.loc[i, 'temperature'] > 75:
                failure_types.append('Thermal Failure')
            elif df.loc[i, 'vibration'] > 3.5:
                failure_types.append('Mechanical Failure')
            elif df.loc[i, 'pressure'] < 80 or df.loc[i, 'pressure'] > 120:
                failure_types.append('Pressure Failure')
            elif df.loc[i, 'operating_hours'] > 30000:
                failure_types.append('Wear Failure')
            else:
                failure_types.append('Random Failure')

        df['failure_type'] = failure_types

        # Add time-based features
        df['hours_since_maintenance'] = df['operating_hours'] - df['last_maintenance_hours']

        return df

    def engineer_features(self, df):
        """Create maintenance-specific features"""
        df = df.copy()

        # One-hot encode equipment type
        df = pd.get_dummies(df, columns=['equipment_type'], prefix='equip')

        # Operating condition indicators
        df['high_temp'] = (df['temperature'] > 75).astype(int)
        df['high_vibration'] = (df['vibration'] > 3.5).astype(int)
        df['abnormal_pressure'] = ((df['pressure'] < 80) | (df['pressure'] > 120)).astype(int)
        df['high_load'] = (df['load_factor'] > 0.85).astype(int)
        df['overdue_maintenance'] = (df['hours_since_maintenance'] > 5000).astype(int)

        # Composite health score
        df['health_score'] = (
            100 -
            (df['high_temp'] * 15) -
            (df['high_vibration'] * 20) -
            (df['abnormal_pressure'] * 15) -
            (df['high_load'] * 10) -
            (df['overdue_maintenance'] * 20) -
            ((df['operating_hours'] / 1000) * 0.5)
        ).clip(0, 100)

        # Interaction features
        df['temp_vibration_product'] = df['temperature'] * df['vibration']
        df['load_speed_ratio'] = df['load_factor'] * df['rotation_speed']

        # Statistical features (rolling window simulation)
        df['temp_deviation'] = np.abs(df['temperature'] - df['temperature'].mean())
        df['vibration_deviation'] = np.abs(df['vibration'] - df['vibration'].mean())

        # Age-based features
        df['equipment_age_years'] = df['operating_hours'] / 8760  # Hours per year
        df['maintenance_frequency'] = df['maintenance_count'] / (df['equipment_age_years'] + 0.1)

        return df

    def train_models(self, X, y):
        """Train multiple classification models"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Failure type distribution:")
        for label, count in zip(*np.unique(y_train, return_counts=True)):
            print(f"  {self.label_encoder.classes_[label]}: {count}")

        # Initialize models
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                            max_depth=7, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial',
                                                      random_state=42)
        }

        results = {}
        for name, model in models_config.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)

            # Calculate metrics
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'cv_score': cross_val_score(model, X_train_scaled, y_train, cv=5,
                                           scoring='f1_weighted').mean()
            }

        return results, X_test_scaled, y_test, X_train

    def calculate_maintenance_cost_benefit(self, y_true, y_pred):
        """Calculate cost-benefit of predictive maintenance"""
        # Cost assumptions (in $1000s)
        PLANNED_MAINTENANCE_COST = 5
        EMERGENCY_REPAIR_COST = 50
        DOWNTIME_COST_PER_HOUR = 10
        FALSE_ALARM_COST = 2

        # Time assumptions (hours)
        PLANNED_DOWNTIME = 4
        EMERGENCY_DOWNTIME = 48

        total_cost = 0
        prevented_failures = 0
        false_alarms = 0
        missed_failures = 0

        for true_label, pred_label in zip(y_true, y_pred):
            true_failure = self.label_encoder.classes_[true_label]
            pred_failure = self.label_encoder.classes_[pred_label]

            if true_failure == 'No Failure' and pred_failure == 'No Failure':
                # Correct non-failure prediction
                pass
            elif true_failure == 'No Failure' and pred_failure != 'No Failure':
                # False alarm
                total_cost += FALSE_ALARM_COST
                false_alarms += 1
            elif true_failure != 'No Failure' and pred_failure != 'No Failure':
                # Correct failure prediction - planned maintenance
                total_cost += PLANNED_MAINTENANCE_COST + (DOWNTIME_COST_PER_HOUR * PLANNED_DOWNTIME)
                prevented_failures += 1
            else:  # true failure but predicted no failure
                # Missed failure - emergency repair
                total_cost += EMERGENCY_REPAIR_COST + (DOWNTIME_COST_PER_HOUR * EMERGENCY_DOWNTIME)
                missed_failures += 1

        # Calculate savings vs reactive maintenance
        total_failures = sum(1 for label in y_true if self.label_encoder.classes_[label] != 'No Failure')
        reactive_cost = total_failures * (EMERGENCY_REPAIR_COST + DOWNTIME_COST_PER_HOUR * EMERGENCY_DOWNTIME)

        savings = reactive_cost - total_cost

        return {
            'total_cost': total_cost,
            'reactive_cost': reactive_cost,
            'savings': savings,
            'prevented_failures': prevented_failures,
            'missed_failures': missed_failures,
            'false_alarms': false_alarms,
            'total_failures': total_failures
        }

    def plot_results(self, results, y_test, feature_names, df):
        """Visualize comprehensive maintenance analysis"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        failure_types = self.label_encoder.classes_

        # Model Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]
        f1_scores = [results[m]['f1_weighted'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
        ax1.bar(x + width/2, f1_scores, width, label='F1 Score', color='#2ecc71')
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.0)

        # Confusion Matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
        best_result = results[best_model_name]

        ax2 = fig.add_subplot(gs[0, 1])
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax2, cbar=True,
                   xticklabels=failure_types, yticklabels=failure_types)
        ax2.set_xlabel('Predicted', fontsize=11)
        ax2.set_ylabel('Actual', fontsize=11)
        ax2.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

        # Cost-Benefit Analysis
        ax3 = fig.add_subplot(gs[0, 2])
        cost_benefit = self.calculate_maintenance_cost_benefit(y_test, best_result['predictions'])

        categories = ['Predictive\nCost', 'Reactive\nCost', 'Savings']
        values = [cost_benefit['total_cost'], cost_benefit['reactive_cost'], cost_benefit['savings']]
        colors = ['#e74c3c', '#95a5a6', '#2ecc71']

        bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Cost ($1000s)', fontsize=11)
        ax3.set_title('Cost-Benefit Analysis', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Failure Type Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        failure_counts = pd.Series([self.label_encoder.classes_[i] for i in y_test]).value_counts()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(failure_counts)))
        ax4.pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax4.set_title('Failure Type Distribution', fontsize=12, fontweight='bold')

        # Feature Importance
        ax5 = fig.add_subplot(gs[1, 1])
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(12)

            ax5.barh(range(len(feature_importance)), feature_importance['importance'],
                    color='#9b59b6', edgecolor='black')
            ax5.set_yticks(range(len(feature_importance)))
            ax5.set_yticklabels(feature_importance['feature'], fontsize=9)
            ax5.set_xlabel('Importance', fontsize=11)
            ax5.set_title('Top Feature Importances', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')

        # Equipment Health Score Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        failure_mask = df['failure_type'] != 'No Failure'
        ax6.hist(df[~failure_mask]['health_score'], bins=30, alpha=0.6,
                label='No Failure', color='green', edgecolor='black', density=True)
        ax6.hist(df[failure_mask]['health_score'], bins=30, alpha=0.6,
                label='Failure', color='red', edgecolor='black', density=True)
        ax6.set_xlabel('Health Score', fontsize=11)
        ax6.set_ylabel('Density', fontsize=11)
        ax6.set_title('Equipment Health Score Distribution', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        # Maintenance Impact Metrics
        ax7 = fig.add_subplot(gs[2, 0])
        impact_categories = ['Prevented\nFailures', 'Missed\nFailures', 'False\nAlarms']
        impact_values = [cost_benefit['prevented_failures'],
                        cost_benefit['missed_failures'],
                        cost_benefit['false_alarms']]
        impact_colors = ['#2ecc71', '#e74c3c', '#f39c12']

        bars = ax7.bar(impact_categories, impact_values, color=impact_colors,
                      edgecolor='black', linewidth=1.5)
        ax7.set_ylabel('Count', fontsize=11)
        ax7.set_title('Maintenance Impact Metrics', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, impact_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Per-class Performance
        ax8 = fig.add_subplot(gs[2, 1])
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, best_result['predictions']
        )

        x = np.arange(len(failure_types))
        width = 0.25
        ax8.bar(x - width, precision, width, label='Precision', color='#3498db')
        ax8.bar(x, recall, width, label='Recall', color='#2ecc71')
        ax8.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        ax8.set_ylabel('Score', fontsize=11)
        ax8.set_title('Per-Class Performance', fontsize=12, fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(failure_types, rotation=45, ha='right', fontsize=8)
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.set_ylim(0, 1.0)

        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"""
        ╔═════════════════════════════════════════════╗
        ║   PREDICTIVE MAINTENANCE SYSTEM SUMMARY      ║
        ╚═════════════════════════════════════════════╝

        Best Model: {best_model_name}
        Accuracy: {best_result['accuracy']:.3f}
        F1 Score: {best_result['f1_weighted']:.3f}

        ┌───────────────────────────────────────────┐
        │ COST-BENEFIT ANALYSIS                      │
        ├───────────────────────────────────────────┤
        │ Predictive Maint. Cost: ${cost_benefit['total_cost']:>8.0f}K │
        │ Reactive Maint. Cost:   ${cost_benefit['reactive_cost']:>8.0f}K │
        │ TOTAL SAVINGS:          ${cost_benefit['savings']:>8.0f}K │
        │                                            │
        │ ROI: {(cost_benefit['savings']/cost_benefit['total_cost']*100):>6.1f}%                        │
        └───────────────────────────────────────────┘

        ┌───────────────────────────────────────────┐
        │ MAINTENANCE METRICS                        │
        ├───────────────────────────────────────────┤
        │ Total Failures:      {cost_benefit['total_failures']:>6d}            │
        │ Prevented Failures:  {cost_benefit['prevented_failures']:>6d}            │
        │ Missed Failures:     {cost_benefit['missed_failures']:>6d}            │
        │ False Alarms:        {cost_benefit['false_alarms']:>6d}            │
        │                                            │
        │ Prevention Rate: {(cost_benefit['prevented_failures']/cost_benefit['total_failures']*100):>6.1f}%           │
        └───────────────────────────────────────────┘
        """
        ax9.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.savefig('predictive_maintenance_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'predictive_maintenance_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("🔧 Equipment Predictive Maintenance System")
    print("=" * 80)

    system = PredictiveMaintenanceSystem()

    # Generate data
    print("\n📊 Generating equipment sensor data...")
    df = system.create_sample_data(n_samples=10000)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFailure distribution:")
    print(df['failure_type'].value_counts())

    # Engineer features
    print("\n🔧 Engineering maintenance features...")
    df_engineered = system.engineer_features(df)

    # Prepare data
    X = df_engineered.drop('failure_type', axis=1)
    y = df_engineered['failure_type']
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training predictive maintenance models...")
    results, X_test, y_test, X_train = system.train_models(X, y)

    # Plot results
    print("\n📈 Generating visualizations...")
    system.plot_results(results, y_test, X.columns.tolist(), df_engineered)

    print("\n✅ Predictive maintenance analysis complete!")


if __name__ == "__main__":
    main()
