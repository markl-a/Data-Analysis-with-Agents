"""
Voting Classifier Analysis - Hard vs Soft Voting
Comprehensive comparison of voting ensemble strategies

Dataset: Synthetic classification data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VotingClassifierAnalysis:
    """Comprehensive Voting Classifier Analysis"""

    def __init__(self):
        self.models = {}
        self.voting_classifiers = {}
        self.results = {}
        self.scaler = StandardScaler()

    def create_dataset(self):
        """Create synthetic classification dataset"""
        print("Creating synthetic dataset...")

        X, y = make_classification(
            n_samples=2500,
            n_features=20,
            n_informative=14,
            n_redundant=4,
            n_repeated=2,
            n_classes=3,
            n_clusters_per_class=2,
            weights=[0.4, 0.35, 0.25],
            flip_y=0.04,
            class_sep=0.75,
            random_state=42
        )

        feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['target'].value_counts()}")

        return df, feature_names

    def train_individual_models(self, X_train, X_test, y_train, y_test):
        """Train individual base models"""
        print("\n" + "="*60)
        print("Training Individual Base Models")
        print("="*60)

        # Define models
        models_def = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Naive Bayes': GaussianNB()
        }

        # Train and evaluate each model
        for name, model in models_def.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }

            print(f"  Accuracy: {accuracy:.4f}")

        return models_def

    def create_hard_voting(self, X_train, X_test, y_train, y_test):
        """Create hard voting classifier"""
        print("\n" + "="*60)
        print("Creating Hard Voting Classifier")
        print("="*60)

        # Define estimators
        estimators = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]

        # Create hard voting classifier
        hard_voting = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)

        print("Training hard voting classifier...")
        hard_voting.fit(X_train, y_train)

        y_pred = hard_voting.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Hard Voting Accuracy: {accuracy:.4f}")

        self.voting_classifiers['hard'] = hard_voting
        self.results['Hard Voting'] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }

        return hard_voting

    def create_soft_voting(self, X_train, X_test, y_train, y_test):
        """Create soft voting classifier"""
        print("\n" + "="*60)
        print("Creating Soft Voting Classifier")
        print("="*60)

        # Define estimators (all must support predict_proba)
        estimators = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('nb', GaussianNB())
        ]

        # Create soft voting classifier
        soft_voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

        print("Training soft voting classifier...")
        soft_voting.fit(X_train, y_train)

        y_pred = soft_voting.predict(X_test)
        y_pred_proba = soft_voting.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Soft Voting Accuracy: {accuracy:.4f}")

        self.voting_classifiers['soft'] = soft_voting
        self.results['Soft Voting'] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        return soft_voting

    def create_weighted_voting(self, X_train, X_test, y_train, y_test):
        """Create weighted soft voting classifier"""
        print("\n" + "="*60)
        print("Creating Weighted Soft Voting Classifier")
        print("="*60)

        # Define estimators
        estimators = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]

        # Assign weights based on individual performance
        # Higher performing models get higher weights
        weights = [1.0, 2.0, 2.5, 1.5, 1.0]  # GB gets highest weight
        print(f"Weights: {weights}")

        # Create weighted soft voting classifier
        weighted_voting = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )

        print("Training weighted soft voting classifier...")
        weighted_voting.fit(X_train, y_train)

        y_pred = weighted_voting.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Weighted Soft Voting Accuracy: {accuracy:.4f}")

        self.voting_classifiers['weighted'] = weighted_voting
        self.results['Weighted Voting'] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'weights': weights
        }

        return weighted_voting

    def test_different_combinations(self, X_train, X_test, y_train, y_test):
        """Test different model combinations"""
        print("\n" + "="*60)
        print("Testing Different Model Combinations")
        print("="*60)

        combinations = {
            '3 Models (LR+RF+GB)': [
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ],
            '5 Models (All Tree)': [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
                ('dt1', DecisionTreeClassifier(max_depth=10, random_state=42)),
                ('dt2', DecisionTreeClassifier(max_depth=15, random_state=43))
            ],
            '7 Models (Diverse)': [
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('nb', GaussianNB()),
                ('dt', DecisionTreeClassifier(max_depth=10, random_state=42))
            ]
        }

        combination_results = {}

        for name, estimators in combinations.items():
            print(f"\nTesting {name}...")

            voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            voting_clf.fit(X_train, y_train)

            y_pred = voting_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            combination_results[name] = accuracy
            print(f"  Accuracy: {accuracy:.4f}")

        self.results['combinations'] = combination_results

        return combination_results

    def analyze_voting_diversity(self, X_test, y_test):
        """Analyze how individual models vote"""
        print("\n" + "="*60)
        print("Analyzing Voting Diversity")
        print("="*60)

        # Get individual predictions
        predictions = {}
        model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN']

        for name in model_names:
            if name in self.models:
                predictions[name] = self.models[name].predict(X_test)

        # Calculate agreement matrix
        n_models = len(predictions)
        agreement_matrix = np.zeros((n_models, n_models))

        model_list = list(predictions.keys())
        for i, model1 in enumerate(model_list):
            for j, model2 in enumerate(model_list):
                agreement = np.mean(predictions[model1] == predictions[model2])
                agreement_matrix[i, j] = agreement

        self.results['agreement_matrix'] = {
            'matrix': agreement_matrix,
            'models': model_list
        }

        print("\nPairwise Agreement Matrix:")
        df_agreement = pd.DataFrame(
            agreement_matrix,
            index=model_list,
            columns=model_list
        )
        print(df_agreement.round(3))

        return agreement_matrix

    def cross_validate_voting(self, X, y):
        """Cross-validate voting classifiers"""
        print("\n" + "="*60)
        print("Cross-Validation Analysis")
        print("="*60)

        estimators = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ]

        models_to_cv = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Hard Voting': VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1),
            'Soft Voting': VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        }

        cv_results = {}

        for name, model in models_to_cv.items():
            print(f"\nCross-validating {name}...")
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            cv_results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

        self.results['cv_results'] = cv_results

        return cv_results

    def visualize_results(self, y_test):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # 1. Individual Models vs Voting
        ax1 = plt.subplot(3, 3, 1)
        models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting',
                 'SVM', 'KNN', 'Hard Voting', 'Soft Voting', 'Weighted Voting']
        accuracies = [self.results[m]['accuracy'] for m in models if m in self.results]
        model_names = [m for m in models if m in self.results]

        colors = ['#1f77b4']*5 + ['#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)-5]
        bars = ax1.barh(range(len(model_names)), accuracies, color=colors)
        ax1.set_yticks(range(len(model_names)))
        ax1.set_yticklabels(model_names, fontsize=9)
        ax1.set_xlabel('Accuracy', fontsize=10)
        ax1.set_title('All Models Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlim([min(accuracies) - 0.05, 1.0])
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Hard vs Soft Voting
        ax2 = plt.subplot(3, 3, 2)
        voting_types = ['Hard Voting', 'Soft Voting', 'Weighted Voting']
        voting_accs = [self.results[v]['accuracy'] for v in voting_types if v in self.results]
        voting_names = [v for v in voting_types if v in self.results]

        colors2 = ['#ff7f0e', '#2ca02c', '#d62728'][:len(voting_names)]
        bars = ax2.bar(range(len(voting_names)), voting_accs, color=colors2, edgecolor='black', linewidth=2)
        ax2.set_xticks(range(len(voting_names)))
        ax2.set_xticklabels(voting_names, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Voting Strategies Comparison', fontsize=12, fontweight='bold')
        for i, (bar, acc) in enumerate(zip(bars, voting_accs)):
            ax2.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Model Combinations
        if 'combinations' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            comb_results = self.results['combinations']
            comb_names = list(comb_results.keys())
            comb_accs = list(comb_results.values())

            bars = ax3.bar(range(len(comb_names)), comb_accs, color='skyblue', edgecolor='black')
            ax3.set_xticks(range(len(comb_names)))
            ax3.set_xticklabels(comb_names, rotation=45, ha='right')
            ax3.set_ylabel('Accuracy', fontsize=10)
            ax3.set_title('Different Model Combinations', fontsize=12, fontweight='bold')
            for i, (bar, acc) in enumerate(zip(bars, comb_accs)):
                ax3.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Weighted Voting Weights
        if 'Weighted Voting' in self.results:
            ax4 = plt.subplot(3, 3, 4)
            weights = self.results['Weighted Voting']['weights']
            weight_labels = ['LR', 'RF', 'GB', 'SVM', 'KNN']

            bars = ax4.barh(range(len(weight_labels)), weights, color='coral', edgecolor='black')
            ax4.set_yticks(range(len(weight_labels)))
            ax4.set_yticklabels(weight_labels)
            ax4.set_xlabel('Weight', fontsize=10)
            ax4.set_title('Weighted Voting - Model Weights', fontsize=12, fontweight='bold')
            for i, (bar, w) in enumerate(zip(bars, weights)):
                ax4.text(w + 0.05, i, f'{w:.1f}', va='center')
            ax4.grid(True, alpha=0.3, axis='x')

        # 5. Agreement Matrix Heatmap
        if 'agreement_matrix' in self.results:
            ax5 = plt.subplot(3, 3, 5)
            agr = self.results['agreement_matrix']
            sns.heatmap(agr['matrix'], annot=True, fmt='.3f', cmap='RdYlGn_r',
                       xticklabels=[m.split()[0] for m in agr['models']],
                       yticklabels=[m.split()[0] for m in agr['models']],
                       ax=ax5, vmin=0.5, vmax=1.0)
            ax5.set_title('Model Agreement Matrix', fontsize=12, fontweight='bold')

        # 6. Cross-Validation Results
        if 'cv_results' in self.results:
            ax6 = plt.subplot(3, 3, 6)
            cv_res = self.results['cv_results']
            cv_names = list(cv_res.keys())
            cv_means = [cv_res[m]['mean'] for m in cv_names]
            cv_stds = [cv_res[m]['std'] for m in cv_names]

            bars = ax6.bar(range(len(cv_names)), cv_means, yerr=cv_stds,
                          capsize=5, color='lightgreen', edgecolor='black', alpha=0.7)
            ax6.set_xticks(range(len(cv_names)))
            ax6.set_xticklabels(cv_names, rotation=45, ha='right')
            ax6.set_ylabel('CV Accuracy', fontsize=10)
            ax6.set_title('Cross-Validation Results', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')

        # 7. Confusion Matrix - Soft Voting
        ax7 = plt.subplot(3, 3, 7)
        cm = confusion_matrix(y_test, self.results['Soft Voting']['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7)
        ax7.set_xlabel('Predicted', fontsize=10)
        ax7.set_ylabel('Actual', fontsize=10)
        ax7.set_title('Confusion Matrix - Soft Voting', fontsize=12, fontweight='bold')

        # 8. Voting Mechanism Diagram
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        mechanism_text = """
        Voting Mechanisms

        Hard Voting (Majority):
        Model 1: Class A   ┐
        Model 2: Class B   ├─→ Majority: Class A
        Model 3: Class A   ┘

        Soft Voting (Average Probabilities):
        Model 1: [0.7, 0.2, 0.1]  ┐
        Model 2: [0.6, 0.3, 0.1]  ├─→ Average →
        Model 3: [0.8, 0.1, 0.1]  ┘   argmax

        Weighted Soft Voting:
        w1*P1 + w2*P2 + w3*P3 → argmax

        Advantage: Soft voting uses
        probability information, often
        more robust than hard voting.
        """
        ax8.text(0.05, 0.95, mechanism_text, transform=ax8.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        # 9. Summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        base_accs = [self.results[m]['accuracy'] for m in
                    ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN']
                    if m in self.results]
        best_base = max(base_accs) if base_accs else 0
        hard_acc = self.results['Hard Voting']['accuracy'] if 'Hard Voting' in self.results else 0
        soft_acc = self.results['Soft Voting']['accuracy'] if 'Soft Voting' in self.results else 0
        weighted_acc = self.results['Weighted Voting']['accuracy'] if 'Weighted Voting' in self.results else 0

        summary_text = f"""
        Voting Classifier Summary
        {'='*40}

        Best Base Model: {best_base:.4f}

        Voting Methods:
        • Hard Voting:     {hard_acc:.4f}
        • Soft Voting:     {soft_acc:.4f}
        • Weighted Voting: {weighted_acc:.4f}

        Improvements:
        • Hard:     +{(hard_acc - best_base)*100:.2f}%
        • Soft:     +{(soft_acc - best_base)*100:.2f}%
        • Weighted: +{(weighted_acc - best_base)*100:.2f}%

        Key Insights:
        • Soft voting usually > Hard voting
        • Weights can improve performance
        • Diversity in models is crucial
        • Simple yet effective ensemble
        """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('/tmp/voting_classifier.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to /tmp/voting_classifier.png")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("Voting Classifier Analysis")
    print("="*60)

    # Initialize
    voting_analysis = VotingClassifierAnalysis()

    # Create dataset
    df, feature_names = voting_analysis.create_dataset()

    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = voting_analysis.scaler.fit_transform(X_train)
    X_test_scaled = voting_analysis.scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train individual models
    voting_analysis.train_individual_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Create voting classifiers
    voting_analysis.create_hard_voting(X_train_scaled, X_test_scaled, y_train, y_test)
    voting_analysis.create_soft_voting(X_train_scaled, X_test_scaled, y_train, y_test)
    voting_analysis.create_weighted_voting(X_train_scaled, X_test_scaled, y_train, y_test)

    # Test combinations
    voting_analysis.test_different_combinations(X_train_scaled, X_test_scaled, y_train, y_test)

    # Analyze diversity
    voting_analysis.analyze_voting_diversity(X_test_scaled, y_test)

    # Cross-validation
    voting_analysis.cross_validate_voting(X_train_scaled, y_train)

    # Visualize
    voting_analysis.visualize_results(y_test)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
