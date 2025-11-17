"""
Explainable Recommendation System
==================================

This solution demonstrates explainable recommendation methods including
LIME-style explanations, feature attribution, user-centric explanations,
and trustworthiness metrics.

Author: Kaggle Solutions Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ExplainableRecommender:
    """Base recommender with explanation capabilities"""

    def __init__(self, model_type: str = 'rf'):
        """
        Initialize explainable recommender

        Args:
            model_type: Type of model ('rf', 'linear', 'tree')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Train the model

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Names of features
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'tree':
            self.model = DecisionTreeRegressor(max_depth=10, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            return {}

        return dict(zip(self.feature_names, importances))

    def explain_prediction(self, x: np.ndarray, n_samples: int = 1000) -> Dict[str, float]:
        """
        LIME-style explanation for a single prediction

        Args:
            x: Single instance to explain
            n_samples: Number of samples for local approximation

        Returns:
            Dictionary of feature contributions
        """
        # Generate perturbations around the instance
        perturbations = np.random.normal(0, 0.1, (n_samples, len(x)))
        perturbed = x + perturbations

        # Get predictions for perturbed samples
        predictions = self.predict(perturbed)

        # Fit linear model locally
        local_model = LinearRegression()
        local_model.fit(perturbations, predictions)

        # Feature contributions
        contributions = local_model.coef_ * x

        return dict(zip(self.feature_names, contributions))


class ContentBasedExplainer:
    """Content-based recommendation with explanations"""

    def __init__(self):
        """Initialize content-based explainer"""
        self.item_features = None
        self.user_profiles = None

    def fit(self, interactions: pd.DataFrame, item_features: pd.DataFrame):
        """
        Build user profiles from interactions

        Args:
            interactions: User-item interactions with ratings
            item_features: Item feature matrix
        """
        self.item_features = item_features

        # Build user profiles as weighted average of item features
        self.user_profiles = {}
        for user_id in interactions['user_id'].unique():
            user_data = interactions[interactions['user_id'] == user_id]
            weighted_features = np.zeros(len(item_features.columns) - 1)

            for _, row in user_data.iterrows():
                item_id = row['item_id']
                rating = row['rating']

                if item_id in item_features['item_id'].values:
                    item_feat = item_features[item_features['item_id'] == item_id].iloc[0, 1:].values
                    weighted_features += rating * item_feat

            # Normalize
            if np.sum(weighted_features) > 0:
                weighted_features /= np.sum(np.abs(weighted_features))

            self.user_profiles[user_id] = weighted_features

    def explain_recommendation(self, user_id: int, item_id: int) -> Dict[str, float]:
        """
        Explain why an item is recommended to a user

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Dictionary of feature-level explanations
        """
        if user_id not in self.user_profiles:
            return {}

        user_profile = self.user_profiles[user_id]
        item_feat = self.item_features[self.item_features['item_id'] == item_id].iloc[0, 1:].values

        # Calculate feature-wise similarity
        explanations = {}
        feature_names = self.item_features.columns[1:]

        for i, feat_name in enumerate(feature_names):
            contribution = user_profile[i] * item_feat[i]
            explanations[feat_name] = contribution

        return explanations


def generate_explainable_data(n_users: int = 500, n_items: int = 200,
                             n_interactions: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data for explainable recommendations

    Args:
        n_users: Number of users
        n_items: Number of items
        n_interactions: Number of interactions

    Returns:
        Interactions DataFrame and item features DataFrame
    """
    np.random.seed(42)

    # Generate item features
    item_features = {
        'item_id': list(range(n_items)),
        'genre_action': np.random.rand(n_items),
        'genre_comedy': np.random.rand(n_items),
        'genre_drama': np.random.rand(n_items),
        'popularity': np.random.rand(n_items),
        'release_year': np.random.randint(1990, 2024, n_items),
        'rating_avg': np.random.uniform(2, 5, n_items)
    }
    item_df = pd.DataFrame(item_features)

    # Normalize year
    item_df['release_year'] = (item_df['release_year'] - 1990) / 34

    # Generate user preferences
    user_preferences = {
        user_id: {
            'genre_action': np.random.rand(),
            'genre_comedy': np.random.rand(),
            'genre_drama': np.random.rand(),
            'popularity_pref': np.random.rand(),
            'recency_pref': np.random.rand()
        }
        for user_id in range(n_users)
    }

    # Generate interactions
    interactions = []
    for _ in range(n_interactions):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)

        # Get user preferences and item features
        user_pref = user_preferences[user_id]
        item_feat = item_df[item_df['item_id'] == item_id].iloc[0]

        # Calculate rating based on preference matching
        rating = 2.5  # Base rating
        rating += user_pref['genre_action'] * item_feat['genre_action'] * 0.8
        rating += user_pref['genre_comedy'] * item_feat['genre_comedy'] * 0.8
        rating += user_pref['genre_drama'] * item_feat['genre_drama'] * 0.8
        rating += user_pref['popularity_pref'] * item_feat['popularity'] * 0.5
        rating += user_pref['recency_pref'] * item_feat['release_year'] * 0.4

        # Add noise
        rating += np.random.normal(0, 0.3)
        rating = np.clip(rating, 1, 5)

        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

    interactions_df = pd.DataFrame(interactions)
    return interactions_df, item_df


def calculate_trustworthiness(model, X_test: np.ndarray, y_test: np.ndarray,
                              explanations: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate trustworthiness metrics

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        explanations: List of explanations

    Returns:
        Dictionary of trustworthiness metrics
    """
    # Prediction accuracy
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Explanation consistency
    explanation_stds = []
    for expl in explanations:
        if expl:
            values = list(expl.values())
            explanation_stds.append(np.std(values))

    consistency = 1.0 / (1.0 + np.mean(explanation_stds)) if explanation_stds else 0.0

    # Feature importance stability
    feature_importances = model.get_feature_importance()
    if feature_importances:
        importance_entropy = -np.sum([p * np.log(p + 1e-10)
                                     for p in feature_importances.values()
                                     if p > 0])
        stability = 1.0 / (1.0 + importance_entropy)
    else:
        stability = 0.0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Explanation Consistency': consistency,
        'Feature Importance Stability': stability
    }


def plot_feature_importance(model: ExplainableRecommender, save_path: str = None):
    """Plot global feature importance"""
    importances = model.get_feature_importance()

    if not importances:
        return

    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(features)), values, color='skyblue', edgecolor='black')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title('Global Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_explanations(explanations: List[Dict[str, float]],
                                 predictions: np.ndarray, actuals: np.ndarray,
                                 n_samples: int = 6, save_path: str = None):
    """Plot explanations for sample predictions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx in range(min(n_samples, len(explanations))):
        expl = explanations[idx]
        if not expl:
            continue

        # Sort by absolute contribution
        sorted_expl = sorted(expl.items(), key=lambda x: abs(x[1]), reverse=True)
        features, contributions = zip(*sorted_expl[:10])  # Top 10 features

        # Color positive and negative contributions differently
        colors = ['green' if c > 0 else 'red' for c in contributions]

        axes[idx].barh(range(len(features)), contributions, color=colors, alpha=0.7,
                      edgecolor='black')
        axes[idx].set_yticks(range(len(features)))
        axes[idx].set_yticklabels(features, fontsize=8)
        axes[idx].set_xlabel('Contribution', fontsize=9)
        axes[idx].set_title(f'Prediction: {predictions[idx]:.2f} | Actual: {actuals[idx]:.2f}',
                           fontsize=10, fontweight='bold')
        axes[idx].axvline(0, color='black', linewidth=1)
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_explanation_comparison(content_explanations: List[Dict[str, float]],
                               lime_explanations: List[Dict[str, float]],
                               save_path: str = None):
    """Compare different explanation methods"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Aggregate feature importance across all explanations
    content_agg = {}
    lime_agg = {}

    for expl in content_explanations:
        for feat, val in expl.items():
            content_agg[feat] = content_agg.get(feat, 0) + abs(val)

    for expl in lime_explanations:
        for feat, val in expl.items():
            lime_agg[feat] = lime_agg.get(feat, 0) + abs(val)

    # Plot content-based explanations
    if content_agg:
        sorted_content = sorted(content_agg.items(), key=lambda x: x[1], reverse=True)[:10]
        features, values = zip(*sorted_content)
        axes[0].barh(range(len(features)), values, color='coral', edgecolor='black')
        axes[0].set_yticks(range(len(features)))
        axes[0].set_yticklabels(features)
        axes[0].set_xlabel('Cumulative Contribution')
        axes[0].set_title('Content-Based Explanations', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')

    # Plot LIME explanations
    if lime_agg:
        sorted_lime = sorted(lime_agg.items(), key=lambda x: x[1], reverse=True)[:10]
        features, values = zip(*sorted_lime)
        axes[1].barh(range(len(features)), values, color='lightgreen', edgecolor='black')
        axes[1].set_yticks(range(len(features)))
        axes[1].set_yticklabels(features)
        axes[1].set_xlabel('Cumulative Contribution')
        axes[1].set_title('LIME-Style Explanations', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_trustworthiness_metrics(metrics: Dict[str, float], save_path: str = None):
    """Plot trustworthiness metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy metrics
    accuracy_metrics = {k: v for k, v in metrics.items() if k in ['MAE', 'RMSE']}
    axes[0].bar(range(len(accuracy_metrics)), list(accuracy_metrics.values()),
                color=['skyblue', 'coral'], edgecolor='black')
    axes[0].set_xticks(range(len(accuracy_metrics)))
    axes[0].set_xticklabels(list(accuracy_metrics.keys()))
    axes[0].set_ylabel('Score')
    axes[0].set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Trustworthiness metrics
    trust_metrics = {k: v for k, v in metrics.items()
                    if k not in ['MAE', 'RMSE']}
    axes[1].bar(range(len(trust_metrics)), list(trust_metrics.values()),
                color=['lightgreen', 'plum'], edgecolor='black')
    axes[1].set_xticks(range(len(trust_metrics)))
    axes[1].set_xticklabels(list(trust_metrics.keys()), rotation=45, ha='right')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Explanation Quality', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_explanation_distribution(explanations: List[Dict[str, float]],
                                  save_path: str = None):
    """Plot distribution of explanation values"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect all explanation values by feature
    feature_values = {}
    for expl in explanations:
        for feat, val in expl.items():
            if feat not in feature_values:
                feature_values[feat] = []
            feature_values[feat].append(val)

    # Plot distribution for top features
    top_features = sorted(feature_values.items(),
                         key=lambda x: np.mean(np.abs(x[1])),
                         reverse=True)[:4]

    for idx, (feat, values) in enumerate(top_features):
        ax = axes[idx // 2, idx % 2]
        ax.hist(values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Contribution Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feat} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Explainable Recommendation System")
    print("=" * 80)

    # Generate data
    print("\n1. Generating explainable recommendation data...")
    interactions_df, item_df = generate_explainable_data(
        n_users=500, n_items=200, n_interactions=5000
    )
    print(f"Generated {len(interactions_df)} interactions")
    print(f"Item features: {list(item_df.columns[1:])}")

    # Prepare features
    print("\n2. Preparing features...")
    X_list = []
    y_list = []

    for _, row in interactions_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']

        item_features = item_df[item_df['item_id'] == item_id].iloc[0, 1:].values
        features = np.concatenate([[user_id], item_features])

        X_list.append(features)
        y_list.append(rating)

    X = np.array(X_list)
    y = np.array(y_list)

    feature_names = ['user_id'] + list(item_df.columns[1:])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train explainable model
    print("\n3. Training explainable model...")
    model = ExplainableRecommender(model_type='rf')
    model.fit(X_train, y_train, feature_names=feature_names)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Model MAE: {mae:.4f}")
    print(f"Model RMSE: {rmse:.4f}")

    # Generate explanations
    print("\n4. Generating explanations...")
    lime_explanations = []
    for i in range(min(100, len(X_test))):
        expl = model.explain_prediction(X_test[i])
        lime_explanations.append(expl)

    # Content-based explanations
    print("\n5. Generating content-based explanations...")
    content_explainer = ContentBasedExplainer()
    content_explainer.fit(interactions_df, item_df)

    content_explanations = []
    for i in range(min(100, len(interactions_df))):
        row = interactions_df.iloc[i]
        expl = content_explainer.explain_recommendation(row['user_id'], row['item_id'])
        content_explanations.append(expl)

    # Calculate trustworthiness
    print("\n6. Calculating trustworthiness metrics...")
    trust_metrics = calculate_trustworthiness(model, X_test, y_test, lime_explanations)

    # Visualizations
    print("\n7. Generating visualizations...")
    plot_feature_importance(model)
    plot_prediction_explanations(lime_explanations, predictions, y_test, n_samples=6)
    plot_explanation_comparison(content_explanations, lime_explanations)
    plot_trustworthiness_metrics(trust_metrics)
    plot_explanation_distribution(lime_explanations)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nModel Performance:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    print("\nTrustworthiness Metrics:")
    for metric, value in trust_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTop Feature Importances:")
    importances = model.get_feature_importance()
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importances[:5]:
        print(f"  {feat}: {imp:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
