"""
Image + Text Multi-Modal Classification
Combining visual and textual features for product categorization

Dataset: Synthetic product data with images and descriptions
Difficulty: ⭐⭐⭐ Intermediate
Modalities: Image + Text
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class ImageTextClassifier:
    """Multi-modal classifier combining image and text features"""

    def __init__(self, fusion_strategy='late'):
        """
        Initialize classifier

        Args:
            fusion_strategy: 'early', 'late', or 'hybrid'
        """
        self.fusion_strategy = fusion_strategy
        self.image_model = None
        self.text_model = None
        self.fusion_model = None
        self.label_encoder = LabelEncoder()

    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic product data with image and text features

        Returns:
            DataFrame with image features, text features, and labels
        """
        np.random.seed(42)

        # Product categories
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        n_categories = len(categories)

        # Generate labels
        labels = np.random.choice(categories, n_samples)

        # Generate image features (simulating CNN embeddings)
        # Different patterns for different categories
        image_features = []
        for label in labels:
            cat_idx = categories.index(label)
            # Create category-specific patterns
            base_features = np.random.randn(512) * 0.5
            base_features[cat_idx*100:(cat_idx+1)*100] += np.random.randn(100) * 2.0
            image_features.append(base_features)

        image_features = np.array(image_features)

        # Generate text features (simulating word embeddings)
        text_features = []
        for label in labels:
            cat_idx = categories.index(label)
            # Create category-specific text patterns
            base_features = np.random.randn(300) * 0.5
            base_features[cat_idx*50:(cat_idx+1)*50] += np.random.randn(50) * 2.0
            text_features.append(base_features)

        text_features = np.array(text_features)

        # Create DataFrame
        data = {
            'label': labels,
            'image_features': list(image_features),
            'text_features': list(text_features)
        }

        df = pd.DataFrame(data)
        return df

    def extract_image_features(self, image_features_list):
        """
        Extract statistical features from image embeddings

        Args:
            image_features_list: List of image feature vectors

        Returns:
            Processed image features
        """
        features = []
        for img_feat in image_features_list:
            # Statistical features
            feat_dict = {
                'img_mean': np.mean(img_feat),
                'img_std': np.std(img_feat),
                'img_max': np.max(img_feat),
                'img_min': np.min(img_feat),
                'img_median': np.median(img_feat),
                'img_q75': np.percentile(img_feat, 75),
                'img_q25': np.percentile(img_feat, 25),
            }

            # Add reduced dimensionality features (top principal components simulation)
            for i in range(min(20, len(img_feat))):
                feat_dict[f'img_comp_{i}'] = img_feat[i]

            features.append(feat_dict)

        return pd.DataFrame(features)

    def extract_text_features(self, text_features_list):
        """
        Extract statistical features from text embeddings

        Args:
            text_features_list: List of text feature vectors

        Returns:
            Processed text features
        """
        features = []
        for txt_feat in text_features_list:
            # Statistical features
            feat_dict = {
                'txt_mean': np.mean(txt_feat),
                'txt_std': np.std(txt_feat),
                'txt_max': np.max(txt_feat),
                'txt_min': np.min(txt_feat),
                'txt_median': np.median(txt_feat),
                'txt_q75': np.percentile(txt_feat, 75),
                'txt_q25': np.percentile(txt_feat, 25),
            }

            # Add reduced dimensionality features
            for i in range(min(20, len(txt_feat))):
                feat_dict[f'txt_comp_{i}'] = txt_feat[i]

            features.append(feat_dict)

        return pd.DataFrame(features)

    def early_fusion(self, image_features, text_features):
        """
        Early fusion: Concatenate features before classification

        Args:
            image_features: Image feature matrix
            text_features: Text feature matrix

        Returns:
            Concatenated features
        """
        return np.concatenate([image_features, text_features], axis=1)

    def train_late_fusion(self, X_img, X_txt, y):
        """
        Late fusion: Train separate models and combine predictions

        Args:
            X_img: Image features
            X_txt: Text features
            y: Labels
        """
        # Train image model
        self.image_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.image_model.fit(X_img, y)

        # Train text model
        self.text_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.text_model.fit(X_txt, y)

    def train_early_fusion(self, X_combined, y):
        """
        Early fusion: Train single model on combined features

        Args:
            X_combined: Combined image and text features
            y: Labels
        """
        self.fusion_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            random_state=42
        )
        self.fusion_model.fit(X_combined, y)

    def train_hybrid_fusion(self, X_img, X_txt, y):
        """
        Hybrid fusion: Combine predictions from separate models with a meta-classifier

        Args:
            X_img: Image features
            X_txt: Text features
            y: Labels
        """
        # Train base models
        self.train_late_fusion(X_img, X_txt, y)

        # Get predictions from base models
        img_probs = self.image_model.predict_proba(X_img)
        txt_probs = self.text_model.predict_proba(X_txt)

        # Combine probabilities as meta-features
        meta_features = np.concatenate([img_probs, txt_probs], axis=1)

        # Train meta-classifier
        self.fusion_model = LogisticRegression(max_iter=1000, random_state=42)
        self.fusion_model.fit(meta_features, y)

    def predict(self, X_img, X_txt):
        """
        Make predictions using the selected fusion strategy

        Args:
            X_img: Image features
            X_txt: Text features

        Returns:
            Predictions
        """
        if self.fusion_strategy == 'late':
            # Average predictions from both models
            img_probs = self.image_model.predict_proba(X_img)
            txt_probs = self.text_model.predict_proba(X_txt)
            combined_probs = (img_probs + txt_probs) / 2
            return self.image_model.classes_[np.argmax(combined_probs, axis=1)]

        elif self.fusion_strategy == 'early':
            # Predict with combined features
            X_combined = self.early_fusion(X_img, X_txt)
            return self.fusion_model.predict(X_combined)

        elif self.fusion_strategy == 'hybrid':
            # Use meta-classifier on base model predictions
            img_probs = self.image_model.predict_proba(X_img)
            txt_probs = self.text_model.predict_proba(X_txt)
            meta_features = np.concatenate([img_probs, txt_probs], axis=1)
            return self.fusion_model.predict(meta_features)

    def train(self, df):
        """
        Train the multi-modal classifier

        Args:
            df: DataFrame with image_features, text_features, and labels
        """
        # Encode labels
        y = self.label_encoder.fit_transform(df['label'])

        # Extract features
        X_img = self.extract_image_features(df['image_features'])
        X_txt = self.extract_text_features(df['text_features'])

        # Train based on fusion strategy
        if self.fusion_strategy == 'late':
            self.train_late_fusion(X_img.values, X_txt.values, y)
        elif self.fusion_strategy == 'early':
            X_combined = self.early_fusion(X_img.values, X_txt.values)
            self.train_early_fusion(X_combined, y)
        elif self.fusion_strategy == 'hybrid':
            self.train_hybrid_fusion(X_img.values, X_txt.values, y)

        return X_img, X_txt, y

    def evaluate(self, df):
        """
        Evaluate the model

        Args:
            df: Test DataFrame

        Returns:
            Accuracy score
        """
        y_true = self.label_encoder.transform(df['label'])
        X_img = self.extract_image_features(df['image_features'])
        X_txt = self.extract_text_features(df['text_features'])

        y_pred = self.predict(X_img.values, X_txt.values)
        # y_pred is already encoded (integers), so compare directly
        return accuracy_score(y_true, y_pred)


def plot_fusion_comparison(results):
    """Plot comparison of different fusion strategies"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy comparison
    ax = axes[0, 0]
    strategies = [k for k in results.keys() if k != 'ablation']
    accuracies = [results[s]['accuracy'] for s in strategies]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    ax.bar(strategies, accuracies, color=colors[:len(strategies)])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Fusion Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Confusion matrix for best model
    best_strategy = max([k for k in results.keys() if k != 'ablation'],
                       key=lambda x: results[x]['accuracy'])
    ax = axes[0, 1]
    cm = results[best_strategy]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {best_strategy} Fusion',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    # Feature importance comparison
    ax = axes[1, 0]
    if 'early' in results:
        report = results['early']['classification_report']
        categories = list(report.keys())[:-3]  # Exclude avg metrics
        f1_scores = [report[cat]['f1-score'] for cat in categories]
        ax.barh(categories, f1_scores, color='#3498db')
        ax.set_xlabel('F1-Score', fontsize=12)
        ax.set_title('Per-Category Performance (Early Fusion)',
                     fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])

    # Ablation study
    ax = axes[1, 1]
    modalities = list(results['ablation'].keys())
    ablation_acc = [results['ablation'][m] for m in modalities]
    ax.bar(modalities, ablation_acc, color=['#e74c3c', '#3498db', '#2ecc71'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Individual vs Combined',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    for i, v in enumerate(ablation_acc):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('multimodal_fusion_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: multimodal_fusion_comparison.png")


def run_ablation_study(train_df, test_df):
    """
    Run ablation study: compare image-only, text-only, and combined

    Returns:
        Dictionary with results for each configuration
    """
    results = {}

    # Image-only model
    img_classifier = ImageTextClassifier(fusion_strategy='early')
    y_train = img_classifier.label_encoder.fit_transform(train_df['label'])
    X_img_train = img_classifier.extract_image_features(train_df['image_features'])

    img_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    img_model.fit(X_img_train.values, y_train)

    X_img_test = img_classifier.extract_image_features(test_df['image_features'])
    y_test = img_classifier.label_encoder.transform(test_df['label'])
    y_pred = img_model.predict(X_img_test.values)
    results['Image Only'] = accuracy_score(y_test, y_pred)

    # Text-only model
    txt_classifier = ImageTextClassifier(fusion_strategy='early')
    y_train = txt_classifier.label_encoder.fit_transform(train_df['label'])
    X_txt_train = txt_classifier.extract_text_features(train_df['text_features'])

    txt_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    txt_model.fit(X_txt_train.values, y_train)

    X_txt_test = txt_classifier.extract_text_features(test_df['text_features'])
    y_pred = txt_model.predict(X_txt_test.values)
    results['Text Only'] = accuracy_score(y_test, y_pred)

    # Combined model (early fusion)
    combined_classifier = ImageTextClassifier(fusion_strategy='early')
    combined_classifier.train(train_df)
    results['Combined'] = combined_classifier.evaluate(test_df)

    return results


def main():
    """Main execution function"""
    print("=" * 80)
    print("Image + Text Multi-Modal Classification")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic product data...")
    classifier = ImageTextClassifier()
    df = classifier.generate_synthetic_data(n_samples=1000)
    print(f"   Generated {len(df)} samples across {df['label'].nunique()} categories")
    print(f"   Categories: {', '.join(df['label'].unique())}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                          stratify=df['label'])
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # Compare fusion strategies
    print("\n2. Comparing fusion strategies...")
    results = {}

    for strategy in ['early', 'late', 'hybrid']:
        print(f"\n   Training {strategy} fusion model...")
        clf = ImageTextClassifier(fusion_strategy=strategy)
        clf.train(train_df)

        # Evaluate
        X_img_test = clf.extract_image_features(test_df['image_features'])
        X_txt_test = clf.extract_text_features(test_df['text_features'])
        y_test = clf.label_encoder.transform(test_df['label'])

        y_pred = clf.predict(X_img_test.values, X_txt_test.values)
        # y_pred is already encoded (integers), no need to transform again

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=clf.label_encoder.classes_,
                                       output_dict=True)

        results[strategy] = {
            'accuracy': acc,
            'confusion_matrix': cm,
            'classification_report': report
        }

        print(f"   {strategy.capitalize()} Fusion Accuracy: {acc:.4f}")

    # Ablation study
    print("\n3. Running ablation study...")
    ablation_results = run_ablation_study(train_df, test_df)
    results['ablation'] = ablation_results

    print("\n   Ablation Study Results:")
    for modality, acc in ablation_results.items():
        print(f"   {modality}: {acc:.4f}")

    # Find best strategy
    best_strategy = max(['early', 'late', 'hybrid'],
                       key=lambda x: results[x]['accuracy'])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest Fusion Strategy: {best_strategy.upper()}")
    print(f"Best Accuracy: {results[best_strategy]['accuracy']:.4f}")
    print(f"\nImprovement over single modality:")
    print(f"  vs Image Only: +{(results[best_strategy]['accuracy'] - ablation_results['Image Only'])*100:.2f}%")
    print(f"  vs Text Only:  +{(results[best_strategy]['accuracy'] - ablation_results['Text Only'])*100:.2f}%")

    # Plot results
    print("\n4. Generating visualizations...")
    plot_fusion_comparison(results)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
