"""
Visual Question Answering (VQA)
Answer questions about images using multi-modal reasoning

Dataset: Synthetic VQA data with images and questions
Difficulty: ⭐⭐⭐⭐ Advanced
Modalities: Image + Question (Text)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


class VisualQuestionAnswering:
    """VQA system combining visual and linguistic understanding"""

    def __init__(self, fusion_strategy='co_attention'):
        """
        Initialize VQA model

        Args:
            fusion_strategy: 'concat', 'co_attention', or 'multimodal_compact'
        """
        self.fusion_strategy = fusion_strategy
        self.model = None
        self.answer_encoder = LabelEncoder()

    def generate_vqa_data(self, n_samples=800):
        """
        Generate synthetic VQA dataset

        Returns:
            DataFrame with images, questions, and answers
        """
        np.random.seed(42)

        # Question types and corresponding answers
        question_templates = {
            'color': ['What color is the object?', 'What is the color of the main object?'],
            'count': ['How many objects are there?', 'What is the count of objects?'],
            'position': ['Where is the object?', 'What is the position of the object?'],
            'action': ['What is happening?', 'What action is taking place?'],
            'attribute': ['What shape is the object?', 'What is the size of the object?']
        }

        answer_sets = {
            'color': ['red', 'blue', 'green', 'yellow', 'black'],
            'count': ['one', 'two', 'three', 'four', 'five'],
            'position': ['left', 'right', 'center', 'top', 'bottom'],
            'action': ['running', 'sitting', 'standing', 'jumping', 'walking'],
            'attribute': ['circle', 'square', 'triangle', 'large', 'small']
        }

        data = []

        for i in range(n_samples):
            # Random question type
            q_type = np.random.choice(list(question_templates.keys()))
            question = np.random.choice(question_templates[q_type])
            answer = np.random.choice(answer_sets[q_type])

            # Generate image features based on answer
            image_features = self._generate_image_for_answer(answer, q_type)

            # Generate question features
            question_features = self._generate_question_features(question, q_type)

            data.append({
                'image_features': image_features,
                'question': question,
                'question_features': question_features,
                'question_type': q_type,
                'answer': answer
            })

        return pd.DataFrame(data)

    def _generate_image_for_answer(self, answer, q_type):
        """Generate image features consistent with answer"""
        # 256-dim image features
        img_feat = np.random.randn(256) * 0.3

        # Encode answer information in image
        answer_hash = hash(answer) % 50
        img_feat[answer_hash:answer_hash+20] += np.random.randn(20) * 2.0

        # Encode question type in image
        type_hash = hash(q_type) % 50 + 50
        img_feat[type_hash:type_hash+20] += np.random.randn(20) * 1.5

        return img_feat

    def _generate_question_features(self, question, q_type):
        """Generate question features (simulating word embeddings)"""
        # 200-dim question features
        q_feat = np.random.randn(200) * 0.3

        # Encode question type
        type_hash = hash(q_type) % 40
        q_feat[type_hash:type_hash+30] += np.random.randn(30) * 2.0

        # Encode specific words
        words = question.split()
        for i, word in enumerate(words[:5]):  # First 5 words
            word_hash = hash(word) % 30 + 40
            q_feat[word_hash:word_hash+10] += np.random.randn(10) * 1.0

        return q_feat

    def extract_visual_features(self, image_features_list):
        """
        Extract features from image embeddings

        Args:
            image_features_list: List of image feature vectors

        Returns:
            Processed visual features
        """
        features = []

        for img_feat in image_features_list:
            feat_dict = {}

            # Statistical features
            feat_dict['img_mean'] = np.mean(img_feat)
            feat_dict['img_std'] = np.std(img_feat)
            feat_dict['img_max'] = np.max(img_feat)
            feat_dict['img_min'] = np.min(img_feat)
            feat_dict['img_skew'] = np.mean((img_feat - np.mean(img_feat))**3) / (np.std(img_feat)**3 + 1e-8)

            # Regional features (simulate spatial pooling)
            n_regions = 4
            region_size = len(img_feat) // n_regions
            for r in range(n_regions):
                region = img_feat[r*region_size:(r+1)*region_size]
                feat_dict[f'region_{r}_mean'] = np.mean(region)
                feat_dict[f'region_{r}_max'] = np.max(region)

            # Top components
            for i in range(min(20, len(img_feat))):
                feat_dict[f'img_c_{i}'] = img_feat[i]

            features.append(feat_dict)

        return pd.DataFrame(features)

    def extract_question_features(self, question_features_list):
        """
        Extract features from question embeddings

        Args:
            question_features_list: List of question feature vectors

        Returns:
            Processed question features
        """
        features = []

        for q_feat in question_features_list:
            feat_dict = {}

            # Statistical features
            feat_dict['q_mean'] = np.mean(q_feat)
            feat_dict['q_std'] = np.std(q_feat)
            feat_dict['q_max'] = np.max(q_feat)
            feat_dict['q_min'] = np.min(q_feat)
            feat_dict['q_norm'] = np.linalg.norm(q_feat)

            # Positional features (beginning vs end)
            feat_dict['q_begin_mean'] = np.mean(q_feat[:len(q_feat)//2])
            feat_dict['q_end_mean'] = np.mean(q_feat[len(q_feat)//2:])

            # Top components
            for i in range(min(20, len(q_feat))):
                feat_dict[f'q_c_{i}'] = q_feat[i]

            features.append(feat_dict)

        return pd.DataFrame(features)

    def co_attention_fusion(self, visual_features, question_features):
        """
        Co-attention mechanism: mutual attention between vision and language

        Args:
            visual_features: Visual feature matrix
            question_features: Question feature matrix

        Returns:
            Co-attended fused features
        """
        # Compute cross-modal attention
        # Question attends to image
        q_to_v_attention = np.dot(question_features, visual_features.T)
        q_to_v_weights = np.exp(q_to_v_attention) / (np.sum(np.exp(q_to_v_attention), axis=1, keepdims=True) + 1e-8)
        attended_visual = np.dot(q_to_v_weights, visual_features)

        # Image attends to question
        v_to_q_attention = np.dot(visual_features, question_features.T)
        v_to_q_weights = np.exp(v_to_q_attention) / (np.sum(np.exp(v_to_q_attention), axis=1, keepdims=True) + 1e-8)
        attended_question = np.dot(v_to_q_weights, question_features)

        # Combine attended features
        fused = np.concatenate([attended_visual, attended_question,
                               visual_features, question_features], axis=1)

        return fused

    def multimodal_compact_fusion(self, visual_features, question_features):
        """
        Multimodal Compact Bilinear pooling approximation

        Args:
            visual_features: Visual features
            question_features: Question features

        Returns:
            Compact bilinear features
        """
        # Element-wise product (simplified version)
        # In practice, would use count sketch or FFT
        outer_product = np.zeros((visual_features.shape[0], 100))

        for i in range(visual_features.shape[0]):
            v = visual_features[i]
            q = question_features[i]

            # Simplified outer product projection
            for j in range(100):
                v_idx = j % visual_features.shape[1]
                q_idx = j % question_features.shape[1]
                outer_product[i, j] = v[v_idx] * q[q_idx]

        # Concatenate with original features
        fused = np.concatenate([outer_product, visual_features, question_features], axis=1)

        return fused

    def train(self, df):
        """
        Train VQA model

        Args:
            df: DataFrame with VQA data
        """
        # Encode answers
        y = self.answer_encoder.fit_transform(df['answer'])

        # Extract features
        X_visual = self.extract_visual_features(df['image_features'])
        X_question = self.extract_question_features(df['question_features'])

        # Apply fusion
        if self.fusion_strategy == 'concat':
            X_fused = np.concatenate([X_visual.values, X_question.values], axis=1)
        elif self.fusion_strategy == 'co_attention':
            X_fused = self.co_attention_fusion(X_visual.values, X_question.values)
        elif self.fusion_strategy == 'multimodal_compact':
            X_fused = self.multimodal_compact_fusion(X_visual.values, X_question.values)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_fused, y)

        return X_visual, X_question, y

    def predict(self, df):
        """Make predictions"""
        X_visual = self.extract_visual_features(df['image_features'])
        X_question = self.extract_question_features(df['question_features'])

        if self.fusion_strategy == 'concat':
            X_fused = np.concatenate([X_visual.values, X_question.values], axis=1)
        elif self.fusion_strategy == 'co_attention':
            X_fused = self.co_attention_fusion(X_visual.values, X_question.values)
        elif self.fusion_strategy == 'multimodal_compact':
            X_fused = self.multimodal_compact_fusion(X_visual.values, X_question.values)

        return self.answer_encoder.inverse_transform(self.model.predict(X_fused))

    def evaluate(self, df):
        """Evaluate model"""
        y_true = df['answer']
        y_pred = self.predict(df)
        return accuracy_score(y_true, y_pred)


def plot_vqa_analysis(results):
    """Visualize VQA results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Fusion strategy comparison
    ax = axes[0, 0]
    strategies = ['Visual\nOnly', 'Question\nOnly', 'Concat', 'Co-Attention', 'Compact\nBilinear']
    accuracies = [
        results['ablation']['Visual Only'],
        results['ablation']['Question Only'],
        results['fusion']['concat'],
        results['fusion']['co_attention'],
        results['fusion']['multimodal_compact']
    ]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#2ecc71', '#f39c12']
    bars = ax.bar(strategies, accuracies, color=colors)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('VQA: Fusion Strategy Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    for bar, v in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.3f}', ha='center', fontweight='bold')

    # Per-question-type accuracy
    ax = axes[0, 1]
    q_types = list(results['question_type_acc'].keys())
    q_accs = list(results['question_type_acc'].values())
    bars = ax.barh(q_types, q_accs, color='#3498db')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Performance by Question Type',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    for bar, acc in zip(bars, q_accs):
        ax.text(acc + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center', fontweight='bold')

    # Answer distribution (top answers)
    ax = axes[1, 0]
    answer_counts = results['answer_distribution']
    top_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    answers, counts = zip(*top_answers)
    bars = ax.bar(range(len(answers)), counts, color='#2ecc71')
    ax.set_xticks(range(len(answers)))
    ax.set_xticklabels(answers, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Top 10 Answers in Test Set',
                 fontsize=14, fontweight='bold')

    # Modality importance
    ax = axes[1, 1]
    modalities = ['Visual\nOnly', 'Question\nOnly', 'Co-Attention\nFusion']
    improvements = [
        0,  # baseline
        results['ablation']['Question Only'] - results['ablation']['Visual Only'],
        results['fusion']['co_attention'] - results['ablation']['Visual Only']
    ]
    colors_imp = ['#95a5a6', '#3498db' if improvements[1] > 0 else '#e74c3c', '#2ecc71']
    bars = ax.bar(modalities, improvements, color=colors_imp)
    ax.set_ylabel('Improvement over Visual Only', fontsize=12)
    ax.set_title('Modality Contribution to VQA',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, imp + 0.01,
                f'{imp:+.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('vqa_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: vqa_analysis.png")


def run_comprehensive_vqa_evaluation(train_df, test_df):
    """Run comprehensive VQA evaluation"""
    results = {
        'fusion': {},
        'ablation': {},
        'question_type_acc': {},
        'answer_distribution': {}
    }

    # Test fusion strategies
    for strategy in ['concat', 'co_attention', 'multimodal_compact']:
        print(f"\n   Training {strategy} fusion model...")
        vqa = VisualQuestionAnswering(fusion_strategy=strategy)
        vqa.train(train_df)

        acc = vqa.evaluate(test_df)
        results['fusion'][strategy] = acc
        print(f"   {strategy.capitalize()} Accuracy: {acc:.4f}")

    # Ablation study
    print("\n   Running ablation study...")

    # Visual only (predict from image alone)
    vqa_vis = VisualQuestionAnswering(fusion_strategy='concat')
    y_train = vqa_vis.answer_encoder.fit_transform(train_df['answer'])
    X_vis = vqa_vis.extract_visual_features(train_df['image_features'])

    vis_model = RandomForestClassifier(n_estimators=100, random_state=42)
    vis_model.fit(X_vis.values, y_train)

    X_vis_test = vqa_vis.extract_visual_features(test_df['image_features'])
    y_test = vqa_vis.answer_encoder.transform(test_df['answer'])
    y_pred = vis_model.predict(X_vis_test.values)
    results['ablation']['Visual Only'] = accuracy_score(y_test, y_pred)

    # Question only
    X_q = vqa_vis.extract_question_features(train_df['question_features'])
    q_model = RandomForestClassifier(n_estimators=100, random_state=42)
    q_model.fit(X_q.values, y_train)

    X_q_test = vqa_vis.extract_question_features(test_df['question_features'])
    y_pred = q_model.predict(X_q_test.values)
    results['ablation']['Question Only'] = accuracy_score(y_test, y_pred)

    print(f"   Visual Only: {results['ablation']['Visual Only']:.4f}")
    print(f"   Question Only: {results['ablation']['Question Only']:.4f}")

    # Per-question-type accuracy
    best_vqa = VisualQuestionAnswering(fusion_strategy='co_attention')
    best_vqa.train(train_df)

    for q_type in test_df['question_type'].unique():
        subset = test_df[test_df['question_type'] == q_type]
        if len(subset) > 0:
            acc = best_vqa.evaluate(subset)
            results['question_type_acc'][q_type] = acc

    # Answer distribution
    results['answer_distribution'] = test_df['answer'].value_counts().to_dict()

    return results


def main():
    """Main execution function"""
    print("=" * 80)
    print("Visual Question Answering (VQA)")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic VQA data...")
    vqa = VisualQuestionAnswering()
    df = vqa.generate_vqa_data(n_samples=800)
    print(f"   Generated {len(df)} VQA pairs")
    print(f"   Question types: {df['question_type'].nunique()}")
    print(f"   Unique answers: {df['answer'].nunique()}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # Run evaluation
    print("\n2. Comparing VQA fusion strategies...")
    results = run_comprehensive_vqa_evaluation(train_df, test_df)

    # Find best
    best_strategy = max(results['fusion'].items(), key=lambda x: x[1])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest Fusion Strategy: {best_strategy[0].upper()}")
    print(f"Best Accuracy: {best_strategy[1]:.4f}")

    print(f"\nAblation Study:")
    print(f"  Visual Only:    {results['ablation']['Visual Only']:.4f}")
    print(f"  Question Only:  {results['ablation']['Question Only']:.4f}")
    print(f"  Combined (Best): {best_strategy[1]:.4f}")

    print(f"\nPer-Question-Type Performance:")
    for q_type, acc in sorted(results['question_type_acc'].items()):
        print(f"  {q_type.capitalize()}: {acc:.4f}")

    # Plot results
    print("\n3. Generating visualizations...")
    plot_vqa_analysis(results)

    print("\n" + "=" * 80)
    print("VQA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
