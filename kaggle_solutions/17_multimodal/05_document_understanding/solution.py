"""
Document Understanding with Multi-Modal Learning
Combining OCR (visual) and NLP (textual) for document classification

Dataset: Synthetic document data with layout and text
Difficulty: ⭐⭐⭐⭐ Advanced
Modalities: Document Layout (Visual) + Text Content (NLP)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class DocumentUnderstanding:
    """Multi-modal document understanding combining layout and text"""

    def __init__(self, fusion_strategy='hierarchical'):
        """
        Initialize document understanding model

        Args:
            fusion_strategy: 'concat', 'hierarchical', 'attention', or 'graph'
        """
        self.fusion_strategy = fusion_strategy
        self.model = None
        self.label_encoder = LabelEncoder()
        self.text_vectorizer = TfidfVectorizer(max_features=100)

    def generate_document_data(self, n_samples=700):
        """
        Generate synthetic document data

        Returns:
            DataFrame with document layout and text features
        """
        np.random.seed(42)

        # Document types
        doc_types = [
            'invoice',
            'resume',
            'research_paper',
            'legal_contract',
            'email'
        ]

        # Text templates for each type
        text_templates = {
            'invoice': [
                'invoice number total amount due payment terms',
                'bill invoice date customer order subtotal tax',
                'payment invoice amount description quantity price'
            ],
            'resume': [
                'experience education skills objective summary employment',
                'work experience qualifications education certifications',
                'career objective professional experience technical skills'
            ],
            'research_paper': [
                'abstract introduction methodology results conclusion references',
                'literature review experimental analysis discussion findings',
                'hypothesis methods data results acknowledgments bibliography'
            ],
            'legal_contract': [
                'agreement parties terms conditions obligations rights',
                'contract party hereby whereas terms execution',
                'parties agreement obligations liabilities terms provisions'
            ],
            'email': [
                'subject from recipient message regards attachment',
                'dear sender email message sincerely regards',
                'hello recipient subject message best regards'
            ]
        }

        data = []

        for i in range(n_samples):
            doc_type = np.random.choice(doc_types)

            # Generate layout features
            layout_features = self._generate_layout_features(doc_type)

            # Generate text content
            text_content = np.random.choice(text_templates[doc_type])

            # Add some random words
            random_words = ' '.join([f'word{j}' for j in range(np.random.randint(3, 8))])
            text_content = f"{text_content} {random_words}"

            data.append({
                'doc_type': doc_type,
                'layout_features': layout_features,
                'text_content': text_content
            })

        return pd.DataFrame(data)

    def _generate_layout_features(self, doc_type):
        """Generate document-specific layout features"""
        # Simulate visual layout features (bounding boxes, structure, etc.)
        layout = {}

        # Document dimensions
        layout['width'] = np.random.uniform(800, 1200)
        layout['height'] = np.random.uniform(1000, 1600)

        # Type-specific layout patterns
        if doc_type == 'invoice':
            layout['has_table'] = 1
            layout['num_columns'] = np.random.randint(2, 5)
            layout['header_height'] = np.random.uniform(100, 200)
            layout['footer_present'] = 1
            layout['text_density'] = np.random.uniform(0.3, 0.5)
        elif doc_type == 'resume':
            layout['has_table'] = 0
            layout['num_columns'] = 1
            layout['header_height'] = np.random.uniform(80, 150)
            layout['footer_present'] = 0
            layout['text_density'] = np.random.uniform(0.6, 0.8)
        elif doc_type == 'research_paper':
            layout['has_table'] = np.random.randint(0, 2)
            layout['num_columns'] = 2
            layout['header_height'] = np.random.uniform(50, 100)
            layout['footer_present'] = 1
            layout['text_density'] = np.random.uniform(0.7, 0.9)
        elif doc_type == 'legal_contract':
            layout['has_table'] = 0
            layout['num_columns'] = 1
            layout['header_height'] = np.random.uniform(120, 180)
            layout['footer_present'] = 1
            layout['text_density'] = np.random.uniform(0.8, 0.95)
        else:  # email
            layout['has_table'] = 0
            layout['num_columns'] = 1
            layout['header_height'] = np.random.uniform(60, 120)
            layout['footer_present'] = np.random.randint(0, 2)
            layout['text_density'] = np.random.uniform(0.4, 0.6)

        # Additional layout features
        layout['num_paragraphs'] = np.random.randint(3, 15)
        layout['avg_line_spacing'] = np.random.uniform(1.0, 2.0)
        layout['margin_left'] = np.random.uniform(50, 100)
        layout['margin_right'] = np.random.uniform(50, 100)

        # Visual complexity
        layout['num_text_blocks'] = np.random.randint(2, 10)
        layout['num_images'] = np.random.randint(0, 3)

        # Font statistics
        layout['avg_font_size'] = np.random.uniform(10, 14)
        layout['font_size_std'] = np.random.uniform(1, 3)

        return layout

    def extract_layout_features(self, layout_features_list):
        """
        Extract features from document layout

        Args:
            layout_features_list: List of layout feature dictionaries

        Returns:
            Layout feature DataFrame
        """
        return pd.DataFrame(layout_features_list)

    def extract_text_features(self, text_content_list, fit=False):
        """
        Extract features from text content

        Args:
            text_content_list: List of text strings
            fit: Whether to fit the vectorizer

        Returns:
            Text feature matrix
        """
        if fit:
            text_matrix = self.text_vectorizer.fit_transform(text_content_list)
        else:
            text_matrix = self.text_vectorizer.transform(text_content_list)

        # Convert to dense and create DataFrame
        text_features = pd.DataFrame(
            text_matrix.toarray(),
            columns=[f'text_{i}' for i in range(text_matrix.shape[1])]
        )

        # Add additional text statistics
        for i, text in enumerate(text_content_list):
            words = text.split()
            text_features.loc[i, 'text_length'] = len(text)
            text_features.loc[i, 'num_words'] = len(words)
            text_features.loc[i, 'avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
            text_features.loc[i, 'unique_words'] = len(set(words))
            text_features.loc[i, 'vocab_richness'] = len(set(words)) / (len(words) + 1e-8)

        return text_features

    def hierarchical_fusion(self, layout_features, text_features):
        """
        Hierarchical fusion: Process layout and text at different levels

        Args:
            layout_features: Layout feature matrix
            text_features: Text feature matrix

        Returns:
            Hierarchically fused features
        """
        # Low-level features (raw)
        low_level = np.concatenate([layout_features[:, :5], text_features[:, :10]], axis=1)

        # Mid-level features (statistical aggregations)
        layout_mid = np.column_stack([
            np.mean(layout_features, axis=1),
            np.std(layout_features, axis=1),
            np.max(layout_features, axis=1)
        ])

        text_mid = np.column_stack([
            np.mean(text_features, axis=1),
            np.std(text_features, axis=1),
            np.max(text_features, axis=1)
        ])

        # High-level features (cross-modal)
        layout_text_corr = np.array([
            np.corrcoef(layout_features[i], text_features[i])[0, 1]
            if len(layout_features[i]) == len(text_features[i])
            else 0
            for i in range(len(layout_features))
        ]).reshape(-1, 1)

        # Concatenate all levels
        fused = np.concatenate([low_level, layout_mid, text_mid, layout_text_corr,
                               layout_features, text_features], axis=1)

        return fused

    def attention_fusion(self, layout_features, text_features):
        """
        Attention-based fusion for documents

        Args:
            layout_features: Layout features
            text_features: Text features

        Returns:
            Attention-fused features
        """
        # Compute attention scores
        # Layout attends to text
        layout_to_text = np.dot(layout_features, text_features.T)
        layout_weights = np.exp(layout_to_text) / (np.sum(np.exp(layout_to_text), axis=1, keepdims=True) + 1e-8)
        attended_text = np.dot(layout_weights, text_features)

        # Text attends to layout
        text_to_layout = np.dot(text_features, layout_features.T)
        text_weights = np.exp(text_to_layout) / (np.sum(np.exp(text_to_layout), axis=1, keepdims=True) + 1e-8)
        attended_layout = np.dot(text_weights, layout_features)

        # Combine
        fused = np.concatenate([layout_features, text_features,
                               attended_layout, attended_text], axis=1)

        return fused

    def graph_fusion(self, layout_features, text_features):
        """
        Graph-based fusion: Model relationships as graph

        Args:
            layout_features: Layout features
            text_features: Text features

        Returns:
            Graph-based fused features
        """
        # Simplified graph representation
        # Node features: layout and text elements
        # Edge features: relationships

        # Compute pairwise similarities (edges)
        layout_sim = np.dot(layout_features, layout_features.T) / (
            np.linalg.norm(layout_features, axis=1, keepdims=True) *
            np.linalg.norm(layout_features, axis=1, keepdims=True).T + 1e-8
        )

        text_sim = np.dot(text_features, text_features.T) / (
            np.linalg.norm(text_features, axis=1, keepdims=True) *
            np.linalg.norm(text_features, axis=1, keepdims=True).T + 1e-8
        )

        # Graph features: connectivity patterns
        graph_features = []
        for i in range(len(layout_features)):
            graph_feat = {
                'layout_connectivity': np.sum(layout_sim[i]),
                'text_connectivity': np.sum(text_sim[i]),
                'cross_modal_align': np.dot(layout_features[i], text_features[i])
            }
            graph_features.append(graph_feat)

        graph_df = pd.DataFrame(graph_features)

        # Combine with original features
        fused = np.concatenate([layout_features, text_features, graph_df.values], axis=1)

        return fused

    def train(self, df):
        """
        Train document understanding model

        Args:
            df: DataFrame with document data
        """
        # Encode labels
        y = self.label_encoder.fit_transform(df['doc_type'])

        # Extract features
        X_layout = self.extract_layout_features(df['layout_features'])
        X_text = self.extract_text_features(df['text_content'], fit=True)

        # Apply fusion
        if self.fusion_strategy == 'concat':
            X_fused = np.concatenate([X_layout.values, X_text.values], axis=1)
        elif self.fusion_strategy == 'hierarchical':
            X_fused = self.hierarchical_fusion(X_layout.values, X_text.values)
        elif self.fusion_strategy == 'attention':
            X_fused = self.attention_fusion(X_layout.values, X_text.values)
        elif self.fusion_strategy == 'graph':
            X_fused = self.graph_fusion(X_layout.values, X_text.values)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_fused, y)

        return X_layout, X_text, y

    def predict(self, df):
        """Make predictions"""
        X_layout = self.extract_layout_features(df['layout_features'])
        X_text = self.extract_text_features(df['text_content'], fit=False)

        if self.fusion_strategy == 'concat':
            X_fused = np.concatenate([X_layout.values, X_text.values], axis=1)
        elif self.fusion_strategy == 'hierarchical':
            X_fused = self.hierarchical_fusion(X_layout.values, X_text.values)
        elif self.fusion_strategy == 'attention':
            X_fused = self.attention_fusion(X_layout.values, X_text.values)
        elif self.fusion_strategy == 'graph':
            X_fused = self.graph_fusion(X_layout.values, X_text.values)

        return self.label_encoder.inverse_transform(self.model.predict(X_fused))

    def evaluate(self, df):
        """Evaluate model"""
        y_true = df['doc_type']
        y_pred = self.predict(df)
        return accuracy_score(y_true, y_pred)


def plot_document_analysis(results):
    """Visualize document understanding results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Fusion strategy comparison
    ax = axes[0, 0]
    strategies = ['Layout\nOnly', 'Text\nOnly', 'Concat', 'Hierarchical', 'Attention', 'Graph']
    accuracies = [
        results['ablation']['Layout Only'],
        results['ablation']['Text Only'],
        results['fusion']['concat'],
        results['fusion']['hierarchical'],
        results['fusion']['attention'],
        results['fusion']['graph']
    ]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(strategies, accuracies, color=colors)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Document Understanding: Fusion Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    for bar, v in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

    # Confusion matrix
    ax = axes[0, 1]
    best_strategy = max(results['fusion'].items(), key=lambda x: x[1])[0]
    cm = results['confusion_matrices'][best_strategy]
    doc_types = [dt.replace('_', '\n') for dt in results['doc_types']]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=doc_types, yticklabels=doc_types)
    ax.set_title(f'Confusion Matrix - {best_strategy.capitalize()}',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Type', fontsize=12)
    ax.set_xlabel('Predicted Type', fontsize=12)

    # Per-document-type performance
    ax = axes[1, 0]
    report = results['classification_reports'][best_strategy]
    doc_types_full = results['doc_types']
    f1_scores = [report[dt.replace('_', ' ')]['f1-score'] for dt in doc_types_full]
    bars = ax.barh([dt.replace('_', ' ') for dt in doc_types_full], f1_scores, color='#2ecc71')
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title(f'Per-Document Performance ({best_strategy.capitalize()})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])

    # Modality importance
    ax = axes[1, 1]
    modalities = ['Layout\nOnly', 'Text\nOnly', 'Concat\nFusion', 'Hierarchical\nFusion']
    improvements = [
        0,
        results['ablation']['Text Only'] - results['ablation']['Layout Only'],
        results['fusion']['concat'] - max(results['ablation']['Layout Only'],
                                          results['ablation']['Text Only']),
        results['fusion']['hierarchical'] - max(results['ablation']['Layout Only'],
                                                results['ablation']['Text Only'])
    ]
    colors_imp = ['#95a5a6',
                  '#3498db' if improvements[1] > 0 else '#e74c3c',
                  '#f39c12', '#2ecc71']
    bars = ax.bar(modalities, improvements, color=colors_imp)
    ax.set_ylabel('Improvement', fontsize=12)
    ax.set_title('Multi-Modal Document Understanding Benefit',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, imp + 0.01,
                f'{imp:+.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('document_understanding_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: document_understanding_analysis.png")


def run_comprehensive_evaluation(train_df, test_df):
    """Run comprehensive document understanding evaluation"""
    results = {
        'fusion': {},
        'ablation': {},
        'confusion_matrices': {},
        'classification_reports': {},
        'doc_types': None
    }

    # Test fusion strategies
    for strategy in ['concat', 'hierarchical', 'attention', 'graph']:
        print(f"\n   Training {strategy} fusion model...")
        model = DocumentUnderstanding(fusion_strategy=strategy)
        model.train(train_df)

        acc = model.evaluate(test_df)
        results['fusion'][strategy] = acc
        print(f"   {strategy.capitalize()} Fusion Accuracy: {acc:.4f}")

        # Get predictions
        y_pred = model.predict(test_df)
        y_true = test_df['doc_type']
        cm = confusion_matrix(y_true, y_pred, labels=model.label_encoder.classes_)
        report = classification_report(y_true, y_pred, output_dict=True)

        results['confusion_matrices'][strategy] = cm
        results['classification_reports'][strategy] = report

    results['doc_types'] = model.label_encoder.classes_

    # Ablation study
    print("\n   Running ablation study...")

    # Layout only
    model_layout = DocumentUnderstanding(fusion_strategy='concat')
    y_train = model_layout.label_encoder.fit_transform(train_df['doc_type'])
    X_layout = model_layout.extract_layout_features(train_df['layout_features'])

    layout_model = RandomForestClassifier(n_estimators=100, random_state=42)
    layout_model.fit(X_layout.values, y_train)

    X_layout_test = model_layout.extract_layout_features(test_df['layout_features'])
    y_test = model_layout.label_encoder.transform(test_df['doc_type'])
    y_pred = layout_model.predict(X_layout_test.values)
    results['ablation']['Layout Only'] = accuracy_score(y_test, y_pred)

    # Text only
    X_text = model_layout.extract_text_features(train_df['text_content'], fit=True)
    text_model = RandomForestClassifier(n_estimators=100, random_state=42)
    text_model.fit(X_text.values, y_train)

    X_text_test = model_layout.extract_text_features(test_df['text_content'], fit=False)
    y_pred = text_model.predict(X_text_test.values)
    results['ablation']['Text Only'] = accuracy_score(y_test, y_pred)

    print(f"   Layout Only: {results['ablation']['Layout Only']:.4f}")
    print(f"   Text Only: {results['ablation']['Text Only']:.4f}")

    return results


def main():
    """Main execution function"""
    print("=" * 80)
    print("Document Understanding with Multi-Modal Learning")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic document data...")
    model = DocumentUnderstanding()
    df = model.generate_document_data(n_samples=700)
    print(f"   Generated {len(df)} document samples")
    print(f"   Document types: {', '.join(df['doc_type'].unique())}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42,
                                          stratify=df['doc_type'])
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # Run evaluation
    print("\n2. Comparing document understanding fusion strategies...")
    results = run_comprehensive_evaluation(train_df, test_df)

    # Find best
    best_strategy = max(results['fusion'].items(), key=lambda x: x[1])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest Fusion Strategy: {best_strategy[0].upper()}")
    print(f"Best Accuracy: {best_strategy[1]:.4f}")

    print(f"\nAblation Study:")
    print(f"  Layout Only: {results['ablation']['Layout Only']:.4f}")
    print(f"  Text Only:   {results['ablation']['Text Only']:.4f}")
    print(f"  Combined:    {best_strategy[1]:.4f}")

    better_single = max(results['ablation']['Layout Only'],
                       results['ablation']['Text Only'])
    improvement = (best_strategy[1] - better_single) * 100

    print(f"\nMulti-Modal Benefit:")
    print(f"  Improvement over best single modality: +{improvement:.2f}%")

    # Plot
    print("\n3. Generating visualizations...")
    plot_document_analysis(results)

    print("\n" + "=" * 80)
    print("Document understanding analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
