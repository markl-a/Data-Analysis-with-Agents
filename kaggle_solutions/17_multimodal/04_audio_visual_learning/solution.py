"""
Audio-Visual Multi-Modal Learning
Learning from synchronized audio and visual signals

Dataset: Synthetic audio-visual event data
Difficulty: ⭐⭐⭐⭐ Advanced
Modalities: Audio + Visual
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class AudioVisualLearning:
    """Audio-visual multi-modal learning system"""

    def __init__(self, fusion_strategy='cross_modal'):
        """
        Initialize audio-visual learning model

        Args:
            fusion_strategy: 'early', 'late', 'cross_modal', or 'adaptive'
        """
        self.fusion_strategy = fusion_strategy
        self.audio_model = None
        self.visual_model = None
        self.fusion_model = None
        self.label_encoder = LabelEncoder()

    def generate_audiovisual_data(self, n_samples=600):
        """
        Generate synthetic audio-visual event data

        Returns:
            DataFrame with synchronized audio-visual features
        """
        np.random.seed(42)

        # Event types with natural audio-visual correspondence
        events = [
            'car_passing',
            'dog_barking',
            'person_talking',
            'music_playing',
            'door_closing'
        ]

        data = []

        for i in range(n_samples):
            event = np.random.choice(events)
            event_idx = events.index(event)

            # Generate visual features (appearance)
            visual_features = self._generate_visual_features(event, event_idx)

            # Generate audio features (sound)
            audio_features = self._generate_audio_features(event, event_idx)

            # Add synchronization signal (correlation between modalities)
            sync_strength = np.random.uniform(0.5, 1.0)

            data.append({
                'event': event,
                'visual_features': visual_features,
                'audio_features': audio_features,
                'sync_strength': sync_strength
            })

        return pd.DataFrame(data)

    def _generate_visual_features(self, event, event_idx):
        """Generate event-specific visual features"""
        # Simulate visual appearance features
        visual = np.random.randn(256) * 0.3

        # Event-specific visual signature
        visual[event_idx*40:(event_idx+1)*40] += np.random.randn(40) * 2.0

        # Add motion patterns
        if event in ['car_passing', 'dog_barking', 'person_talking']:
            visual[200:220] += np.random.randn(20) * 1.5  # Motion signature

        # Add color patterns
        if event in ['car_passing']:
            visual[220:240] += np.random.randn(20) * 1.0  # Color signature

        return visual

    def _generate_audio_features(self, event, event_idx):
        """Generate event-specific audio features"""
        # Simulate audio spectral features
        audio = np.random.randn(128) * 0.3

        # Event-specific audio signature
        audio[event_idx*20:(event_idx+1)*20] += np.random.randn(20) * 2.0

        # Add frequency patterns
        if event in ['music_playing', 'person_talking']:
            audio[100:120] += np.random.randn(20) * 1.5  # Harmonic signature

        # Add temporal patterns
        if event in ['car_passing', 'door_closing']:
            audio[80:100] += np.random.randn(20) * 1.0  # Transient signature

        return audio

    def extract_visual_features(self, visual_features_list):
        """
        Extract features from visual data

        Args:
            visual_features_list: List of visual feature vectors

        Returns:
            Processed visual features
        """
        features = []

        for vis_feat in visual_features_list:
            feat_dict = {}

            # Appearance features
            feat_dict['vis_mean'] = np.mean(vis_feat)
            feat_dict['vis_std'] = np.std(vis_feat)
            feat_dict['vis_max'] = np.max(vis_feat)
            feat_dict['vis_min'] = np.min(vis_feat)
            feat_dict['vis_energy'] = np.sum(vis_feat**2)

            # Motion features (simulated)
            motion_region = vis_feat[200:220]
            feat_dict['vis_motion_intensity'] = np.mean(np.abs(motion_region))
            feat_dict['vis_motion_var'] = np.var(motion_region)

            # Color features (simulated)
            color_region = vis_feat[220:240]
            feat_dict['vis_color_diversity'] = np.std(color_region)

            # Spatial structure
            n_blocks = 4
            block_size = len(vis_feat) // n_blocks
            for b in range(n_blocks):
                block = vis_feat[b*block_size:(b+1)*block_size]
                feat_dict[f'vis_block_{b}_mean'] = np.mean(block)

            # Top components
            for i in range(min(15, len(vis_feat))):
                feat_dict[f'vis_{i}'] = vis_feat[i]

            features.append(feat_dict)

        return pd.DataFrame(features)

    def extract_audio_features(self, audio_features_list):
        """
        Extract features from audio data

        Args:
            audio_features_list: List of audio feature vectors

        Returns:
            Processed audio features
        """
        features = []

        for aud_feat in audio_features_list:
            feat_dict = {}

            # Spectral features
            feat_dict['aud_mean'] = np.mean(aud_feat)
            feat_dict['aud_std'] = np.std(aud_feat)
            feat_dict['aud_max'] = np.max(aud_feat)
            feat_dict['aud_min'] = np.min(aud_feat)
            feat_dict['aud_energy'] = np.sum(aud_feat**2)

            # Frequency characteristics
            low_freq = aud_feat[:len(aud_feat)//3]
            mid_freq = aud_feat[len(aud_feat)//3:2*len(aud_feat)//3]
            high_freq = aud_feat[2*len(aud_feat)//3:]

            feat_dict['aud_low_energy'] = np.sum(low_freq**2)
            feat_dict['aud_mid_energy'] = np.sum(mid_freq**2)
            feat_dict['aud_high_energy'] = np.sum(high_freq**2)

            # Harmonic content
            harmonic_region = aud_feat[100:120]
            feat_dict['aud_harmonic'] = np.mean(np.abs(harmonic_region))

            # Transient content
            transient_region = aud_feat[80:100]
            feat_dict['aud_transient'] = np.max(np.abs(transient_region))

            # Spectral centroid
            feat_dict['aud_centroid'] = np.sum(aud_feat * np.arange(len(aud_feat))) / (np.sum(np.abs(aud_feat)) + 1e-8)

            # Top components
            for i in range(min(15, len(aud_feat))):
                feat_dict[f'aud_{i}'] = aud_feat[i]

            features.append(feat_dict)

        return pd.DataFrame(features)

    def cross_modal_correlation_features(self, visual_features, audio_features):
        """
        Extract cross-modal correlation features

        Args:
            visual_features: Visual feature matrix
            audio_features: Audio feature matrix

        Returns:
            Cross-modal features
        """
        cross_features = []

        for i in range(len(visual_features)):
            v = visual_features[i]
            a = audio_features[i]

            # Cross-correlation
            cross_corr = np.correlate(v, a, mode='valid')[0] if len(v) == len(a) else 0

            # Synchronized energy
            sync_energy = np.dot(v, a) / (np.linalg.norm(v) * np.linalg.norm(a) + 1e-8)

            # Modality agreement
            v_norm = (v - np.mean(v)) / (np.std(v) + 1e-8)
            a_norm = (a - np.mean(a)) / (np.std(a) + 1e-8)

            # Only use matching dimensions
            min_len = min(len(v_norm), len(a_norm))
            agreement = np.mean(v_norm[:min_len] * a_norm[:min_len])

            cross_features.append({
                'cross_correlation': cross_corr,
                'sync_energy': sync_energy,
                'modality_agreement': agreement
            })

        return pd.DataFrame(cross_features)

    def adaptive_fusion(self, visual_features, audio_features):
        """
        Adaptive fusion: Learn to weight modalities based on reliability

        Args:
            visual_features: Visual features
            audio_features: Audio features

        Returns:
            Adaptively fused features
        """
        # Compute modality reliability scores
        vis_reliability = np.std(visual_features, axis=0) / (np.mean(np.abs(visual_features), axis=0) + 1e-8)
        aud_reliability = np.std(audio_features, axis=0) / (np.mean(np.abs(audio_features), axis=0) + 1e-8)

        # Normalize reliability scores
        vis_weights = vis_reliability / (vis_reliability + aud_reliability + 1e-8)
        aud_weights = aud_reliability / (vis_reliability + aud_reliability + 1e-8)

        # Ensure same dimensionality for adaptive weighting
        min_dim = min(visual_features.shape[1], audio_features.shape[1])

        # Apply adaptive weights
        weighted_vis = visual_features[:, :min_dim] * vis_weights[:min_dim]
        weighted_aud = audio_features[:, :min_dim] * aud_weights[:min_dim]

        # Concatenate original and weighted features
        fused = np.concatenate([visual_features, audio_features, weighted_vis, weighted_aud], axis=1)

        return fused

    def train(self, df):
        """
        Train audio-visual model

        Args:
            df: DataFrame with audio-visual data
        """
        # Encode labels
        y = self.label_encoder.fit_transform(df['event'])

        # Extract features
        X_visual = self.extract_visual_features(df['visual_features'])
        X_audio = self.extract_audio_features(df['audio_features'])

        # Train modality-specific models
        self.visual_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.visual_model.fit(X_visual.values, y)

        self.audio_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.audio_model.fit(X_audio.values, y)

        # Fusion
        if self.fusion_strategy == 'early':
            X_fused = np.concatenate([X_visual.values, X_audio.values], axis=1)
        elif self.fusion_strategy == 'cross_modal':
            cross_features = self.cross_modal_correlation_features(
                X_visual.values, X_audio.values
            )
            X_fused = np.concatenate([X_visual.values, X_audio.values,
                                     cross_features.values], axis=1)
        elif self.fusion_strategy == 'adaptive':
            X_fused = self.adaptive_fusion(X_visual.values, X_audio.values)
        elif self.fusion_strategy == 'late':
            # For late fusion, we'll combine predictions later
            return X_visual, X_audio, y

        # Train fusion model
        self.fusion_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.fusion_model.fit(X_fused, y)

        return X_visual, X_audio, y

    def predict(self, df):
        """Make predictions"""
        X_visual = self.extract_visual_features(df['visual_features'])
        X_audio = self.extract_audio_features(df['audio_features'])

        if self.fusion_strategy == 'late':
            # Average probabilities
            vis_probs = self.visual_model.predict_proba(X_visual.values)
            aud_probs = self.audio_model.predict_proba(X_audio.values)
            combined_probs = (vis_probs + aud_probs) / 2
            return self.label_encoder.inverse_transform(
                np.argmax(combined_probs, axis=1)
            )
        else:
            if self.fusion_strategy == 'early':
                X_fused = np.concatenate([X_visual.values, X_audio.values], axis=1)
            elif self.fusion_strategy == 'cross_modal':
                cross_features = self.cross_modal_correlation_features(
                    X_visual.values, X_audio.values
                )
                X_fused = np.concatenate([X_visual.values, X_audio.values,
                                         cross_features.values], axis=1)
            elif self.fusion_strategy == 'adaptive':
                X_fused = self.adaptive_fusion(X_visual.values, X_audio.values)

            return self.label_encoder.inverse_transform(
                self.fusion_model.predict(X_fused)
            )

    def evaluate(self, df):
        """Evaluate model"""
        y_true = df['event']
        y_pred = self.predict(df)
        return accuracy_score(y_true, y_pred)


def plot_audiovisual_analysis(results):
    """Visualize audio-visual learning results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Fusion strategy comparison
    ax = axes[0, 0]
    strategies = ['Audio\nOnly', 'Visual\nOnly', 'Early', 'Late', 'Cross-Modal', 'Adaptive']
    accuracies = [
        results['ablation']['Audio Only'],
        results['ablation']['Visual Only'],
        results['fusion']['early'],
        results['fusion']['late'],
        results['fusion']['cross_modal'],
        results['fusion']['adaptive']
    ]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#9b59b6', '#2ecc71', '#f39c12']
    bars = ax.bar(strategies, accuracies, color=colors)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Audio-Visual Learning: Fusion Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    for bar, v in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

    # Confusion matrix
    ax = axes[0, 1]
    best_strategy = max(results['fusion'].items(), key=lambda x: x[1])[0]
    cm = results['confusion_matrices'][best_strategy]
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
                xticklabels=results['events'], yticklabels=results['events'])
    ax.set_title(f'Confusion Matrix - {best_strategy.capitalize()}',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Event', fontsize=12)
    ax.set_xlabel('Predicted Event', fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Per-event performance
    ax = axes[1, 0]
    report = results['classification_reports'][best_strategy]
    events = results['events']
    f1_scores = [report[event.replace('_', ' ')]['f1-score'] for event in events]
    bars = ax.barh([e.replace('_', ' ') for e in events], f1_scores, color='#2ecc71')
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title(f'Per-Event Performance ({best_strategy.capitalize()})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])

    # Modality synergy analysis
    ax = axes[1, 1]
    modalities = ['Audio\nOnly', 'Visual\nOnly', 'Late\nFusion', 'Cross-Modal\nFusion']
    improvements = [
        0,
        results['ablation']['Visual Only'] - results['ablation']['Audio Only'],
        results['fusion']['late'] - max(results['ablation']['Audio Only'],
                                        results['ablation']['Visual Only']),
        results['fusion']['cross_modal'] - max(results['ablation']['Audio Only'],
                                                results['ablation']['Visual Only'])
    ]
    colors_imp = ['#95a5a6', '#3498db' if improvements[1] > 0 else '#e74c3c',
                  '#9b59b6', '#2ecc71']
    bars = ax.bar(modalities, improvements, color=colors_imp)
    ax.set_ylabel('Improvement', fontsize=12)
    ax.set_title('Multi-Modal Synergy Analysis', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, imp + 0.01,
                f'{imp:+.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('audiovisual_learning_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: audiovisual_learning_analysis.png")


def run_comprehensive_evaluation(train_df, test_df):
    """Run comprehensive audio-visual evaluation"""
    results = {
        'fusion': {},
        'ablation': {},
        'confusion_matrices': {},
        'classification_reports': {},
        'events': None
    }

    # Test fusion strategies
    for strategy in ['early', 'late', 'cross_modal', 'adaptive']:
        print(f"\n   Training {strategy} fusion model...")
        model = AudioVisualLearning(fusion_strategy=strategy)
        model.train(train_df)

        acc = model.evaluate(test_df)
        results['fusion'][strategy] = acc
        print(f"   {strategy.capitalize()} Fusion Accuracy: {acc:.4f}")

        # Get predictions for confusion matrix
        y_pred = model.predict(test_df)
        y_true = test_df['event']
        cm = confusion_matrix(y_true, y_pred, labels=model.label_encoder.classes_)
        report = classification_report(y_true, y_pred, output_dict=True)

        results['confusion_matrices'][strategy] = cm
        results['classification_reports'][strategy] = report

    results['events'] = model.label_encoder.classes_

    # Ablation study
    print("\n   Running ablation study...")

    # Audio only
    model_aud = AudioVisualLearning(fusion_strategy='early')
    model_aud.train(train_df)
    X_aud_test = model_aud.extract_audio_features(test_df['audio_features'])
    y_test = model_aud.label_encoder.transform(test_df['event'])
    y_pred = model_aud.audio_model.predict(X_aud_test.values)
    results['ablation']['Audio Only'] = accuracy_score(y_test, y_pred)

    # Visual only
    X_vis_test = model_aud.extract_visual_features(test_df['visual_features'])
    y_pred = model_aud.visual_model.predict(X_vis_test.values)
    results['ablation']['Visual Only'] = accuracy_score(y_test, y_pred)

    print(f"   Audio Only: {results['ablation']['Audio Only']:.4f}")
    print(f"   Visual Only: {results['ablation']['Visual Only']:.4f}")

    return results


def main():
    """Main execution function"""
    print("=" * 80)
    print("Audio-Visual Multi-Modal Learning")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic audio-visual data...")
    model = AudioVisualLearning()
    df = model.generate_audiovisual_data(n_samples=600)
    print(f"   Generated {len(df)} audio-visual samples")
    print(f"   Events: {', '.join(df['event'].unique())}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42,
                                          stratify=df['event'])
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # Run evaluation
    print("\n2. Comparing audio-visual fusion strategies...")
    results = run_comprehensive_evaluation(train_df, test_df)

    # Find best
    best_strategy = max(results['fusion'].items(), key=lambda x: x[1])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest Fusion Strategy: {best_strategy[0].upper()}")
    print(f"Best Accuracy: {best_strategy[1]:.4f}")

    print(f"\nAblation Study:")
    print(f"  Audio Only:  {results['ablation']['Audio Only']:.4f}")
    print(f"  Visual Only: {results['ablation']['Visual Only']:.4f}")
    print(f"  Combined:    {best_strategy[1]:.4f}")

    better_single = max(results['ablation']['Audio Only'],
                       results['ablation']['Visual Only'])
    improvement = (best_strategy[1] - better_single) * 100

    print(f"\nMulti-Modal Synergy:")
    print(f"  Improvement over best single modality: +{improvement:.2f}%")

    # Plot
    print("\n3. Generating visualizations...")
    plot_audiovisual_analysis(results)

    print("\n" + "=" * 80)
    print("Audio-visual learning analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
