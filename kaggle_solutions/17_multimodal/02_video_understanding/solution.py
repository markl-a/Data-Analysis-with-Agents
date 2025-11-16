"""
Video Understanding with Multi-Modal Learning
Combining visual frames and audio features for video action recognition

Dataset: Synthetic video data with visual and audio features
Difficulty: ⭐⭐⭐⭐ Advanced
Modalities: Video Frames + Audio
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


class VideoUnderstandingModel:
    """Multi-modal video understanding combining visual and audio features"""

    def __init__(self, fusion_type='attention'):
        """
        Initialize video understanding model

        Args:
            fusion_type: 'concat', 'attention', or 'temporal'
        """
        self.fusion_type = fusion_type
        self.visual_model = None
        self.audio_model = None
        self.fusion_model = None
        self.label_encoder = LabelEncoder()

    def generate_video_data(self, n_videos=500, frames_per_video=16):
        """
        Generate synthetic video data with temporal sequences

        Args:
            n_videos: Number of video samples
            frames_per_video: Number of frames per video

        Returns:
            DataFrame with video features and labels
        """
        np.random.seed(42)

        # Action classes
        actions = ['Walking', 'Running', 'Dancing', 'Waving', 'Sitting']
        n_actions = len(actions)

        data = []

        for i in range(n_videos):
            action = np.random.choice(actions)
            action_idx = actions.index(action)

            # Generate temporal visual features (frames over time)
            visual_frames = []
            for t in range(frames_per_video):
                # Create action-specific visual patterns with temporal dynamics
                frame_feat = np.random.randn(128) * 0.3
                # Action signature
                frame_feat[action_idx*20:(action_idx+1)*20] += np.random.randn(20) * 1.5
                # Temporal progression
                frame_feat += np.sin(2 * np.pi * t / frames_per_video) * 0.5
                visual_frames.append(frame_feat)

            visual_frames = np.array(visual_frames)

            # Generate audio features (temporal audio signal)
            audio_frames = []
            for t in range(frames_per_video):
                # Create action-specific audio patterns
                audio_feat = np.random.randn(64) * 0.3
                # Action signature in audio
                audio_feat[action_idx*10:(action_idx+1)*10] += np.random.randn(10) * 1.5
                # Temporal rhythm
                audio_feat += np.cos(2 * np.pi * t / frames_per_video) * 0.4
                audio_frames.append(audio_feat)

            audio_frames = np.array(audio_frames)

            data.append({
                'video_id': i,
                'action': action,
                'visual_frames': visual_frames,
                'audio_frames': audio_frames,
                'duration': frames_per_video
            })

        return pd.DataFrame(data)

    def extract_temporal_visual_features(self, visual_frames_list):
        """
        Extract temporal features from video frames

        Args:
            visual_frames_list: List of frame sequences

        Returns:
            Temporal visual features
        """
        features = []

        for frames in visual_frames_list:
            feat_dict = {}

            # Statistical features across frames
            feat_dict['vis_temporal_mean'] = np.mean(frames)
            feat_dict['vis_temporal_std'] = np.std(frames)
            feat_dict['vis_temporal_max'] = np.max(frames)
            feat_dict['vis_temporal_min'] = np.min(frames)

            # Temporal dynamics
            frame_means = np.mean(frames, axis=1)
            feat_dict['vis_trend'] = np.polyfit(range(len(frame_means)), frame_means, 1)[0]
            feat_dict['vis_velocity'] = np.mean(np.diff(frame_means))
            feat_dict['vis_acceleration'] = np.mean(np.diff(np.diff(frame_means)))

            # Motion features
            frame_diff = np.diff(frames, axis=0)
            feat_dict['vis_motion_magnitude'] = np.mean(np.abs(frame_diff))
            feat_dict['vis_motion_std'] = np.std(frame_diff)

            # First and last frame features (temporal bookends)
            for i in range(min(10, frames.shape[1])):
                feat_dict[f'vis_first_{i}'] = frames[0, i]
                feat_dict[f'vis_last_{i}'] = frames[-1, i]

            # Aggregate frame features
            for i in range(min(10, frames.shape[1])):
                feat_dict[f'vis_agg_{i}'] = np.mean(frames[:, i])

            features.append(feat_dict)

        return pd.DataFrame(features)

    def extract_temporal_audio_features(self, audio_frames_list):
        """
        Extract temporal features from audio

        Args:
            audio_frames_list: List of audio frame sequences

        Returns:
            Temporal audio features
        """
        features = []

        for frames in audio_frames_list:
            feat_dict = {}

            # Statistical features
            feat_dict['aud_temporal_mean'] = np.mean(frames)
            feat_dict['aud_temporal_std'] = np.std(frames)
            feat_dict['aud_temporal_max'] = np.max(frames)
            feat_dict['aud_temporal_min'] = np.min(frames)

            # Temporal dynamics
            frame_means = np.mean(frames, axis=1)
            feat_dict['aud_trend'] = np.polyfit(range(len(frame_means)), frame_means, 1)[0]
            feat_dict['aud_velocity'] = np.mean(np.diff(frame_means))

            # Rhythm features
            frame_energies = np.sum(frames**2, axis=1)
            feat_dict['aud_energy_mean'] = np.mean(frame_energies)
            feat_dict['aud_energy_std'] = np.std(frame_energies)
            feat_dict['aud_rhythm_regularity'] = np.std(np.diff(frame_energies))

            # Spectral features (simulated)
            feat_dict['aud_spectral_centroid'] = np.mean(np.sum(frames * np.arange(frames.shape[1]), axis=1))
            feat_dict['aud_spectral_spread'] = np.std(frames)

            # Aggregate audio features
            for i in range(min(10, frames.shape[1])):
                feat_dict[f'aud_agg_{i}'] = np.mean(frames[:, i])

            features.append(feat_dict)

        return pd.DataFrame(features)

    def attention_fusion(self, visual_features, audio_features):
        """
        Attention-based fusion: Learn to weight modalities

        Args:
            visual_features: Visual feature matrix
            audio_features: Audio feature matrix

        Returns:
            Fused features with attention
        """
        # Simple attention mechanism: compute importance weights
        vis_importance = np.mean(np.abs(visual_features), axis=0, keepdims=True)
        aud_importance = np.mean(np.abs(audio_features), axis=0, keepdims=True)

        # Normalize importances
        vis_weights = vis_importance / (vis_importance + 1e-8)
        aud_weights = aud_importance / (aud_importance + 1e-8)

        # Apply attention weights
        weighted_vis = visual_features * vis_weights
        weighted_aud = audio_features * aud_weights

        # Concatenate weighted features
        return np.concatenate([weighted_vis, weighted_aud], axis=1)

    def temporal_fusion(self, visual_features, audio_features):
        """
        Temporal fusion: Emphasize temporal alignment

        Args:
            visual_features: Visual features
            audio_features: Audio features

        Returns:
            Temporally aligned fused features
        """
        # Cross-modal temporal correlation
        correlation = np.corrcoef(visual_features.T, audio_features.T)
        vis_aud_corr = correlation[:visual_features.shape[1], visual_features.shape[1]:]

        # Use correlation as fusion weights
        fusion_weights = np.mean(np.abs(vis_aud_corr), axis=1, keepdims=True).T

        # Weight and concatenate
        weighted_vis = visual_features * fusion_weights[:, :visual_features.shape[1]]
        concatenated = np.concatenate([weighted_vis, audio_features], axis=1)

        return concatenated

    def train(self, df):
        """
        Train the multi-modal video understanding model

        Args:
            df: DataFrame with video data
        """
        # Encode labels
        y = self.label_encoder.fit_transform(df['action'])

        # Extract features
        X_vis = self.extract_temporal_visual_features(df['visual_frames'])
        X_aud = self.extract_temporal_audio_features(df['audio_frames'])

        # Apply fusion
        if self.fusion_type == 'concat':
            X_fused = np.concatenate([X_vis.values, X_aud.values], axis=1)
        elif self.fusion_type == 'attention':
            X_fused = self.attention_fusion(X_vis.values, X_aud.values)
        elif self.fusion_type == 'temporal':
            X_fused = self.temporal_fusion(X_vis.values, X_aud.values)

        # Train fusion model
        self.fusion_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.fusion_model.fit(X_fused, y)

        # Also train modality-specific models for comparison
        self.visual_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.visual_model.fit(X_vis.values, y)

        self.audio_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.audio_model.fit(X_aud.values, y)

        return X_vis, X_aud, y

    def predict(self, df):
        """
        Predict actions from videos

        Args:
            df: DataFrame with video data

        Returns:
            Predictions
        """
        X_vis = self.extract_temporal_visual_features(df['visual_frames'])
        X_aud = self.extract_temporal_audio_features(df['audio_frames'])

        if self.fusion_type == 'concat':
            X_fused = np.concatenate([X_vis.values, X_aud.values], axis=1)
        elif self.fusion_type == 'attention':
            X_fused = self.attention_fusion(X_vis.values, X_aud.values)
        elif self.fusion_type == 'temporal':
            X_fused = self.temporal_fusion(X_vis.values, X_aud.values)

        return self.label_encoder.inverse_transform(self.fusion_model.predict(X_fused))

    def evaluate(self, df):
        """Evaluate model on test data"""
        y_true = df['action']
        y_pred = self.predict(df)
        return accuracy_score(y_true, y_pred)


def plot_video_analysis(results):
    """Visualize video understanding results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Fusion strategy comparison
    ax = axes[0, 0]
    strategies = ['Visual Only', 'Audio Only', 'Concat', 'Attention', 'Temporal']
    accuracies = [
        results['ablation']['Visual Only'],
        results['ablation']['Audio Only'],
        results['fusion']['concat'],
        results['fusion']['attention'],
        results['fusion']['temporal']
    ]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#2ecc71', '#f39c12']
    bars = ax.bar(strategies, accuracies, color=colors)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Video Understanding: Fusion Strategy Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=45)
    for i, (bar, v) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.3f}', ha='center', fontweight='bold')

    # Confusion matrix
    ax = axes[0, 1]
    best_fusion = max(results['fusion'].items(), key=lambda x: x[1])[0]
    cm = results['confusion_matrices'][best_fusion]
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=results['actions'], yticklabels=results['actions'])
    ax.set_title(f'Confusion Matrix - {best_fusion.capitalize()} Fusion',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Action', fontsize=12)
    ax.set_xlabel('Predicted Action', fontsize=12)

    # Per-action performance
    ax = axes[1, 0]
    report = results['classification_reports'][best_fusion]
    actions = results['actions']
    f1_scores = [report[action]['f1-score'] for action in actions]
    bars = ax.barh(actions, f1_scores, color='#3498db')
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title(f'Per-Action Performance ({best_fusion.capitalize()})',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    for bar, score in zip(bars, f1_scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontweight='bold')

    # Modality contribution
    ax = axes[1, 1]
    modalities = ['Visual\nOnly', 'Audio\nOnly', 'Early\nFusion', 'Attention\nFusion', 'Temporal\nFusion']
    improvements = [
        0,  # baseline
        results['ablation']['Audio Only'] - results['ablation']['Visual Only'],
        results['fusion']['concat'] - results['ablation']['Visual Only'],
        results['fusion']['attention'] - results['ablation']['Visual Only'],
        results['fusion']['temporal'] - results['ablation']['Visual Only']
    ]
    colors_imp = ['#95a5a6' if x <= 0 else '#2ecc71' for x in improvements]
    bars = ax.bar(modalities, improvements, color=colors_imp)
    ax.set_ylabel('Improvement over Visual Only', fontsize=12)
    ax.set_title('Modality Contribution Analysis', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, imp + 0.01,
                f'{imp:+.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('video_understanding_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: video_understanding_analysis.png")


def run_comprehensive_evaluation(train_df, test_df):
    """Run comprehensive evaluation across all fusion strategies"""
    results = {
        'fusion': {},
        'ablation': {},
        'confusion_matrices': {},
        'classification_reports': {},
        'actions': None
    }

    # Test each fusion strategy
    for fusion_type in ['concat', 'attention', 'temporal']:
        print(f"\n   Training {fusion_type} fusion model...")
        model = VideoUnderstandingModel(fusion_type=fusion_type)
        model.train(train_df)

        y_pred = model.predict(test_df)
        y_true = test_df['action']

        acc = accuracy_score(y_true, y_pred)
        results['fusion'][fusion_type] = acc

        # Confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred, labels=model.label_encoder.classes_)
        report = classification_report(y_true, y_pred, output_dict=True)

        results['confusion_matrices'][fusion_type] = cm
        results['classification_reports'][fusion_type] = report

        print(f"   {fusion_type.capitalize()} Fusion Accuracy: {acc:.4f}")

    # Store action names
    results['actions'] = model.label_encoder.classes_

    # Ablation study
    print("\n   Running ablation study...")

    # Visual only
    model_vis = VideoUnderstandingModel(fusion_type='concat')
    model_vis.train(train_df)
    X_vis_test = model_vis.extract_temporal_visual_features(test_df['visual_frames'])
    y_pred_vis = model_vis.visual_model.predict(X_vis_test.values)
    results['ablation']['Visual Only'] = accuracy_score(
        model_vis.label_encoder.transform(test_df['action']), y_pred_vis)

    # Audio only
    X_aud_test = model_vis.extract_temporal_audio_features(test_df['audio_frames'])
    y_pred_aud = model_vis.audio_model.predict(X_aud_test.values)
    results['ablation']['Audio Only'] = accuracy_score(
        model_vis.label_encoder.transform(test_df['action']), y_pred_aud)

    print(f"   Visual Only: {results['ablation']['Visual Only']:.4f}")
    print(f"   Audio Only: {results['ablation']['Audio Only']:.4f}")

    return results


def main():
    """Main execution function"""
    print("=" * 80)
    print("Video Understanding with Multi-Modal Learning")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic video data...")
    model = VideoUnderstandingModel()
    df = model.generate_video_data(n_videos=500, frames_per_video=16)
    print(f"   Generated {len(df)} video samples")
    print(f"   Actions: {', '.join(df['action'].unique())}")
    print(f"   Frames per video: {df['duration'].iloc[0]}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42,
                                          stratify=df['action'])
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # Run comprehensive evaluation
    print("\n2. Comparing multi-modal fusion strategies...")
    results = run_comprehensive_evaluation(train_df, test_df)

    # Find best strategy
    best_strategy = max(results['fusion'].items(), key=lambda x: x[1])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest Fusion Strategy: {best_strategy[0].upper()}")
    print(f"Best Accuracy: {best_strategy[1]:.4f}")

    print(f"\nAblation Study:")
    print(f"  Visual Only:  {results['ablation']['Visual Only']:.4f}")
    print(f"  Audio Only:   {results['ablation']['Audio Only']:.4f}")
    print(f"  Combined:     {best_strategy[1]:.4f}")

    print(f"\nImprovement from Multi-Modal Learning:")
    vis_improvement = (best_strategy[1] - results['ablation']['Visual Only']) * 100
    aud_improvement = (best_strategy[1] - results['ablation']['Audio Only']) * 100
    print(f"  vs Visual Only: +{vis_improvement:.2f}%")
    print(f"  vs Audio Only:  +{aud_improvement:.2f}%")

    # Plot results
    print("\n3. Generating visualizations...")
    plot_video_analysis(results)

    print("\n" + "=" * 80)
    print("Video understanding analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
