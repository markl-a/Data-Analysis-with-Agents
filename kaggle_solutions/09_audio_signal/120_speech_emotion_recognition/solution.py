"""
Speech Emotion Recognition - Audio Signal Processing

This module implements speech emotion recognition using acoustic
features extracted from audio signals.

Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
Difficulty: ⭐⭐⭐ Advanced Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class SpeechEmotionRecognizer:
    """Speech Emotion Recognition using Acoustic Features."""

    EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_names: List[str] = []

    def create_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic audio feature dataset."""
        np.random.seed(42)
        n_samples_per_emotion = 200
        n_mfcc = 40
        n_chroma = 12
        n_contrast = 7

        features_list = []
        labels_list = []

        # Define characteristic patterns for each emotion
        emotion_patterns = {
            'neutral': {'energy': 0.5, 'pitch_var': 0.3, 'tempo': 1.0},
            'calm': {'energy': 0.3, 'pitch_var': 0.2, 'tempo': 0.8},
            'happy': {'energy': 0.8, 'pitch_var': 0.7, 'tempo': 1.3},
            'sad': {'energy': 0.3, 'pitch_var': 0.4, 'tempo': 0.7},
            'angry': {'energy': 0.9, 'pitch_var': 0.8, 'tempo': 1.4},
            'fearful': {'energy': 0.6, 'pitch_var': 0.9, 'tempo': 1.2},
            'disgust': {'energy': 0.5, 'pitch_var': 0.5, 'tempo': 0.9},
            'surprised': {'energy': 0.7, 'pitch_var': 0.8, 'tempo': 1.1}
        }

        for emotion in self.EMOTIONS:
            pattern = emotion_patterns[emotion]

            for _ in range(n_samples_per_emotion):
                # MFCC features (mean and std)
                mfcc_mean = np.random.normal(0, 10, n_mfcc) * pattern['pitch_var']
                mfcc_std = np.abs(np.random.normal(5, 2, n_mfcc))

                # Delta MFCC
                mfcc_delta_mean = np.random.normal(0, 3, n_mfcc) * pattern['tempo']
                mfcc_delta_std = np.abs(np.random.normal(2, 1, n_mfcc))

                # Chroma features
                chroma_mean = np.random.uniform(0.2, 0.8, n_chroma)
                chroma_std = np.random.uniform(0.05, 0.2, n_chroma)

                # Mel spectrogram features (summarized)
                mel_mean = np.random.normal(-20, 10, 20) * pattern['energy']
                mel_std = np.abs(np.random.normal(5, 2, 20))

                # Spectral contrast
                contrast = np.random.normal(25, 10, n_contrast) * pattern['energy']

                # Zero crossing rate
                zcr_mean = 0.05 + 0.1 * pattern['tempo'] + np.random.normal(0, 0.02)
                zcr_std = 0.02 + np.random.normal(0, 0.005)

                # RMS energy
                rms_mean = pattern['energy'] * 0.1 + np.random.normal(0, 0.02)
                rms_std = 0.02 + np.random.normal(0, 0.005)

                # Pitch features
                pitch_mean = 150 + 100 * pattern['pitch_var'] + np.random.normal(0, 20)
                pitch_std = 30 * pattern['pitch_var'] + np.random.normal(0, 10)

                # Tempo
                tempo = 100 * pattern['tempo'] + np.random.normal(0, 15)

                # Combine all features
                features = np.concatenate([
                    mfcc_mean, mfcc_std, mfcc_delta_mean, mfcc_delta_std,
                    chroma_mean, chroma_std,
                    mel_mean, mel_std,
                    contrast,
                    [zcr_mean, zcr_std, rms_mean, rms_std, pitch_mean, pitch_std, tempo]
                ])

                features_list.append(features)
                labels_list.append(emotion)

        return np.array(features_list), np.array(labels_list)

    def create_feature_names(self) -> List[str]:
        """Create feature name list."""
        names = []

        # MFCC
        for i in range(40):
            names.append(f'mfcc_{i}_mean')
        for i in range(40):
            names.append(f'mfcc_{i}_std')
        for i in range(40):
            names.append(f'mfcc_delta_{i}_mean')
        for i in range(40):
            names.append(f'mfcc_delta_{i}_std')

        # Chroma
        for i in range(12):
            names.append(f'chroma_{i}_mean')
        for i in range(12):
            names.append(f'chroma_{i}_std')

        # Mel
        for i in range(20):
            names.append(f'mel_{i}_mean')
        for i in range(20):
            names.append(f'mel_{i}_std')

        # Contrast
        for i in range(7):
            names.append(f'contrast_{i}')

        # Other
        names.extend(['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std',
                     'pitch_mean', 'pitch_std', 'tempo'])

        self.feature_names = names
        return names

    def plot_analysis(self, X: np.ndarray, y: np.ndarray, output_dir: str = '.') -> None:
        """Generate feature analysis visualizations."""
        df = pd.DataFrame(X, columns=self.create_feature_names())
        df['emotion'] = y

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Speech Emotion Recognition - Feature Analysis', fontsize=16)

        # Emotion distribution
        pd.Series(y).value_counts().plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Emotion Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # MFCC_0 by emotion
        df.boxplot(column='mfcc_0_mean', by='emotion', ax=axes[0, 1])
        axes[0, 1].set_title('MFCC_0 Mean by Emotion')
        plt.suptitle('')

        # RMS Energy by emotion
        df.boxplot(column='rms_mean', by='emotion', ax=axes[0, 2])
        axes[0, 2].set_title('RMS Energy by Emotion')
        plt.suptitle('')

        # Pitch by emotion
        df.boxplot(column='pitch_mean', by='emotion', ax=axes[1, 0])
        axes[1, 0].set_title('Pitch Mean by Emotion')
        plt.suptitle('')

        # Tempo by emotion
        df.boxplot(column='tempo', by='emotion', ax=axes[1, 1])
        axes[1, 1].set_title('Tempo by Emotion')
        plt.suptitle('')

        # ZCR by emotion
        df.boxplot(column='zcr_mean', by='emotion', ax=axes[1, 2])
        axes[1, 2].set_title('Zero Crossing Rate by Emotion')
        plt.suptitle('')

        # Feature correlation (sample)
        sample_features = ['mfcc_0_mean', 'mfcc_1_mean', 'rms_mean', 'pitch_mean', 'tempo', 'zcr_mean']
        sns.heatmap(df[sample_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 0])
        axes[2, 0].set_title('Feature Correlations')

        # MFCC distribution
        axes[2, 1].hist(df['mfcc_0_mean'], bins=50, alpha=0.7, color='coral')
        axes[2, 1].set_title('MFCC_0 Distribution')

        # 2D feature space (using first 2 MFCCs)
        emotions_encoded = LabelEncoder().fit_transform(y)
        scatter = axes[2, 2].scatter(df['mfcc_0_mean'], df['mfcc_1_mean'],
                                     c=emotions_encoded, cmap='tab10', alpha=0.5, s=10)
        axes[2, 2].set_title('MFCC Feature Space')
        axes[2, 2].set_xlabel('MFCC_0')
        axes[2, 2].set_ylabel('MFCC_1')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/emotion_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/emotion_analysis.png")
        plt.close()

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train emotion recognition models."""
        X_scaled = self.scaler.fit_transform(X_train)
        y_encoded = self.label_encoder.fit_transform(y_train)

        print("\nTraining models...")

        # SVM
        self.models['SVM'] = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        self.models['SVM'].fit(X_scaled, y_encoded)

        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_scaled, y_encoded)

        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_scaled, y_encoded)

        # MLP
        self.models['MLP'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=500,
            random_state=42, early_stopping=True
        )
        self.models['MLP'].fit(X_scaled, y_encoded)

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='mlogloss'
            )
            self.models['XGBoost'].fit(X_scaled, y_encoded)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models."""
        X_scaled = self.scaler.transform(X_test)
        y_encoded = self.label_encoder.transform(y_test)
        results = []

        print("\n=== Model Evaluation ===")

        for name, model in self.models.items():
            y_pred = model.predict(X_scaled)

            acc = accuracy_score(y_encoded, y_pred)
            f1_macro = f1_score(y_encoded, y_pred, average='macro')
            f1_weighted = f1_score(y_encoded, y_pred, average='weighted')

            results.append({
                'Model': name,
                'Accuracy': acc,
                'F1-Macro': f1_macro,
                'F1-Weighted': f1_weighted
            })

            print(f"{name}: Acc={acc:.4f}, F1-Macro={f1_macro:.4f}")

        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_results(self, results_df: pd.DataFrame, X_test: np.ndarray,
                    y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize classification results."""
        X_scaled = self.scaler.transform(X_test)
        y_encoded = self.label_encoder.transform(y_test)
        y_pred = self.best_model.predict(X_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Model comparison
        results_df.set_index('Model')[['Accuracy', 'F1-Macro']].plot(
            kind='bar', ax=axes[0, 0], color=['steelblue', 'coral']
        )
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim([0, 1])

        # Confusion matrix
        cm = confusion_matrix(y_encoded, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=self.EMOTIONS, yticklabels=self.EMOTIONS)
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # Per-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)
        axes[1, 0].bar(self.EMOTIONS, class_acc, color='green')
        axes[1, 0].set_title('Per-Class Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim([0, 1])

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[-15:]
            axes[1, 1].barh(range(15), importance[indices], color='purple')
            axes[1, 1].set_yticks(range(15))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_title('Top 15 Features')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/emotion_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/emotion_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("SPEECH EMOTION RECOGNITION")
    print("=" * 70)

    recognizer = SpeechEmotionRecognizer()

    # Create data
    X, y = recognizer.create_sample_data()
    print(f"\nDataset: {X.shape}, {len(recognizer.EMOTIONS)} emotions")

    # Analysis
    recognizer.plot_analysis(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")

    # Train and evaluate
    recognizer.train_models(X_train, y_train)
    results = recognizer.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    recognizer.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best Accuracy: {results.iloc[0]['Accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
