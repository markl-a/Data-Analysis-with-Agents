"""
Speech Emotion Recognition from Audio Signals
==============================================

This solution demonstrates emotion recognition from speech using audio features.
We generate synthetic speech-like audio with different emotional characteristics
and classify emotions using machine learning.

Author: Kaggle Solutions Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import signal
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')


class SpeechEmotionRecognizer:
    """Speech emotion recognition using audio features"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'fear']
        self.scaler = StandardScaler()
        self.model = None

    def generate_speech_audio(self, emotion, duration=2.0, num_samples=1):
        """Generate synthetic speech-like audio with emotional characteristics"""
        audios = []
        labels = []

        for _ in range(num_samples):
            t = np.linspace(0, duration, int(self.sample_rate * duration))

            # Base frequency and modulation patterns for different emotions
            if emotion == 'happy':
                # Higher pitch, faster tempo, more variation
                base_freq = np.random.uniform(200, 280)
                pitch_variation = np.random.uniform(40, 60)
                tempo = np.random.uniform(1.5, 2.0)
                energy = np.random.uniform(0.7, 0.9)

            elif emotion == 'sad':
                # Lower pitch, slower tempo, less variation
                base_freq = np.random.uniform(100, 150)
                pitch_variation = np.random.uniform(10, 20)
                tempo = np.random.uniform(0.5, 0.8)
                energy = np.random.uniform(0.3, 0.5)

            elif emotion == 'angry':
                # Variable pitch, fast tempo, high energy
                base_freq = np.random.uniform(150, 220)
                pitch_variation = np.random.uniform(50, 80)
                tempo = np.random.uniform(1.8, 2.5)
                energy = np.random.uniform(0.8, 1.0)

            elif emotion == 'fear':
                # High pitch, trembling effect
                base_freq = np.random.uniform(220, 300)
                pitch_variation = np.random.uniform(60, 90)
                tempo = np.random.uniform(1.2, 1.8)
                energy = np.random.uniform(0.5, 0.7)

            else:  # neutral
                # Moderate everything
                base_freq = np.random.uniform(150, 200)
                pitch_variation = np.random.uniform(15, 25)
                tempo = np.random.uniform(1.0, 1.3)
                energy = np.random.uniform(0.5, 0.6)

            # Create speech-like waveform with formants
            audio = np.zeros_like(t)

            # Add multiple harmonics to simulate speech formants
            for harmonic in range(1, 6):
                freq_mod = base_freq * harmonic + pitch_variation * np.sin(2 * np.pi * tempo * t)
                audio += (1/harmonic) * np.sin(2 * np.pi * freq_mod * t)

            # Add tremolo for some emotions
            if emotion in ['fear', 'angry']:
                tremolo = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
                audio *= tremolo

            # Apply amplitude envelope
            envelope = np.exp(-2 * t / duration)
            audio *= envelope * energy

            # Add slight noise for realism
            noise = np.random.normal(0, 0.02, len(audio))
            audio += noise

            # Normalize
            audio = audio / np.max(np.abs(audio))

            audios.append(audio)
            labels.append(emotion)

        return audios, labels

    def extract_features(self, audio):
        """Extract audio features for emotion recognition"""
        features = []

        # 1. Spectral features
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])

        # Spectral centroid
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]
        spectral_centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag)
        features.append(spectral_centroid)

        # Spectral bandwidth
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_mag) / np.sum(fft_mag))
        features.append(spectral_spread)

        # Spectral rolloff
        cumsum = np.cumsum(fft_mag)
        rolloff = freqs[np.where(cumsum >= 0.85 * cumsum[-1])[0][0]]
        features.append(rolloff)

        # 2. Time-domain features
        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)

        # Energy
        energy = np.sum(audio**2) / len(audio)
        features.append(energy)

        # RMS
        rms = np.sqrt(np.mean(audio**2))
        features.append(rms)

        # 3. MFCC-like features (simplified)
        # Power in different frequency bands
        n_bands = 13
        for i in range(n_bands):
            band_start = i * len(fft_mag) // n_bands
            band_end = (i + 1) * len(fft_mag) // n_bands
            band_power = np.sum(fft_mag[band_start:band_end])
            features.append(band_power)

        # 4. Statistical features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.min(audio),
            np.max(audio),
            np.percentile(audio, 25),
            np.percentile(audio, 75)
        ])

        return np.array(features)

    def train(self, X, y):
        """Train the emotion recognition model"""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

    def predict(self, audio):
        """Predict emotion from audio"""
        features = self.extract_features(audio).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]

    def predict_proba(self, audio):
        """Get emotion probabilities"""
        features = self.extract_features(audio).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[0]


def visualize_audio_features(recognizer, audios, labels, emotions):
    """Visualize audio signals and their spectrograms"""
    fig, axes = plt.subplots(len(emotions), 2, figsize=(15, 12))

    for idx, emotion in enumerate(emotions):
        # Find first sample of this emotion
        emotion_idx = labels.index(emotion)
        audio = audios[emotion_idx]

        # Waveform
        t = np.linspace(0, len(audio)/recognizer.sample_rate, len(audio))
        axes[idx, 0].plot(t, audio, linewidth=0.5)
        axes[idx, 0].set_title(f'{emotion.capitalize()} - Waveform')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(audio, recognizer.sample_rate, nperseg=256)
        axes[idx, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        axes[idx, 1].set_title(f'{emotion.capitalize()} - Spectrogram')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Frequency (Hz)')
        axes[idx, 1].set_ylim([0, 2000])

    plt.tight_layout()
    plt.savefig('speech_emotion_audio_features.png', dpi=300, bbox_inches='tight')
    print("Saved: speech_emotion_audio_features.png")
    plt.close()


def visualize_results(y_true, y_pred, emotions):
    """Visualize classification results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=emotions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions, ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted Emotion')
    axes[0].set_ylabel('True Emotion')

    # Accuracy per emotion
    accuracies = []
    for i, emotion in enumerate(emotions):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            accuracies.append(acc)
        else:
            accuracies.append(0)

    axes[1].bar(emotions, accuracies, color='steelblue', alpha=0.7)
    axes[1].set_title('Per-Emotion Accuracy')
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, acc in enumerate(accuracies):
        axes[1].text(i, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('speech_emotion_results.png', dpi=300, bbox_inches='tight')
    print("Saved: speech_emotion_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Speech Emotion Recognition")
    print("=" * 60)

    # Initialize recognizer
    recognizer = SpeechEmotionRecognizer(sample_rate=16000)

    # Generate dataset
    print("\n1. Generating synthetic speech data...")
    all_audios = []
    all_labels = []

    for emotion in recognizer.emotions:
        audios, labels = recognizer.generate_speech_audio(emotion, duration=2.0, num_samples=50)
        all_audios.extend(audios)
        all_labels.extend(labels)

    print(f"   Generated {len(all_audios)} audio samples")
    print(f"   Emotions: {recognizer.emotions}")

    # Extract features
    print("\n2. Extracting audio features...")
    X = np.array([recognizer.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training emotion recognition model...")
    recognizer.train(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model...")
    train_audios = [all_audios[i] for i in range(len(all_audios)) if i < len(X_train)]
    test_indices = list(range(len(X_train), len(all_audios)))

    y_pred = recognizer.model.predict(recognizer.scaler.transform(X_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualizations
    print("\n5. Creating visualizations...")
    visualize_audio_features(recognizer, all_audios, all_labels, recognizer.emotions)
    visualize_results(y_test, y_pred, recognizer.emotions)

    # Demo prediction
    print("\n6. Demo: Predicting emotion for sample audio...")
    test_audio, test_label = recognizer.generate_speech_audio('happy', num_samples=1)
    predicted_emotion = recognizer.predict(test_audio[0])
    probabilities = recognizer.predict_proba(test_audio[0])

    print(f"\n   True emotion: {test_label[0]}")
    print(f"   Predicted emotion: {predicted_emotion}")
    print("\n   Probabilities:")
    for emotion, prob in zip(recognizer.emotions, probabilities):
        print(f"   {emotion:10s}: {prob:.3f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
