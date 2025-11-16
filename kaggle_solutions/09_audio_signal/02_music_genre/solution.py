"""
Music Genre Classification
===========================

This solution demonstrates music genre classification using audio features.
We generate synthetic music with different genre characteristics and classify
genres using machine learning.

Author: Kaggle Solutions Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import signal
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')


class MusicGenreClassifier:
    """Music genre classification using audio features"""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.genres = ['rock', 'classical', 'jazz', 'electronic', 'hip_hop']
        self.scaler = StandardScaler()
        self.model = None

    def generate_music(self, genre, duration=3.0, num_samples=1):
        """Generate synthetic music with genre-specific characteristics"""
        audios = []
        labels = []

        for _ in range(num_samples):
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.zeros_like(t)

            if genre == 'rock':
                # Distorted guitar-like sound with strong beat
                # Power chords (multiple frequencies)
                for freq in [220, 277, 330]:  # A, C#, E
                    audio += np.sin(2 * np.pi * freq * t)
                # Add distortion (harmonic saturation)
                audio = np.tanh(audio * 2)
                # Strong drum beat
                beat_freq = 2.0  # 120 BPM
                beat = signal.square(2 * np.pi * beat_freq * t) * 0.5
                audio += beat

            elif genre == 'classical':
                # Orchestra-like with smooth harmonics
                # String section
                for harmonic in [1, 2, 3, 4]:
                    freq = 262 * harmonic  # C major
                    audio += (1/harmonic) * np.sin(2 * np.pi * freq * t)
                # Add vibrato
                vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
                audio *= vibrato
                # Smooth dynamics
                envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
                audio *= envelope

            elif genre == 'jazz':
                # Complex harmonies with swing
                # Jazz chord (7th chords)
                for freq in [220, 277, 330, 392]:  # A7
                    phase = np.random.uniform(0, 2*np.pi)
                    audio += np.sin(2 * np.pi * freq * t + phase)
                # Swing rhythm
                swing = 1 + 0.3 * signal.square(2 * np.pi * 3 * t)
                audio *= swing
                # Add improvisation-like variations
                mod = 1 + 0.2 * np.sin(2 * np.pi * 0.7 * t)
                audio *= mod

            elif genre == 'electronic':
                # Synthesized sounds with strong bass
                # Bass line (sub-bass)
                bass_freq = 55  # A1
                audio += 2 * np.sin(2 * np.pi * bass_freq * t)
                # Synth lead
                lead_freq = 440 + 50 * np.sin(2 * np.pi * 0.5 * t)
                audio += np.sin(2 * np.pi * lead_freq * t)
                # Electronic beat
                beat = signal.sawtooth(2 * np.pi * 4 * t) * 0.5
                audio += beat
                # Add filter sweep
                cutoff = 0.1 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
                audio *= (1 + cutoff)

            elif genre == 'hip_hop':
                # Heavy bass with rhythmic elements
                # Sub-bass
                bass = 1.5 * np.sin(2 * np.pi * 50 * t)
                audio += bass
                # Snare and kick pattern
                kick = signal.square(2 * np.pi * 2 * t) * 0.8
                snare = signal.square(2 * np.pi * 4 * t + np.pi) * 0.5
                audio += kick + snare
                # Hi-hat
                hihat = np.random.uniform(-0.1, 0.1, len(t))
                hihat *= signal.square(2 * np.pi * 8 * t)
                audio += hihat

            # Add slight noise for realism
            noise = np.random.normal(0, 0.01, len(audio))
            audio += noise

            # Normalize
            audio = audio / np.max(np.abs(audio))

            audios.append(audio)
            labels.append(genre)

        return audios, labels

    def extract_features(self, audio):
        """Extract comprehensive music features"""
        features = []

        # FFT analysis
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]

        # 1. Spectral features
        # Spectral centroid (brightness)
        spectral_centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag)
        features.append(spectral_centroid)

        # Spectral bandwidth
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_mag) / np.sum(fft_mag))
        features.append(spectral_spread)

        # Spectral rolloff
        cumsum = np.cumsum(fft_mag)
        rolloff = freqs[np.where(cumsum >= 0.85 * cumsum[-1])[0][0]]
        features.append(rolloff)

        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
        arithmetic_mean = np.mean(fft_mag)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(flatness)

        # 2. Rhythm features
        # Tempo estimation (simplified)
        onset_env = np.abs(np.diff(audio))
        autocorr = np.correlate(onset_env, onset_env, mode='full')[len(onset_env)-1:]
        tempo_estimate = np.argmax(autocorr[10:100]) + 10
        features.append(tempo_estimate)

        # Beat strength
        beat_strength = np.max(autocorr[10:100]) / np.mean(autocorr)
        features.append(beat_strength)

        # 3. Harmonic features
        # Harmonic-to-noise ratio
        peaks, _ = signal.find_peaks(fft_mag, height=np.mean(fft_mag))
        if len(peaks) > 0:
            harmonic_energy = np.sum(fft_mag[peaks])
            total_energy = np.sum(fft_mag)
            hnr = harmonic_energy / (total_energy + 1e-10)
        else:
            hnr = 0
        features.append(hnr)

        # 4. Energy features in different bands
        # Sub-bass (20-60 Hz)
        sub_bass = np.sum(fft_mag[(freqs >= 20) & (freqs < 60)])
        features.append(sub_bass)

        # Bass (60-250 Hz)
        bass = np.sum(fft_mag[(freqs >= 60) & (freqs < 250)])
        features.append(bass)

        # Mid-range (250-2000 Hz)
        mid = np.sum(fft_mag[(freqs >= 250) & (freqs < 2000)])
        features.append(mid)

        # High (2000+ Hz)
        high = np.sum(fft_mag[freqs >= 2000])
        features.append(high)

        # 5. MFCC-like features
        n_mfcc = 10
        for i in range(n_mfcc):
            band_start = i * len(fft_mag) // n_mfcc
            band_end = (i + 1) * len(fft_mag) // n_mfcc
            mfcc_coeff = np.sum(np.log(fft_mag[band_start:band_end] + 1e-10))
            features.append(mfcc_coeff)

        # 6. Time-domain features
        features.extend([
            np.mean(np.abs(audio)),  # Mean absolute value
            np.std(audio),  # Standard deviation
            np.sqrt(np.mean(audio**2)),  # RMS
            np.mean(np.abs(np.diff(audio))),  # Zero crossing rate
        ])

        return np.array(features)

    def train(self, X, y):
        """Train the genre classification model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)

    def predict(self, audio):
        """Predict genre from audio"""
        features = self.extract_features(audio).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]


def visualize_music_samples(classifier, audios, labels, genres):
    """Visualize music samples"""
    fig, axes = plt.subplots(len(genres), 2, figsize=(15, 12))

    for idx, genre in enumerate(genres):
        genre_idx = labels.index(genre)
        audio = audios[genre_idx]

        # Waveform
        t = np.linspace(0, len(audio)/classifier.sample_rate, len(audio))
        axes[idx, 0].plot(t, audio, linewidth=0.5, color='darkblue', alpha=0.7)
        axes[idx, 0].set_title(f'{genre.capitalize()} - Waveform')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # Frequency spectrum
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/classifier.sample_rate)[:len(audio)//2]

        axes[idx, 1].plot(freqs, fft_mag, linewidth=0.5, color='darkred', alpha=0.7)
        axes[idx, 1].set_title(f'{genre.capitalize()} - Frequency Spectrum')
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_xlim([0, 2000])
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('music_genre_samples.png', dpi=300, bbox_inches='tight')
    print("Saved: music_genre_samples.png")
    plt.close()


def visualize_results(y_true, y_pred, genres, feature_importance, feature_names):
    """Visualize classification results"""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, :])
    cm = confusion_matrix(y_true, y_pred, labels=genres)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=genres, yticklabels=genres, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Genre')
    ax1.set_ylabel('True Genre')

    # Per-genre accuracy
    ax2 = fig.add_subplot(gs[1, 0])
    accuracies = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(genres))]
    colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
    ax2.barh(genres, accuracies, color=colors, alpha=0.7)
    ax2.set_title('Per-Genre Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Accuracy')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')

    # Feature importance
    ax3 = fig.add_subplot(gs[1, 1])
    top_n = 10
    top_indices = np.argsort(feature_importance)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = feature_importance[top_indices]

    ax3.barh(top_features, top_importance, color='steelblue', alpha=0.7)
    ax3.set_title(f'Top {top_n} Important Features', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Importance')
    ax3.grid(True, alpha=0.3, axis='x')

    plt.savefig('music_genre_results.png', dpi=300, bbox_inches='tight')
    print("Saved: music_genre_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Music Genre Classification")
    print("=" * 60)

    # Initialize classifier
    classifier = MusicGenreClassifier(sample_rate=22050)

    # Generate dataset
    print("\n1. Generating synthetic music data...")
    all_audios = []
    all_labels = []

    for genre in classifier.genres:
        audios, labels = classifier.generate_music(genre, duration=3.0, num_samples=40)
        all_audios.extend(audios)
        all_labels.extend(labels)

    print(f"   Generated {len(all_audios)} music samples")
    print(f"   Genres: {classifier.genres}")

    # Extract features
    print("\n2. Extracting music features...")
    X = np.array([classifier.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training genre classification model...")
    classifier.train(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(classifier.model, classifier.scaler.transform(X),
                                y, cv=5, scoring='accuracy')
    print(f"   Cross-validation scores: {cv_scores}")
    print(f"   Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = classifier.model.predict(classifier.scaler.transform(X_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualizations
    print("\n5. Creating visualizations...")
    visualize_music_samples(classifier, all_audios, all_labels, classifier.genres)

    feature_names = ['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff',
                     'Spectral Flatness', 'Tempo', 'Beat Strength', 'HNR',
                     'Sub-bass', 'Bass', 'Mid', 'High'] + \
                    [f'MFCC-{i}' for i in range(10)] + \
                    ['Mean Abs', 'Std', 'RMS', 'ZCR']

    visualize_results(y_test, y_pred, classifier.genres,
                     classifier.model.feature_importances_, feature_names)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
