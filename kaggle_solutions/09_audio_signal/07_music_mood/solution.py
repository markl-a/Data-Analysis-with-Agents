"""
Music Mood Classification
==========================

This solution demonstrates mood classification from music audio.
We generate synthetic music with different mood characteristics
and classify moods using machine learning.

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


class MusicMoodClassifier:
    """Music mood classification using audio features"""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.moods = ['happy', 'sad', 'energetic', 'calm', 'angry']
        self.scaler = StandardScaler()
        self.model = None

    def generate_mood_music(self, mood, duration=4.0, num_samples=1):
        """Generate synthetic music with mood-specific characteristics"""
        audios = []
        labels = []

        for _ in range(num_samples):
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.zeros_like(t)

            if mood == 'happy':
                # Major key, upbeat tempo, bright timbre
                # C major chord progression
                chords = [
                    [262, 330, 392],  # C major
                    [392, 494, 587],  # G major
                    [349, 440, 523],  # F major
                    [262, 330, 392]   # C major
                ]
                tempo = np.random.uniform(2.5, 3.5)  # Fast
                brightness = 0.8

                for i, chord in enumerate(chords):
                    start = i * duration / 4
                    end = (i + 1) * duration / 4
                    chord_mask = (t >= start) & (t < end)

                    for freq in chord:
                        audio[chord_mask] += np.sin(2 * np.pi * freq * t[chord_mask])

                # Add bright harmonics
                for harmonic in range(2, 5):
                    audio += brightness * (1/harmonic) * np.sin(2 * np.pi * 262 * harmonic * t)

                # Upbeat rhythm
                rhythm = 1 + 0.5 * signal.square(2 * np.pi * tempo * t)
                audio *= rhythm

            elif mood == 'sad':
                # Minor key, slow tempo, dark timbre
                # A minor chord progression
                chords = [
                    [220, 262, 330],  # A minor
                    [196, 247, 294],  # G major
                    [262, 330, 392],  # C major
                    [220, 262, 330]   # A minor
                ]
                tempo = np.random.uniform(0.8, 1.2)  # Slow
                brightness = 0.3

                for i, chord in enumerate(chords):
                    start = i * duration / 4
                    end = (i + 1) * duration / 4
                    chord_mask = (t >= start) & (t < end)

                    for freq in chord:
                        audio[chord_mask] += np.sin(2 * np.pi * freq * t[chord_mask])

                # Dark, mellow sound
                for harmonic in range(1, 3):
                    audio += brightness * (1/harmonic) * np.sin(2 * np.pi * 220 * harmonic * t)

                # Slow, gentle rhythm
                envelope = 0.5 + 0.5 * np.sin(2 * np.pi * tempo * t)
                audio *= envelope

            elif mood == 'energetic':
                # Fast tempo, high energy, complex rhythm
                tempo = np.random.uniform(3.0, 4.5)  # Very fast
                energy = 1.0

                # Power chords with distortion
                for freq in [165, 220, 330]:
                    audio += np.sin(2 * np.pi * freq * t)

                # Add distortion
                audio = np.tanh(audio * 2)

                # Driving beat
                beat = signal.square(2 * np.pi * tempo * t)
                kick = signal.square(2 * np.pi * tempo/2 * t)
                audio = audio * beat + 0.5 * kick

                # High energy throughout
                audio *= energy

            elif mood == 'calm':
                # Slow tempo, simple harmony, soft dynamics
                tempo = np.random.uniform(0.5, 1.0)  # Very slow

                # Soft, sustained chords
                for freq in [262, 330, 392, 494]:  # C major 7th
                    phase = np.random.uniform(0, 2*np.pi)
                    audio += np.sin(2 * np.pi * freq * t + phase)

                # Add ambient pad
                for i in range(3):
                    freq = np.random.uniform(150, 400)
                    audio += 0.3 * np.sin(2 * np.pi * freq * t)

                # Gentle swells
                swell = 0.3 + 0.2 * np.sin(2 * np.pi * tempo * t)
                audio *= swell

                # Soft overall
                audio *= 0.5

            elif mood == 'angry':
                # Dissonant, aggressive, harsh timbre
                tempo = np.random.uniform(2.5, 3.5)  # Fast-medium

                # Dissonant intervals
                for freq in [100, 147, 220, 311]:  # Tritone intervals
                    audio += np.sin(2 * np.pi * freq * t)

                # Heavy distortion
                audio = np.tanh(audio * 3)

                # Aggressive rhythm
                aggression = signal.square(2 * np.pi * tempo * t)
                audio *= (1 + aggression)

                # Add noise for harshness
                audio += 0.2 * np.random.normal(0, 1, len(t))

            # Add slight variation
            noise = np.random.normal(0, 0.01, len(audio))
            audio += noise

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            audios.append(audio)
            labels.append(mood)

        return audios, labels

    def extract_features(self, audio):
        """Extract mood-related musical features"""
        features = []

        # FFT analysis
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]

        # 1. Valence features (happiness/sadness)
        # Mode (major/minor) indicated by spectral centroid
        spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        features.append(spectral_centroid)

        # Brightness (high frequency content)
        high_freq_energy = np.sum(fft_mag[freqs > 2000])
        total_energy = np.sum(fft_mag) + 1e-10
        brightness = high_freq_energy / total_energy
        features.append(brightness)

        # 2. Arousal features (energy/calmness)
        # Tempo estimation
        onset_env = np.abs(np.diff(audio))
        autocorr = np.correlate(onset_env, onset_env, mode='full')[len(onset_env)-1:]
        tempo_estimate = np.argmax(autocorr[10:200]) + 10
        features.append(tempo_estimate)

        # Overall energy
        rms_energy = np.sqrt(np.mean(audio**2))
        features.append(rms_energy)

        # Dynamic range
        dynamic_range = np.max(np.abs(audio)) - np.mean(np.abs(audio))
        features.append(dynamic_range)

        # 3. Timbre features
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_mag) /
                                     (np.sum(fft_mag) + 1e-10))
        features.append(spectral_bandwidth)

        # Spectral rolloff
        cumsum = np.cumsum(fft_mag)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            rolloff = 0
        features.append(rolloff)

        # Spectral flatness (noise vs tonal)
        geometric_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
        arithmetic_mean = np.mean(fft_mag)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(flatness)

        # 4. Harmony features
        # Harmonic-to-noise ratio
        peaks, _ = signal.find_peaks(fft_mag, height=np.mean(fft_mag))
        if len(peaks) > 0:
            harmonic_energy = np.sum(fft_mag[peaks])
            hnr = harmonic_energy / (total_energy + 1e-10)
        else:
            hnr = 0
        features.append(hnr)

        # Dissonance (variance in peak spacing)
        if len(peaks) > 1:
            peak_freqs = freqs[peaks]
            intervals = np.diff(peak_freqs)
            dissonance = np.std(intervals) / (np.mean(intervals) + 1e-10)
        else:
            dissonance = 0
        features.append(dissonance)

        # 5. Rhythm features
        # Beat strength
        beat_strength = np.max(autocorr[10:200]) / (np.mean(autocorr) + 1e-10)
        features.append(beat_strength)

        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)

        # 6. MFCC features (12 coefficients)
        n_mfcc = 12
        for i in range(n_mfcc):
            band_start = i * len(fft_mag) // n_mfcc
            band_end = (i + 1) * len(fft_mag) // n_mfcc
            if band_end > band_start:
                mfcc = np.sum(np.log(fft_mag[band_start:band_end] + 1e-10))
            else:
                mfcc = 0
            features.append(mfcc)

        # 7. Temporal features
        # Attack time (how quickly sound starts)
        envelope = np.abs(signal.hilbert(audio))
        max_idx = np.argmax(envelope)
        attack_time = max_idx / len(envelope)
        features.append(attack_time)

        # 8. Statistical features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.percentile(audio, 25),
            np.percentile(audio, 75)
        ])

        return np.array(features)

    def train(self, X, y):
        """Train the mood classification model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)


def visualize_mood_audio(classifier, audios, labels, moods):
    """Visualize audio characteristics for different moods"""
    fig, axes = plt.subplots(len(moods), 2, figsize=(15, 12))

    for idx, mood in enumerate(moods):
        mood_idx = labels.index(mood)
        audio = audios[mood_idx]

        t = np.linspace(0, len(audio)/classifier.sample_rate, len(audio))

        # Waveform with envelope
        axes[idx, 0].plot(t, audio, linewidth=0.3, alpha=0.7, color='steelblue')
        envelope = np.abs(signal.hilbert(audio))
        axes[idx, 0].plot(t, envelope, linewidth=1.5, color='red', label='Envelope')
        axes[idx, 0].plot(t, -envelope, linewidth=1.5, color='red')
        axes[idx, 0].set_title(f'{mood.capitalize()} - Waveform', fontweight='bold')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend()

        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(audio, classifier.sample_rate, nperseg=512)
        axes[idx, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                               shading='gouraud', cmap='magma')
        axes[idx, 1].set_title(f'{mood.capitalize()} - Spectrogram', fontweight='bold')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Frequency (Hz)')
        axes[idx, 1].set_ylim([0, 3000])

    plt.tight_layout()
    plt.savefig('music_mood_audio.png', dpi=300, bbox_inches='tight')
    print("Saved: music_mood_audio.png")
    plt.close()


def visualize_results(y_true, y_pred, moods):
    """Visualize mood classification results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=moods)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu_r',
                xticklabels=moods, yticklabels=moods, ax=axes[0])
    axes[0].set_title('Mood Classification Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Mood')
    axes[0].set_ylabel('True Mood')

    # Accuracy and arousal/valence
    accuracies = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                 for i in range(len(moods))]

    # Arousal-Valence model
    arousal_valence = {
        'happy': (0.8, 0.8),      # High arousal, positive valence
        'sad': (0.2, 0.2),        # Low arousal, negative valence
        'energetic': (0.9, 0.6),  # Very high arousal, positive valence
        'calm': (0.2, 0.6),       # Low arousal, positive valence
        'angry': (0.8, 0.1)       # High arousal, negative valence
    }

    for mood, (arousal, valence) in arousal_valence.items():
        idx = moods.index(mood)
        size = accuracies[idx] * 1000
        axes[1].scatter(valence, arousal, s=size, alpha=0.6, label=mood)
        axes[1].text(valence + 0.02, arousal, f'{mood}\n({accuracies[idx]:.2f})',
                    fontsize=9, ha='left')

    axes[1].set_xlabel('Valence (Negative ← → Positive)', fontsize=12)
    axes[1].set_ylabel('Arousal (Calm ← → Energetic)', fontsize=12)
    axes[1].set_title('Mood Distribution (Arousal-Valence Model)', fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('music_mood_results.png', dpi=300, bbox_inches='tight')
    print("Saved: music_mood_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Music Mood Classification")
    print("=" * 60)

    # Initialize classifier
    classifier = MusicMoodClassifier(sample_rate=22050)

    # Generate dataset
    print("\n1. Generating synthetic music with different moods...")
    all_audios = []
    all_labels = []

    for mood in classifier.moods:
        audios, labels = classifier.generate_mood_music(mood, duration=4.0, num_samples=35)
        all_audios.extend(audios)
        all_labels.extend(labels)

    print(f"   Generated {len(all_audios)} music samples")
    print(f"   Moods: {classifier.moods}")

    # Extract features
    print("\n2. Extracting music mood features...")
    X = np.array([classifier.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training mood classification model...")
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
    visualize_mood_audio(classifier, all_audios, all_labels, classifier.moods)
    visualize_results(y_test, y_pred, classifier.moods)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
