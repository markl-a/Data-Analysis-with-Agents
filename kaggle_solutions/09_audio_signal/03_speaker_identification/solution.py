"""
Speaker Identification System
==============================

This solution demonstrates speaker identification using voice characteristics.
We generate synthetic voices with speaker-specific vocal tract features and
identify speakers using machine learning.

Author: Kaggle Solutions Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import signal
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')


class SpeakerIdentifier:
    """Speaker identification using voice biometrics"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.speakers = ['Speaker_A', 'Speaker_B', 'Speaker_C', 'Speaker_D', 'Speaker_E']
        # Speaker-specific characteristics (fundamental frequency, formants)
        self.speaker_params = {
            'Speaker_A': {'f0': 120, 'formants': [700, 1220, 2600], 'variation': 15},  # Male 1
            'Speaker_B': {'f0': 100, 'formants': [650, 1080, 2500], 'variation': 10},  # Male 2
            'Speaker_C': {'f0': 220, 'formants': [800, 1400, 2800], 'variation': 20},  # Female 1
            'Speaker_D': {'f0': 200, 'formants': [850, 1500, 2900], 'variation': 18},  # Female 2
            'Speaker_E': {'f0': 140, 'formants': [720, 1300, 2650], 'variation': 12},  # Male 3
        }
        self.scaler = StandardScaler()
        self.model = None

    def generate_voice(self, speaker, duration=2.0, num_samples=1):
        """Generate synthetic voice with speaker-specific characteristics"""
        audios = []
        labels = []

        params = self.speaker_params[speaker]

        for _ in range(num_samples):
            t = np.linspace(0, duration, int(self.sample_rate * duration))

            # Fundamental frequency with natural variation
            f0 = params['f0'] + params['variation'] * np.sin(2 * np.pi * 0.5 * t)

            # Generate glottal pulse train
            glottal = signal.square(2 * np.pi * f0 * t)

            # Apply formant filtering (vocal tract resonances)
            audio = np.zeros_like(t)

            for formant_freq in params['formants']:
                # Create resonance at formant frequency
                bandwidth = 50  # Hz
                q_factor = formant_freq / bandwidth

                # Bandpass filter around formant
                sos = signal.butter(4, [formant_freq - bandwidth, formant_freq + bandwidth],
                                   btype='band', fs=self.sample_rate, output='sos')
                formant_signal = signal.sosfilt(sos, glottal)
                audio += formant_signal

            # Add harmonic richness
            for harmonic in range(2, 5):
                audio += 0.5 / harmonic * np.sin(2 * np.pi * harmonic * f0 * t)

            # Apply natural amplitude envelope
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
            audio *= envelope

            # Add slight noise for realism
            noise = np.random.normal(0, 0.01, len(audio))
            audio += noise

            # Normalize
            audio = audio / np.max(np.abs(audio))

            audios.append(audio)
            labels.append(speaker)

        return audios, labels

    def extract_features(self, audio):
        """Extract speaker-discriminative features"""
        features = []

        # FFT analysis
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]

        # 1. Pitch features
        # Fundamental frequency estimation (autocorrelation method)
        autocorr = np.correlate(audio, audio, mode='full')[len(audio)-1:]
        # Look for peak in typical voice range (80-400 Hz)
        min_period = int(self.sample_rate / 400)
        max_period = int(self.sample_rate / 80)
        peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
        f0_estimate = self.sample_rate / peak_idx
        features.append(f0_estimate)

        # Pitch variation (jitter)
        pitch_variation = np.std(autocorr[min_period:max_period])
        features.append(pitch_variation)

        # 2. Formant frequencies (first 5 peaks in spectrum)
        peaks, properties = signal.find_peaks(fft_mag, height=np.mean(fft_mag), distance=20)
        formants = sorted(freqs[peaks][:5]) if len(peaks) >= 5 else list(freqs[peaks]) + [0]*(5-len(peaks))
        features.extend(formants[:5])

        # 3. Spectral features
        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag)
        features.append(spectral_centroid)

        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_mag) / np.sum(fft_mag))
        features.append(spectral_bandwidth)

        # Spectral tilt (slope of spectrum)
        spectral_tilt = np.polyfit(freqs[:1000], fft_mag[:1000], 1)[0]
        features.append(spectral_tilt)

        # 4. MFCC-like features (13 coefficients)
        n_mfcc = 13
        for i in range(n_mfcc):
            band_start = i * len(fft_mag) // n_mfcc
            band_end = (i + 1) * len(fft_mag) // n_mfcc
            mfcc = np.sum(np.log(fft_mag[band_start:band_end] + 1e-10))
            features.append(mfcc)

        # 5. Energy features
        features.extend([
            np.sum(audio**2) / len(audio),  # Energy
            np.sqrt(np.mean(audio**2)),  # RMS
            np.max(np.abs(audio)),  # Peak amplitude
        ])

        # 6. Temporal features
        features.extend([
            np.mean(np.abs(np.diff(audio))),  # Zero crossing rate
            np.std(audio),  # Standard deviation
        ])

        return np.array(features)

    def train(self, X, y):
        """Train the speaker identification model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        self.model.fit(X_scaled, y)

    def identify_speaker(self, audio):
        """Identify speaker from audio"""
        features = self.extract_features(audio).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]

    def get_probabilities(self, audio):
        """Get identification probabilities"""
        features = self.extract_features(audio).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[0]


def visualize_voice_characteristics(identifier, audios, labels, speakers):
    """Visualize voice characteristics for each speaker"""
    fig, axes = plt.subplots(len(speakers), 3, figsize=(18, 12))

    for idx, speaker in enumerate(speakers):
        speaker_idx = labels.index(speaker)
        audio = audios[speaker_idx]

        t = np.linspace(0, len(audio)/identifier.sample_rate, len(audio))

        # Waveform
        axes[idx, 0].plot(t, audio, linewidth=0.5, color='darkblue')
        axes[idx, 0].set_title(f'{speaker} - Waveform')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # Spectrum
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/identifier.sample_rate)[:len(audio)//2]

        axes[idx, 1].plot(freqs, fft_mag, linewidth=0.7, color='darkred')
        axes[idx, 1].set_title(f'{speaker} - Frequency Spectrum')
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_xlim([0, 3000])
        axes[idx, 1].grid(True, alpha=0.3)

        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(audio, identifier.sample_rate, nperseg=256)
        axes[idx, 2].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                               shading='gouraud', cmap='viridis')
        axes[idx, 2].set_title(f'{speaker} - Spectrogram')
        axes[idx, 2].set_xlabel('Time (s)')
        axes[idx, 2].set_ylabel('Frequency (Hz)')
        axes[idx, 2].set_ylim([0, 3000])

    plt.tight_layout()
    plt.savefig('speaker_voice_characteristics.png', dpi=300, bbox_inches='tight')
    print("Saved: speaker_voice_characteristics.png")
    plt.close()


def visualize_results(y_true, y_pred, speakers, identifier):
    """Visualize identification results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=speakers)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=speakers, yticklabels=speakers, ax=axes[0])
    axes[0].set_title('Speaker Identification Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Speaker')
    axes[0].set_ylabel('True Speaker')

    # Speaker-specific accuracy
    accuracies = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(len(speakers))]

    # Add F0 (pitch) info
    f0_values = [identifier.speaker_params[s]['f0'] for s in speakers]

    x = np.arange(len(speakers))
    width = 0.35

    ax2 = axes[1]
    ax2.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.7)

    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, f0_values, width, label='F0 (Hz)', color='coral', alpha=0.7)

    ax2.set_xlabel('Speaker')
    ax2.set_ylabel('Accuracy', color='steelblue')
    ax2_twin.set_ylabel('Fundamental Frequency (Hz)', color='coral')
    ax2.set_title('Speaker Accuracy and Pitch', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(speakers, rotation=45)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')

    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('speaker_identification_results.png', dpi=300, bbox_inches='tight')
    print("Saved: speaker_identification_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Speaker Identification System")
    print("=" * 60)

    # Initialize identifier
    identifier = SpeakerIdentifier(sample_rate=16000)

    # Generate dataset
    print("\n1. Generating synthetic voice data...")
    all_audios = []
    all_labels = []

    for speaker in identifier.speakers:
        audios, labels = identifier.generate_voice(speaker, duration=2.0, num_samples=30)
        all_audios.extend(audios)
        all_labels.extend(labels)

    print(f"   Generated {len(all_audios)} voice samples")
    print(f"   Speakers: {identifier.speakers}")

    # Display speaker parameters
    print("\n   Speaker Characteristics:")
    for speaker, params in identifier.speaker_params.items():
        print(f"   {speaker}: F0={params['f0']} Hz, Formants={params['formants']} Hz")

    # Extract features
    print("\n2. Extracting voice features...")
    X = np.array([identifier.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training speaker identification model...")
    identifier.train(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = identifier.model.predict(identifier.scaler.transform(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Overall Accuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualizations
    print("\n5. Creating visualizations...")
    visualize_voice_characteristics(identifier, all_audios, all_labels, identifier.speakers)
    visualize_results(y_test, y_pred, identifier.speakers, identifier)

    # Demo identification
    print("\n6. Demo: Identifying speaker from sample...")
    test_audio, test_label = identifier.generate_voice('Speaker_C', num_samples=1)
    identified_speaker = identifier.identify_speaker(test_audio[0])
    probabilities = identifier.get_probabilities(test_audio[0])

    print(f"\n   True speaker: {test_label[0]}")
    print(f"   Identified speaker: {identified_speaker}")
    print("\n   Identification probabilities:")
    for speaker, prob in zip(identifier.speakers, probabilities):
        print(f"   {speaker:12s}: {prob:.3f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
