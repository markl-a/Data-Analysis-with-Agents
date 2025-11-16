"""
Simple Speech-to-Text System
=============================

This solution demonstrates a simplified speech-to-text system using
pattern recognition. We generate synthetic speech patterns for digits
and recognize them using machine learning.

Author: Kaggle Solutions Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import signal
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')


class SimpleSpeechRecognizer:
    """Simple speech-to-text for spoken digits"""

    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate
        self.digits = ['zero', 'one', 'two', 'three', 'four',
                      'five', 'six', 'seven', 'eight', 'nine']
        # Formant frequencies for different vowel sounds in digits
        self.digit_formants = {
            'zero': [(400, 800), (900, 1300)],  # /ɪ/, /oʊ/
            'one': [(400, 900), (700, 1100)],   # /w/, /ʌ/
            'two': [(350, 750), (900, 1400)],   # /t/, /u/
            'three': [(500, 1000), (1400, 1800)],  # /θ/, /i/
            'four': [(600, 1100), (1000, 1400)],   # /f/, /ɔ/
            'five': [(450, 900), (700, 1200)],     # /f/, /aɪ/
            'six': [(500, 1000), (700, 1300)],     # /s/, /ɪ/
            'seven': [(550, 1100), (1200, 1700)],  # /s/, /ɛ/
            'eight': [(550, 1050), (1100, 1600)],  # /eɪ/
            'nine': [(450, 950), (700, 1300)]      # /n/, /aɪ/
        }
        self.scaler = StandardScaler()
        self.model = None

    def generate_digit_audio(self, digit, duration=0.8, num_samples=1):
        """Generate synthetic speech for a digit"""
        audios = []
        labels = []

        formants = self.digit_formants[digit]

        for _ in range(num_samples):
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.zeros_like(t)

            # Fundamental frequency (pitch) with variation
            f0_base = np.random.uniform(100, 200)
            f0 = f0_base + 20 * np.sin(2 * np.pi * 3 * t)

            # Generate glottal pulse train
            glottal = signal.square(2 * np.pi * f0 * t)

            # Number of phonemes in digit (simplified)
            n_phonemes = len(formants)
            phoneme_duration = duration / n_phonemes

            for i, (f1_range, f2_range) in enumerate(formants):
                # Time window for this phoneme
                start_time = i * phoneme_duration
                end_time = (i + 1) * phoneme_duration
                phoneme_mask = (t >= start_time) & (t < end_time)

                # Formant frequencies for this phoneme
                f1 = np.random.uniform(*f1_range)
                f2 = np.random.uniform(*f2_range)

                # Create formant resonances
                phoneme_audio = glottal.copy()

                # Apply formant filtering
                for formant_freq in [f1, f2]:
                    # Create resonance
                    resonance = np.sin(2 * np.pi * formant_freq * t)
                    phoneme_audio += 0.5 * resonance

                # Apply to correct time window
                audio[phoneme_mask] += phoneme_audio[phoneme_mask]

            # Apply overall amplitude envelope
            envelope = np.exp(-2 * (t - duration/2)**2 / duration**2)
            audio *= envelope

            # Add slight noise
            noise = np.random.normal(0, 0.02, len(audio))
            audio += noise

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            audios.append(audio)
            labels.append(digit)

        return audios, labels

    def extract_features(self, audio):
        """Extract speech features for digit recognition"""
        features = []

        # FFT analysis
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]

        # 1. Formant frequencies (first 3 peaks)
        peaks, properties = signal.find_peaks(fft_mag, height=np.mean(fft_mag), distance=10)
        formant_freqs = sorted(freqs[peaks][:3]) if len(peaks) >= 3 else \
                       list(freqs[peaks]) + [0]*(3-len(peaks))
        features.extend(formant_freqs)

        # 2. Spectral features
        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        features.append(spectral_centroid)

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

        # 3. MFCC-like features (12 coefficients)
        n_mfcc = 12
        for i in range(n_mfcc):
            band_start = i * len(fft_mag) // n_mfcc
            band_end = (i + 1) * len(fft_mag) // n_mfcc
            if band_end > band_start:
                mfcc = np.sum(np.log(fft_mag[band_start:band_end] + 1e-10))
            else:
                mfcc = 0
            features.append(mfcc)

        # 4. Temporal features
        # Duration-related
        envelope = np.abs(signal.hilbert(audio))
        features.append(np.sum(envelope > 0.1 * np.max(envelope)) / len(envelope))

        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)

        # 5. Energy features
        features.extend([
            np.sum(audio**2) / len(audio),  # Energy
            np.sqrt(np.mean(audio**2)),     # RMS
            np.max(np.abs(audio))            # Peak
        ])

        # 6. Pitch features
        # Autocorrelation for pitch
        autocorr = np.correlate(audio, audio, mode='full')[len(audio)-1:]
        min_period = int(self.sample_rate / 400)
        max_period = int(self.sample_rate / 80)
        if max_period > min_period and max_period < len(autocorr):
            peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
            pitch = self.sample_rate / peak_idx
        else:
            pitch = 150
        features.append(pitch)

        return np.array(features)

    def train(self, X, y):
        """Train the speech recognition model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

    def recognize(self, audio):
        """Recognize digit from audio"""
        features = self.extract_features(audio).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]

    def get_probabilities(self, audio):
        """Get recognition probabilities"""
        features = self.extract_features(audio).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[0]


def visualize_digit_spectrograms(recognizer, audios, labels, digits):
    """Visualize spectrograms for each digit"""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for idx, digit in enumerate(digits):
        digit_idx = labels.index(digit)
        audio = audios[digit_idx]

        # Spectrogram
        f, t, Sxx = signal.spectrogram(audio, recognizer.sample_rate, nperseg=128)
        axes[idx].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                            shading='gouraud', cmap='viridis')
        axes[idx].set_title(f'"{digit}"', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Frequency (Hz)')
        axes[idx].set_ylim([0, 3000])

    plt.suptitle('Speech Spectrograms for Digits 0-9', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('speech_digit_spectrograms.png', dpi=300, bbox_inches='tight')
    print("Saved: speech_digit_spectrograms.png")
    plt.close()


def visualize_results(y_true, y_pred, digits):
    """Visualize recognition results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=digits)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=digits, yticklabels=digits, ax=axes[0])
    axes[0].set_title('Digit Recognition Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Digit')
    axes[0].set_ylabel('True Digit')

    # Per-digit accuracy
    accuracies = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                 for i in range(len(digits))]

    colors = plt.cm.viridis(np.linspace(0, 1, len(digits)))
    axes[1].bar(digits, accuracies, color=colors, alpha=0.7)
    axes[1].set_title('Per-Digit Recognition Accuracy', fontweight='bold')
    axes[1].set_xlabel('Digit')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, acc in enumerate(accuracies):
        axes[1].text(i, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('speech_recognition_results.png', dpi=300, bbox_inches='tight')
    print("Saved: speech_recognition_results.png")
    plt.close()


def demonstrate_recognition(recognizer):
    """Demonstrate recognition with probability distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    test_digits = ['zero', 'three', 'seven']

    for idx, digit in enumerate(test_digits):
        audio, _ = recognizer.generate_digit_audio(digit, num_samples=1)
        audio = audio[0]

        # Waveform
        t = np.linspace(0, len(audio)/recognizer.sample_rate, len(audio))
        axes[idx, 0].plot(t, audio, linewidth=0.7, color='darkblue')
        axes[idx, 0].set_title(f'True: "{digit}" - Waveform')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(audio, recognizer.sample_rate, nperseg=128)
        axes[idx, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                               shading='gouraud', cmap='viridis')
        axes[idx, 1].set_title('Spectrogram')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Frequency (Hz)')
        axes[idx, 1].set_ylim([0, 3000])

        # Recognition probabilities
        recognized = recognizer.recognize(audio)
        probs = recognizer.get_probabilities(audio)

        colors = ['green' if d == digit else 'gray' for d in recognizer.digits]
        axes[idx, 2].bar(recognizer.digits, probs, color=colors, alpha=0.7)
        axes[idx, 2].set_title(f'Predicted: "{recognized}"')
        axes[idx, 2].set_xlabel('Digit')
        axes[idx, 2].set_ylabel('Probability')
        axes[idx, 2].set_ylim([0, 1])
        axes[idx, 2].tick_params(axis='x', rotation=45)
        axes[idx, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('speech_recognition_demo.png', dpi=300, bbox_inches='tight')
    print("Saved: speech_recognition_demo.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Simple Speech-to-Text System (Digit Recognition)")
    print("=" * 60)

    # Initialize recognizer
    recognizer = SimpleSpeechRecognizer(sample_rate=8000)

    # Generate dataset
    print("\n1. Generating synthetic speech data for digits...")
    all_audios = []
    all_labels = []

    for digit in recognizer.digits:
        audios, labels = recognizer.generate_digit_audio(digit, duration=0.8, num_samples=40)
        all_audios.extend(audios)
        all_labels.extend(labels)

    print(f"   Generated {len(all_audios)} speech samples")
    print(f"   Vocabulary: {recognizer.digits}")

    # Extract features
    print("\n2. Extracting speech features...")
    X = np.array([recognizer.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training speech recognition model...")
    recognizer.train(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = recognizer.model.predict(recognizer.scaler.transform(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Overall Accuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualizations
    print("\n5. Creating visualizations...")
    visualize_digit_spectrograms(recognizer, all_audios, all_labels, recognizer.digits)
    visualize_results(y_test, y_pred, recognizer.digits)
    demonstrate_recognition(recognizer)

    # Demo recognition
    print("\n6. Demo: Recognizing spoken digits...")
    for test_digit in ['five', 'eight', 'two']:
        audio, _ = recognizer.generate_digit_audio(test_digit, num_samples=1)
        recognized = recognizer.recognize(audio[0])
        probs = recognizer.get_probabilities(audio[0])

        print(f"\n   True digit: '{test_digit}'")
        print(f"   Recognized: '{recognized}'")
        print(f"   Confidence: {probs[recognizer.digits.index(recognized)]:.3f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
