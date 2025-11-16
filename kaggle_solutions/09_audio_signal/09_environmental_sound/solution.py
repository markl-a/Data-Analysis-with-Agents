"""
Environmental Sound Classification
===================================

This solution demonstrates classification of environmental sounds.
We generate synthetic environmental sounds and classify them using
machine learning.

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


class EnvironmentalSoundClassifier:
    """Environmental sound classification"""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.sounds = ['rain', 'wind', 'birds', 'traffic', 'ocean', 'thunder']
        self.scaler = StandardScaler()
        self.model = None

    def generate_sound(self, sound_type, duration=3.0, num_samples=1):
        """Generate synthetic environmental sounds"""
        audios = []
        labels = []

        for _ in range(num_samples):
            t = np.linspace(0, duration, int(self.sample_rate * duration))

            if sound_type == 'rain':
                # Rain: continuous random impacts
                audio = np.random.normal(0, 0.5, len(t))
                # Filter to rain-like frequency
                sos = signal.butter(4, [200, 3000], btype='band',
                                   fs=self.sample_rate, output='sos')
                audio = signal.sosfilt(sos, audio)
                # Add amplitude modulation
                modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
                audio *= modulation

            elif sound_type == 'wind':
                # Wind: low-frequency turbulent noise
                audio = np.random.normal(0, 1, len(t))
                # Low-pass filter
                sos = signal.butter(4, 800, btype='low', fs=self.sample_rate, output='sos')
                audio = signal.sosfilt(sos, audio)
                # Add gusts
                gusts = 1 + 0.7 * np.sin(2 * np.pi * 0.2 * t) + \
                       0.3 * np.sin(2 * np.pi * 0.5 * t)
                audio *= gusts

            elif sound_type == 'birds':
                # Bird chirps: high-frequency discrete calls
                audio = np.zeros_like(t)
                n_chirps = np.random.randint(10, 20)

                for _ in range(n_chirps):
                    chirp_time = np.random.uniform(0, duration)
                    chirp_start = int(chirp_time * self.sample_rate)
                    chirp_duration = int(np.random.uniform(0.1, 0.3) * self.sample_rate)

                    if chirp_start + chirp_duration < len(t):
                        chirp_t = np.linspace(0, chirp_duration/self.sample_rate, chirp_duration)

                        # Frequency sweep (bird call)
                        f_start = np.random.uniform(2000, 4000)
                        f_end = np.random.uniform(2500, 5000)
                        freq_sweep = f_start + (f_end - f_start) * chirp_t

                        chirp = np.sin(2 * np.pi * freq_sweep * chirp_t)
                        # Envelope
                        chirp *= np.exp(-10 * chirp_t)

                        audio[chirp_start:chirp_start+chirp_duration] += chirp

            elif sound_type == 'traffic':
                # Traffic: rumble with occasional peaks
                # Low-frequency rumble
                rumble = np.random.normal(0, 0.5, len(t))
                sos = signal.butter(4, [80, 500], btype='band',
                                   fs=self.sample_rate, output='sos')
                rumble = signal.sosfilt(sos, rumble)

                # Car passes
                n_cars = np.random.randint(3, 8)
                for _ in range(n_cars):
                    car_time = np.random.uniform(0, duration)
                    car_start = int(car_time * self.sample_rate)
                    car_duration = int(np.random.uniform(0.5, 1.5) * self.sample_rate)

                    if car_start + car_duration < len(t):
                        car_t = np.linspace(0, car_duration/self.sample_rate, car_duration)
                        # Doppler-like effect
                        doppler = 1 + 0.3 * np.exp(-3 * (car_t - car_duration/(2*self.sample_rate))**2 /
                                                   (car_duration/self.sample_rate)**2)
                        rumble[car_start:car_start+car_duration] *= doppler

                audio = rumble

            elif sound_type == 'ocean':
                # Ocean waves: periodic low-frequency swells
                audio = np.random.normal(0, 0.3, len(t))

                # Band-pass filter for wave sound
                sos = signal.butter(4, [100, 1500], btype='band',
                                   fs=self.sample_rate, output='sos')
                audio = signal.sosfilt(sos, audio)

                # Wave periodicity
                wave_freq = np.random.uniform(0.1, 0.3)
                waves = 1 + 0.8 * np.sin(2 * np.pi * wave_freq * t)
                audio *= waves

            elif sound_type == 'thunder':
                # Thunder: low-frequency rumble with attack
                audio = np.zeros_like(t)

                # Number of thunder claps
                n_claps = np.random.randint(1, 3)

                for _ in range(n_claps):
                    clap_time = np.random.uniform(0, duration - 1)
                    clap_start = int(clap_time * self.sample_rate)
                    clap_duration = int(np.random.uniform(0.5, 2.0) * self.sample_rate)

                    if clap_start + clap_duration < len(t):
                        clap_t = np.linspace(0, clap_duration/self.sample_rate, clap_duration)

                        # Low-frequency noise
                        clap = np.random.normal(0, 1, clap_duration)

                        # Low-pass filter
                        sos = signal.butter(4, 300, btype='low',
                                          fs=self.sample_rate, output='sos')
                        clap = signal.sosfilt(sos, clap)

                        # Envelope: sharp attack, slow decay
                        envelope = np.exp(-2 * clap_t)
                        clap *= envelope

                        audio[clap_start:clap_start+clap_duration] += clap

            # Add slight background noise
            noise = np.random.normal(0, 0.01, len(audio))
            audio += noise

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            audios.append(audio)
            labels.append(sound_type)

        return audios, labels

    def extract_features(self, audio):
        """Extract environmental sound features"""
        features = []

        # FFT analysis
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]

        # 1. Spectral features
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

        # Spectral flatness (noise vs tonal)
        geometric_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
        arithmetic_mean = np.mean(fft_mag)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(flatness)

        # Spectral contrast (different bands)
        n_bands = 6
        for i in range(n_bands):
            band_start = i * len(fft_mag) // n_bands
            band_end = (i + 1) * len(fft_mag) // n_bands
            if band_end > band_start:
                band_mag = fft_mag[band_start:band_end]
                contrast = np.max(band_mag) - np.min(band_mag)
            else:
                contrast = 0
            features.append(contrast)

        # 2. Temporal features
        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)

        # Temporal centroid
        envelope = np.abs(signal.hilbert(audio))
        time_indices = np.arange(len(envelope))
        temporal_centroid = np.sum(time_indices * envelope) / (np.sum(envelope) + 1e-10)
        features.append(temporal_centroid / len(envelope))

        # 3. Energy distribution
        # Sub-bass (20-60 Hz)
        sub_bass = np.sum(fft_mag[(freqs >= 20) & (freqs < 60)])
        features.append(sub_bass)

        # Bass (60-250 Hz)
        bass = np.sum(fft_mag[(freqs >= 60) & (freqs < 250)])
        features.append(bass)

        # Low-mid (250-800 Hz)
        low_mid = np.sum(fft_mag[(freqs >= 250) & (freqs < 800)])
        features.append(low_mid)

        # Mid (800-2000 Hz)
        mid = np.sum(fft_mag[(freqs >= 800) & (freqs < 2000)])
        features.append(mid)

        # High-mid (2000-5000 Hz)
        high_mid = np.sum(fft_mag[(freqs >= 2000) & (freqs < 5000)])
        features.append(high_mid)

        # High (5000+ Hz)
        high = np.sum(fft_mag[freqs >= 5000])
        features.append(high)

        # 4. MFCC features (13 coefficients)
        n_mfcc = 13
        for i in range(n_mfcc):
            band_start = i * len(fft_mag) // n_mfcc
            band_end = (i + 1) * len(fft_mag) // n_mfcc
            if band_end > band_start:
                mfcc = np.sum(np.log(fft_mag[band_start:band_end] + 1e-10))
            else:
                mfcc = 0
            features.append(mfcc)

        # 5. Statistical features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.min(audio),
            np.max(audio),
            np.percentile(audio, 25),
            np.percentile(audio, 75),
            np.sqrt(np.mean(audio**2))  # RMS
        ])

        return np.array(features)

    def train(self, X, y):
        """Train the environmental sound classifier"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)


def visualize_environmental_sounds(classifier, audios, labels, sounds):
    """Visualize environmental sound characteristics"""
    fig, axes = plt.subplots(len(sounds), 2, figsize=(15, 14))

    for idx, sound in enumerate(sounds):
        sound_idx = labels.index(sound)
        audio = audios[sound_idx]

        t = np.linspace(0, len(audio)/classifier.sample_rate, len(audio))

        # Waveform
        axes[idx, 0].plot(t, audio, linewidth=0.5, alpha=0.7)
        axes[idx, 0].set_title(f'{sound.capitalize()} - Waveform', fontweight='bold')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(audio, classifier.sample_rate, nperseg=512)
        axes[idx, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                               shading='gouraud', cmap='viridis')
        axes[idx, 1].set_title(f'{sound.capitalize()} - Spectrogram', fontweight='bold')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Frequency (Hz)')
        axes[idx, 1].set_ylim([0, 8000])

    plt.tight_layout()
    plt.savefig('environmental_sounds.png', dpi=300, bbox_inches='tight')
    print("Saved: environmental_sounds.png")
    plt.close()


def visualize_results(y_true, y_pred, sounds):
    """Visualize classification results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=sounds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn',
                xticklabels=sounds, yticklabels=sounds, ax=axes[0])
    axes[0].set_title('Environmental Sound Classification', fontweight='bold')
    axes[0].set_xlabel('Predicted Sound')
    axes[0].set_ylabel('True Sound')

    # F1-scores
    f1_scores = []
    for i in range(len(sounds)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(sounds)))
    bars = axes[1].bar(sounds, f1_scores, color=colors, alpha=0.7)
    axes[1].set_title('Per-Sound F1-Score', fontweight='bold')
    axes[1].set_xlabel('Sound Type')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('environmental_results.png', dpi=300, bbox_inches='tight')
    print("Saved: environmental_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Environmental Sound Classification")
    print("=" * 60)

    # Initialize classifier
    classifier = EnvironmentalSoundClassifier(sample_rate=22050)

    # Generate dataset
    print("\n1. Generating synthetic environmental sounds...")
    all_audios = []
    all_labels = []

    for sound in classifier.sounds:
        audios, labels = classifier.generate_sound(sound, duration=3.0, num_samples=35)
        all_audios.extend(audios)
        all_labels.extend(labels)

    print(f"   Generated {len(all_audios)} sound samples")
    print(f"   Sound types: {classifier.sounds}")

    # Extract features
    print("\n2. Extracting environmental sound features...")
    X = np.array([classifier.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training environmental sound classifier...")
    classifier.train(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = classifier.model.predict(classifier.scaler.transform(X_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualizations
    print("\n5. Creating visualizations...")
    visualize_environmental_sounds(classifier, all_audios, all_labels, classifier.sounds)
    visualize_results(y_test, y_pred, classifier.sounds)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
