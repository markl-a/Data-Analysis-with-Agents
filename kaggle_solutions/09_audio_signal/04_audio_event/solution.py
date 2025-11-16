"""
Audio Event Detection
=====================

This solution demonstrates detection and classification of audio events in
continuous audio streams. We generate synthetic audio with various sound
events and detect their occurrence and type.

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


class AudioEventDetector:
    """Audio event detection and classification"""

    def __init__(self, sample_rate=16000, window_size=0.5):
        self.sample_rate = sample_rate
        self.window_size = window_size  # seconds
        self.window_samples = int(sample_rate * window_size)
        self.events = ['silence', 'bell', 'alarm', 'door_slam', 'glass_break', 'applause']
        self.scaler = StandardScaler()
        self.model = None

    def generate_event_sound(self, event_type):
        """Generate synthetic sound for specific event"""
        duration = self.window_size
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        if event_type == 'silence':
            # Very low amplitude noise
            audio = np.random.normal(0, 0.01, len(t))

        elif event_type == 'bell':
            # Bell-like sound with decay
            freq = 880  # A5
            audio = np.sin(2 * np.pi * freq * t)
            # Add harmonics
            audio += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
            audio += 0.25 * np.sin(2 * np.pi * freq * 3 * t)
            # Exponential decay
            audio *= np.exp(-3 * t)
            # Add slight vibrato
            vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
            audio *= vibrato

        elif event_type == 'alarm':
            # Alarm sound with rapid oscillation
            freq1, freq2 = 1000, 1200
            oscillation = 4  # Hz
            freq = freq1 + (freq2 - freq1) * (np.sin(2 * np.pi * oscillation * t) + 1) / 2
            audio = signal.square(2 * np.pi * freq * t)
            # Maintain constant amplitude
            audio *= 0.8

        elif event_type == 'door_slam':
            # Impact sound with immediate decay
            # White noise burst
            audio = np.random.normal(0, 1, len(t))
            # Very fast exponential decay
            audio *= np.exp(-20 * t)
            # Add low frequency thump
            thump = 2 * np.sin(2 * np.pi * 60 * t) * np.exp(-10 * t)
            audio += thump

        elif event_type == 'glass_break':
            # High-frequency shattering sound
            # Burst of high-frequency noise
            audio = np.random.normal(0, 1, len(t))
            # High-pass characteristic
            sos = signal.butter(4, 2000, btype='high', fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos, audio)
            # Multiple small bursts
            burst_envelope = np.zeros_like(t)
            for i in range(5):
                start = i * len(t) // 10
                end = start + len(t) // 20
                if end < len(t):
                    burst_envelope[start:end] = np.exp(-10 * t[:end-start])
            audio *= burst_envelope

        elif event_type == 'applause':
            # Multiple random claps
            audio = np.zeros_like(t)
            n_claps = np.random.randint(20, 40)
            for _ in range(n_claps):
                clap_time = np.random.uniform(0, duration)
                clap_idx = int(clap_time * self.sample_rate)
                clap_duration = int(0.05 * self.sample_rate)
                if clap_idx + clap_duration < len(t):
                    clap = np.random.normal(0, 1, clap_duration)
                    clap *= np.exp(-20 * np.linspace(0, 0.05, clap_duration))
                    audio[clap_idx:clap_idx+clap_duration] += clap

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio

    def extract_features(self, audio):
        """Extract features for event detection"""
        features = []

        # FFT analysis
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]

        # 1. Energy features
        total_energy = np.sum(audio**2) / len(audio)
        features.append(total_energy)

        rms = np.sqrt(np.mean(audio**2))
        features.append(rms)

        peak_amplitude = np.max(np.abs(audio))
        features.append(peak_amplitude)

        # 2. Temporal features
        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)

        # Temporal centroid (where energy is concentrated in time)
        envelope = np.abs(signal.hilbert(audio))
        time_indices = np.arange(len(envelope))
        temporal_centroid = np.sum(time_indices * envelope) / (np.sum(envelope) + 1e-10)
        features.append(temporal_centroid / len(envelope))

        # Decay rate (how fast energy decreases)
        if len(envelope) > 10:
            first_half_energy = np.sum(envelope[:len(envelope)//2]**2)
            second_half_energy = np.sum(envelope[len(envelope)//2:]**2)
            decay_rate = (first_half_energy - second_half_energy) / (first_half_energy + 1e-10)
        else:
            decay_rate = 0
        features.append(decay_rate)

        # 3. Spectral features
        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        features.append(spectral_centroid)

        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_mag) / (np.sum(fft_mag) + 1e-10))
        features.append(spectral_bandwidth)

        # Spectral rolloff
        cumsum = np.cumsum(fft_mag)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        else:
            rolloff = 0
        features.append(rolloff)

        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
        arithmetic_mean = np.mean(fft_mag)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(flatness)

        # 4. Frequency band energies
        # Low (0-500 Hz)
        low_energy = np.sum(fft_mag[freqs < 500])
        features.append(low_energy)

        # Mid (500-2000 Hz)
        mid_energy = np.sum(fft_mag[(freqs >= 500) & (freqs < 2000)])
        features.append(mid_energy)

        # High (2000+ Hz)
        high_energy = np.sum(fft_mag[freqs >= 2000])
        features.append(high_energy)

        # Band energy ratios
        total_freq_energy = low_energy + mid_energy + high_energy + 1e-10
        features.extend([
            low_energy / total_freq_energy,
            mid_energy / total_freq_energy,
            high_energy / total_freq_energy
        ])

        # 5. Statistical features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.min(audio),
            np.max(audio),
            np.percentile(audio, 25),
            np.percentile(audio, 75)
        ])

        return np.array(features)

    def create_audio_stream(self, duration=10.0, events_per_second=1.0):
        """Create continuous audio stream with random events"""
        total_samples = int(self.sample_rate * duration)
        audio_stream = np.random.normal(0, 0.01, total_samples)  # Background noise

        n_events = int(duration * events_per_second)
        event_times = sorted(np.random.uniform(0, duration, n_events))
        event_labels = []

        for event_time in event_times:
            # Random event type (excluding silence)
            event_type = np.random.choice(self.events[1:])
            event_audio = self.generate_event_sound(event_type)

            # Insert into stream
            start_idx = int(event_time * self.sample_rate)
            end_idx = min(start_idx + len(event_audio), total_samples)
            audio_stream[start_idx:end_idx] += event_audio[:end_idx-start_idx]

            event_labels.append((event_time, event_type))

        return audio_stream, event_labels

    def train(self, X, y):
        """Train the event detection model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)


def visualize_events(detector):
    """Visualize different event types"""
    fig, axes = plt.subplots(len(detector.events), 2, figsize=(15, 12))

    for idx, event in enumerate(detector.events):
        audio = detector.generate_event_sound(event)
        t = np.linspace(0, detector.window_size, len(audio))

        # Waveform
        axes[idx, 0].plot(t, audio, linewidth=0.7, color='darkgreen')
        axes[idx, 0].set_title(f'{event.replace("_", " ").title()} - Waveform')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(audio, detector.sample_rate, nperseg=128)
        axes[idx, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                               shading='gouraud', cmap='hot')
        axes[idx, 1].set_title(f'{event.replace("_", " ").title()} - Spectrogram')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Frequency (Hz)')
        axes[idx, 1].set_ylim([0, 4000])

    plt.tight_layout()
    plt.savefig('audio_event_types.png', dpi=300, bbox_inches='tight')
    print("Saved: audio_event_types.png")
    plt.close()


def visualize_results(y_true, y_pred, events):
    """Visualize detection results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=events)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=events, yticklabels=events, ax=axes[0])
    axes[0].set_title('Event Detection Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Event')
    axes[0].set_ylabel('True Event')

    # Per-event metrics
    precisions = []
    recalls = []
    for i in range(len(events)):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum()
        else:
            recall = 0

        if cm[:, i].sum() > 0:
            precision = cm[i, i] / cm[:, i].sum()
        else:
            precision = 0

        recalls.append(recall)
        precisions.append(precision)

    x = np.arange(len(events))
    width = 0.35

    axes[1].bar(x - width/2, precisions, width, label='Precision', color='steelblue', alpha=0.7)
    axes[1].bar(x + width/2, recalls, width, label='Recall', color='coral', alpha=0.7)
    axes[1].set_xlabel('Event Type')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Per-Event Precision and Recall', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(events, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('audio_event_results.png', dpi=300, bbox_inches='tight')
    print("Saved: audio_event_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Audio Event Detection")
    print("=" * 60)

    # Initialize detector
    detector = AudioEventDetector(sample_rate=16000, window_size=0.5)

    # Generate training dataset
    print("\n1. Generating audio event dataset...")
    all_audios = []
    all_labels = []

    for event in detector.events:
        for _ in range(50):
            audio = detector.generate_event_sound(event)
            all_audios.append(audio)
            all_labels.append(event)

    print(f"   Generated {len(all_audios)} event samples")
    print(f"   Events: {detector.events}")

    # Extract features
    print("\n2. Extracting event features...")
    X = np.array([detector.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training event detection model...")
    detector.train(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = detector.model.predict(detector.scaler.transform(X_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualizations
    print("\n5. Creating visualizations...")
    visualize_events(detector)
    visualize_results(y_test, y_pred, detector.events)

    # Demo: Event detection in stream
    print("\n6. Demo: Detecting events in audio stream...")
    stream, true_events = detector.create_audio_stream(duration=5.0, events_per_second=2)

    print(f"\n   Generated {len(stream)/detector.sample_rate:.1f}s audio stream")
    print(f"   True events: {len(true_events)}")
    for event_time, event_type in true_events[:5]:
        print(f"   - {event_time:.2f}s: {event_type}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
