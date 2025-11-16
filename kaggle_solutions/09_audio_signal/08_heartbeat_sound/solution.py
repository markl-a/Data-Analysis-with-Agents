"""
Heartbeat Sound Classification
================================

This solution demonstrates classification of heartbeat sounds for
medical diagnosis. We generate synthetic heart sounds (normal and abnormal)
and classify them using machine learning.

Author: Kaggle Solutions Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import seaborn as sns
from scipy import signal
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')


class HeartbeatClassifier:
    """Heartbeat sound classification for medical diagnosis"""

    def __init__(self, sample_rate=4000):
        self.sample_rate = sample_rate
        self.conditions = ['normal', 'murmur', 'tachycardia', 'arrhythmia', 'bradycardia']
        self.scaler = StandardScaler()
        self.model = None

    def generate_heartbeat(self, condition, duration=5.0, num_samples=1):
        """Generate synthetic heartbeat sounds"""
        audios = []
        labels = []

        for _ in range(num_samples):
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.zeros_like(t)

            if condition == 'normal':
                # Normal heartbeat: lub-dub pattern at 60-100 BPM
                heart_rate = np.random.uniform(60, 100) / 60  # beats per second
                beat_interval = 1 / heart_rate

                # S1 (lub) and S2 (dub) sounds
                for beat_time in np.arange(0, duration, beat_interval):
                    # S1 sound (lub) - lower frequency, longer
                    s1_start = int(beat_time * self.sample_rate)
                    s1_duration = int(0.12 * self.sample_rate)  # 120ms

                    if s1_start + s1_duration < len(t):
                        s1_t = np.linspace(0, 0.12, s1_duration)
                        s1 = np.sin(2 * np.pi * 40 * s1_t) * np.exp(-20 * s1_t)
                        audio[s1_start:s1_start+s1_duration] += s1

                    # S2 sound (dub) - higher frequency, shorter
                    s2_start = int((beat_time + 0.3 * beat_interval) * self.sample_rate)
                    s2_duration = int(0.08 * self.sample_rate)  # 80ms

                    if s2_start + s2_duration < len(t):
                        s2_t = np.linspace(0, 0.08, s2_duration)
                        s2 = np.sin(2 * np.pi * 60 * s2_t) * np.exp(-25 * s2_t)
                        audio[s2_start:s2_start+s2_duration] += s2

            elif condition == 'murmur':
                # Heart murmur: additional whooshing sound between S1 and S2
                heart_rate = np.random.uniform(65, 95) / 60
                beat_interval = 1 / heart_rate

                for beat_time in np.arange(0, duration, beat_interval):
                    # S1 sound
                    s1_start = int(beat_time * self.sample_rate)
                    s1_duration = int(0.12 * self.sample_rate)

                    if s1_start + s1_duration < len(t):
                        s1_t = np.linspace(0, 0.12, s1_duration)
                        s1 = np.sin(2 * np.pi * 40 * s1_t) * np.exp(-20 * s1_t)
                        audio[s1_start:s1_start+s1_duration] += s1

                    # Murmur (whoosh between S1 and S2)
                    murmur_start = int((beat_time + 0.12) * self.sample_rate)
                    murmur_duration = int(0.15 * self.sample_rate)

                    if murmur_start + murmur_duration < len(t):
                        murmur = np.random.normal(0, 0.3, murmur_duration)
                        # Low-pass filter for whooshing sound
                        sos = signal.butter(4, 200, btype='low', fs=self.sample_rate, output='sos')
                        murmur = signal.sosfilt(sos, murmur)
                        audio[murmur_start:murmur_start+murmur_duration] += murmur

                    # S2 sound
                    s2_start = int((beat_time + 0.3 * beat_interval) * self.sample_rate)
                    s2_duration = int(0.08 * self.sample_rate)

                    if s2_start + s2_duration < len(t):
                        s2_t = np.linspace(0, 0.08, s2_duration)
                        s2 = np.sin(2 * np.pi * 60 * s2_t) * np.exp(-25 * s2_t)
                        audio[s2_start:s2_start+s2_duration] += s2

            elif condition == 'tachycardia':
                # Fast heartbeat: >100 BPM
                heart_rate = np.random.uniform(110, 150) / 60
                beat_interval = 1 / heart_rate

                for beat_time in np.arange(0, duration, beat_interval):
                    s1_start = int(beat_time * self.sample_rate)
                    s1_duration = int(0.10 * self.sample_rate)  # Shorter due to faster rate

                    if s1_start + s1_duration < len(t):
                        s1_t = np.linspace(0, 0.10, s1_duration)
                        s1 = np.sin(2 * np.pi * 40 * s1_t) * np.exp(-25 * s1_t)
                        audio[s1_start:s1_start+s1_duration] += s1

                    s2_start = int((beat_time + 0.25 * beat_interval) * self.sample_rate)
                    s2_duration = int(0.06 * self.sample_rate)

                    if s2_start + s2_duration < len(t):
                        s2_t = np.linspace(0, 0.06, s2_duration)
                        s2 = np.sin(2 * np.pi * 60 * s2_t) * np.exp(-30 * s2_t)
                        audio[s2_start:s2_start+s2_duration] += s2

            elif condition == 'arrhythmia':
                # Irregular heartbeat: variable intervals
                time_elapsed = 0

                while time_elapsed < duration:
                    # Variable beat interval
                    beat_interval = np.random.uniform(0.5, 1.5)

                    s1_start = int(time_elapsed * self.sample_rate)
                    s1_duration = int(0.12 * self.sample_rate)

                    if s1_start + s1_duration < len(t):
                        s1_t = np.linspace(0, 0.12, s1_duration)
                        s1 = np.sin(2 * np.pi * 40 * s1_t) * np.exp(-20 * s1_t)
                        audio[s1_start:s1_start+s1_duration] += s1

                    s2_start = int((time_elapsed + 0.3 * beat_interval) * self.sample_rate)
                    s2_duration = int(0.08 * self.sample_rate)

                    if s2_start + s2_duration < len(t):
                        s2_t = np.linspace(0, 0.08, s2_duration)
                        s2 = np.sin(2 * np.pi * 60 * s2_t) * np.exp(-25 * s2_t)
                        audio[s2_start:s2_start+s2_duration] += s2

                    time_elapsed += beat_interval

            elif condition == 'bradycardia':
                # Slow heartbeat: <60 BPM
                heart_rate = np.random.uniform(40, 55) / 60
                beat_interval = 1 / heart_rate

                for beat_time in np.arange(0, duration, beat_interval):
                    s1_start = int(beat_time * self.sample_rate)
                    s1_duration = int(0.14 * self.sample_rate)  # Slightly longer

                    if s1_start + s1_duration < len(t):
                        s1_t = np.linspace(0, 0.14, s1_duration)
                        s1 = np.sin(2 * np.pi * 40 * s1_t) * np.exp(-18 * s1_t)
                        audio[s1_start:s1_start+s1_duration] += s1

                    s2_start = int((beat_time + 0.3 * beat_interval) * self.sample_rate)
                    s2_duration = int(0.10 * self.sample_rate)

                    if s2_start + s2_duration < len(t):
                        s2_t = np.linspace(0, 0.10, s2_duration)
                        s2 = np.sin(2 * np.pi * 60 * s2_t) * np.exp(-22 * s2_t)
                        audio[s2_start:s2_start+s2_duration] += s2

            # Add slight noise
            noise = np.random.normal(0, 0.02, len(audio))
            audio += noise

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            audios.append(audio)
            labels.append(condition)

        return audios, labels

    def extract_features(self, audio):
        """Extract heartbeat diagnostic features"""
        features = []

        # FFT analysis
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(audio)//2]

        # 1. Heart rate features
        # Detect peaks (heartbeats)
        envelope = np.abs(signal.hilbert(audio))
        peaks, properties = signal.find_peaks(envelope, height=0.2*np.max(envelope),
                                             distance=int(0.3*self.sample_rate))

        # Heart rate (beats per minute)
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sample_rate
            avg_interval = np.mean(intervals)
            heart_rate = 60 / avg_interval
            hr_variability = np.std(intervals)
        else:
            heart_rate = 60
            hr_variability = 0

        features.append(heart_rate)
        features.append(hr_variability)

        # Number of beats
        features.append(len(peaks))

        # 2. Frequency features
        # Low frequency power (20-60 Hz) - S1/S2 sounds
        low_freq_power = np.sum(fft_mag[(freqs >= 20) & (freqs < 60)])
        features.append(low_freq_power)

        # Mid frequency power (60-150 Hz) - murmurs
        mid_freq_power = np.sum(fft_mag[(freqs >= 60) & (freqs < 150)])
        features.append(mid_freq_power)

        # High frequency power (150+ Hz) - noise
        high_freq_power = np.sum(fft_mag[freqs >= 150])
        features.append(high_freq_power)

        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
        features.append(spectral_centroid)

        # 3. Temporal features
        # S1-S2 interval (systolic period)
        if len(peaks) >= 2:
            # Find pairs of peaks (S1 and S2)
            peak_pairs = []
            i = 0
            while i < len(peaks) - 1:
                interval = (peaks[i+1] - peaks[i]) / self.sample_rate
                if interval < 0.4:  # S1-S2 interval typically < 400ms
                    peak_pairs.append(interval)
                    i += 2
                else:
                    i += 1

            if peak_pairs:
                s1_s2_interval = np.mean(peak_pairs)
                s1_s2_var = np.std(peak_pairs)
            else:
                s1_s2_interval = 0.2
                s1_s2_var = 0
        else:
            s1_s2_interval = 0.2
            s1_s2_var = 0

        features.append(s1_s2_interval)
        features.append(s1_s2_var)

        # 4. Energy features
        features.extend([
            np.sum(audio**2) / len(audio),  # Total energy
            np.sqrt(np.mean(audio**2)),     # RMS
            np.max(np.abs(audio))            # Peak amplitude
        ])

        # 5. MFCC-like features (8 coefficients)
        n_mfcc = 8
        for i in range(n_mfcc):
            band_start = i * len(fft_mag) // n_mfcc
            band_end = (i + 1) * len(fft_mag) // n_mfcc
            if band_end > band_start:
                mfcc = np.sum(np.log(fft_mag[band_start:band_end] + 1e-10))
            else:
                mfcc = 0
            features.append(mfcc)

        # 6. Statistical features
        features.extend([
            np.mean(audio),
            np.std(audio),
            np.percentile(audio, 25),
            np.percentile(audio, 75)
        ])

        return np.array(features)

    def train(self, X, y):
        """Train the heartbeat classification model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(X_scaled, y)


def visualize_heartbeat_patterns(classifier, audios, labels, conditions):
    """Visualize heartbeat patterns for different conditions"""
    fig, axes = plt.subplots(len(conditions), 2, figsize=(15, 12))

    for idx, condition in enumerate(conditions):
        cond_idx = labels.index(condition)
        audio = audios[cond_idx]

        t = np.linspace(0, len(audio)/classifier.sample_rate, len(audio))

        # Waveform with envelope
        axes[idx, 0].plot(t, audio, linewidth=0.5, alpha=0.6, color='blue')
        envelope = np.abs(signal.hilbert(audio))
        axes[idx, 0].plot(t, envelope, linewidth=2, color='red', label='Envelope')
        axes[idx, 0].set_title(f'{condition.capitalize()} - Heartbeat Pattern', fontweight='bold')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend()
        axes[idx, 0].set_xlim([0, 3])  # Show first 3 seconds

        # Frequency spectrum
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/classifier.sample_rate)[:len(audio)//2]

        axes[idx, 1].plot(freqs, fft_mag, linewidth=0.7, color='darkgreen')
        axes[idx, 1].set_title(f'{condition.capitalize()} - Frequency Spectrum', fontweight='bold')
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_xlim([0, 200])
        axes[idx, 1].grid(True, alpha=0.3)

        # Mark S1 and S2 frequency ranges
        axes[idx, 1].axvspan(30, 50, alpha=0.2, color='red', label='S1')
        axes[idx, 1].axvspan(50, 70, alpha=0.2, color='blue', label='S2')
        axes[idx, 1].legend()

    plt.tight_layout()
    plt.savefig('heartbeat_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: heartbeat_patterns.png")
    plt.close()


def visualize_results(y_true, y_pred, conditions, y_proba):
    """Visualize classification results with ROC curves"""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, :])
    cm = confusion_matrix(y_true, y_pred, labels=conditions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=conditions, yticklabels=conditions, ax=ax1)
    ax1.set_title('Heartbeat Condition Classification', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Predicted Condition')
    ax1.set_ylabel('True Condition')

    # Per-condition metrics
    ax2 = fig.add_subplot(gs[1, 0])
    precisions = []
    recalls = []

    for i in range(len(conditions)):
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

    x = np.arange(len(conditions))
    width = 0.35

    ax2.barh(x - width/2, precisions, width, label='Precision', color='steelblue', alpha=0.7)
    ax2.barh(x + width/2, recalls, width, label='Recall', color='coral', alpha=0.7)
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Condition')
    ax2.set_title('Per-Condition Metrics', fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(conditions)
    ax2.set_xlim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    # ROC curves (one-vs-rest)
    ax3 = fig.add_subplot(gs[1, 1])
    y_true_bin = label_binarize(y_true, classes=conditions)

    for i, condition in enumerate(conditions):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, linewidth=2, label=f'{condition} (AUC = {roc_auc:.2f})')

    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.savefig('heartbeat_results.png', dpi=300, bbox_inches='tight')
    print("Saved: heartbeat_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Heartbeat Sound Classification")
    print("=" * 60)

    # Initialize classifier
    classifier = HeartbeatClassifier(sample_rate=4000)

    # Generate dataset
    print("\n1. Generating synthetic heartbeat sounds...")
    all_audios = []
    all_labels = []

    for condition in classifier.conditions:
        audios, labels = classifier.generate_heartbeat(condition, duration=5.0, num_samples=40)
        all_audios.extend(audios)
        all_labels.extend(labels)

    print(f"   Generated {len(all_audios)} heartbeat samples")
    print(f"   Conditions: {classifier.conditions}")

    # Extract features
    print("\n2. Extracting heartbeat features...")
    X = np.array([classifier.extract_features(audio) for audio in all_audios])
    y = np.array(all_labels)
    print(f"   Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train model
    print("\n3. Training heartbeat classification model...")
    classifier.train(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = classifier.model.predict(classifier.scaler.transform(X_test))
    y_proba = classifier.model.predict_proba(classifier.scaler.transform(X_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualizations
    print("\n5. Creating visualizations...")
    visualize_heartbeat_patterns(classifier, all_audios, all_labels, classifier.conditions)
    visualize_results(y_test, y_pred, classifier.conditions, y_proba)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
