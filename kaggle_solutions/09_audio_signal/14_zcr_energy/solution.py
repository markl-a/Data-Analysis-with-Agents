"""
Zero-Crossing Rate and Energy Features
=======================================

This solution demonstrates comprehensive analysis of zero-crossing rate (ZCR)
and energy features for audio signal processing and classification.

Dataset: Synthetic audio signals with varying characteristics
Techniques:
- Zero-crossing rate computation
- Short-time energy calculation
- Energy entropy
- Spectral energy distribution
- Voice/unvoiced detection
- Signal classification based on ZCR and energy

Author: Data Science Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import rfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ZCRandEnergyAnalyzer:
    """
    Comprehensive ZCR and energy feature extraction.
    """

    def __init__(self, sample_rate=22050, frame_size=2048, hop_length=512):
        """
        Initialize analyzer.

        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate
        frame_size : int
            Frame size for analysis
        hop_length : int
            Number of samples between frames
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length

    def zero_crossing_rate(self, audio):
        """
        Compute zero-crossing rate.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        zcr : ndarray
            Zero-crossing rate for each frame
        """
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_length
        zcr = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_size]

            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

            # Count zero crossings
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            # Normalize by frame length
            zcr[i] = zero_crossings / len(frame)

        return zcr

    def short_time_energy(self, audio):
        """
        Compute short-time energy.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        energy : ndarray
            Short-time energy for each frame
        """
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_length
        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_size]

            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

            # Compute energy
            energy[i] = np.sum(frame ** 2) / len(frame)

        return energy

    def root_mean_square_energy(self, audio):
        """
        Compute RMS energy.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        rms : ndarray
            RMS energy for each frame
        """
        energy = self.short_time_energy(audio)
        return np.sqrt(energy)

    def energy_entropy(self, audio, num_short_blocks=10):
        """
        Compute energy entropy.

        Measures the abrupt changes in energy level.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        num_short_blocks : int
            Number of sub-frames for entropy computation

        Returns:
        --------
        entropy : ndarray
            Energy entropy for each frame
        """
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_length
        entropy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_size]

            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

            # Divide frame into sub-frames
            sub_frame_size = len(frame) // num_short_blocks
            sub_energies = []

            for j in range(num_short_blocks):
                sub_start = j * sub_frame_size
                sub_end = sub_start + sub_frame_size
                sub_frame = frame[sub_start:sub_end]

                if len(sub_frame) > 0:
                    sub_energy = np.sum(sub_frame ** 2)
                    sub_energies.append(sub_energy)

            # Normalize energies to probabilities
            sub_energies = np.array(sub_energies)
            total_energy = np.sum(sub_energies)

            if total_energy > 0:
                probs = sub_energies / total_energy
                # Compute entropy
                entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    def spectral_energy(self, audio):
        """
        Compute spectral energy distribution.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        spectral_energy : ndarray
            Spectral energy for each frame
        low_energy : ndarray
            Low-frequency energy
        high_energy : ndarray
            High-frequency energy
        """
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_length
        spectral_energy = np.zeros(n_frames)
        low_energy = np.zeros(n_frames)
        high_energy = np.zeros(n_frames)

        # Frequency bins
        freqs = rfftfreq(self.frame_size, 1/self.sample_rate)
        low_freq_idx = np.where(freqs < 1000)[0]
        high_freq_idx = np.where(freqs >= 1000)[0]

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_size]

            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

            # Compute spectrum
            spectrum = np.abs(rfft(frame))
            spectral_energy[i] = np.sum(spectrum ** 2)

            # Low and high frequency energy
            if len(low_freq_idx) > 0:
                low_energy[i] = np.sum(spectrum[low_freq_idx] ** 2)
            if len(high_freq_idx) > 0:
                high_energy[i] = np.sum(spectrum[high_freq_idx] ** 2)

        return spectral_energy, low_energy, high_energy

    def voice_unvoiced_detection(self, audio, zcr_threshold=0.1, energy_threshold=0.01):
        """
        Detect voiced and unvoiced segments.

        Voiced: Low ZCR, High Energy (vowels)
        Unvoiced: High ZCR, Low Energy (consonants)

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        zcr_threshold : float
            ZCR threshold for voice/unvoiced detection
        energy_threshold : float
            Energy threshold

        Returns:
        --------
        voiced : ndarray
            Boolean array indicating voiced frames
        """
        zcr = self.zero_crossing_rate(audio)
        energy = self.short_time_energy(audio)

        # Normalize
        zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)

        # Voiced: low ZCR and high energy
        voiced = (zcr_norm < zcr_threshold) & (energy_norm > energy_threshold)

        return voiced

    def extract_all_features(self, audio):
        """
        Extract all ZCR and energy features.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        features : dict
            Dictionary of all features
        """
        zcr = self.zero_crossing_rate(audio)
        energy = self.short_time_energy(audio)
        rms = self.root_mean_square_energy(audio)
        entropy = self.energy_entropy(audio)
        spec_energy, low_energy, high_energy = self.spectral_energy(audio)
        voiced = self.voice_unvoiced_detection(audio)

        return {
            'zcr': zcr,
            'energy': energy,
            'rms': rms,
            'entropy': entropy,
            'spectral_energy': spec_energy,
            'low_energy': low_energy,
            'high_energy': high_energy,
            'voiced': voiced
        }


def generate_voiced_segment(duration=0.5, sample_rate=22050):
    """
    Generate voiced speech-like segment (vowel).

    Parameters:
    -----------
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate

    Returns:
    --------
    audio : ndarray
        Synthetic voiced segment
    """
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Fundamental frequency (pitch)
    f0 = 120  # Hz

    # Formants (vowel 'a')
    formants = [730, 1090, 2440]
    audio = np.zeros_like(t)

    # Generate periodic signal
    for formant in formants:
        audio += np.sin(2 * np.pi * formant * t)

    # Add pitch
    audio += 2 * np.sin(2 * np.pi * f0 * t)

    # Apply envelope
    envelope = np.exp(-2 * t) * (1 - np.exp(-20 * t))
    audio *= envelope

    return audio / np.max(np.abs(audio))


def generate_unvoiced_segment(duration=0.2, sample_rate=22050):
    """
    Generate unvoiced speech-like segment (fricative).

    Parameters:
    -----------
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate

    Returns:
    --------
    audio : ndarray
        Synthetic unvoiced segment
    """
    # White noise filtered to speech range
    audio = np.random.randn(int(sample_rate * duration))

    # High-pass filter to simulate fricative
    sos = signal.butter(4, 2000, 'hp', fs=sample_rate, output='sos')
    audio = signal.sosfilt(sos, audio)

    # Apply envelope
    t = np.linspace(0, duration, len(audio))
    envelope = np.exp(-5 * t) * (1 - np.exp(-30 * t))
    audio *= envelope

    return audio / np.max(np.abs(audio))


def generate_complex_signal(sample_rate=22050):
    """
    Generate complex signal with voiced, unvoiced, and silence segments.

    Returns:
    --------
    audio : ndarray
        Complex synthetic signal
    segment_labels : list
        Labels for each segment
    """
    segments = []
    labels = []

    # Alternating pattern
    pattern = ['voiced', 'unvoiced', 'silence', 'voiced', 'unvoiced']

    for label in pattern:
        if label == 'voiced':
            segment = generate_voiced_segment(duration=0.5, sample_rate=sample_rate)
        elif label == 'unvoiced':
            segment = generate_unvoiced_segment(duration=0.3, sample_rate=sample_rate)
        else:  # silence
            segment = np.zeros(int(0.2 * sample_rate))

        segments.append(segment)
        labels.append(label)

    audio = np.concatenate(segments)
    return audio, labels


def visualize_zcr_energy(audio, analyzer, save_path='zcr_energy_analysis.png'):
    """
    Visualize ZCR and energy features.

    Parameters:
    -----------
    audio : ndarray
        Audio signal
    analyzer : ZCRandEnergyAnalyzer
        Analyzer instance
    save_path : str
        Path to save figure
    """
    features = analyzer.extract_all_features(audio)

    fig = plt.figure(figsize=(16, 12))

    times_audio = np.arange(len(audio)) / analyzer.sample_rate
    times_features = np.arange(len(features['zcr'])) * analyzer.hop_length / analyzer.sample_rate

    # Waveform
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(times_audio, audio, linewidth=0.5)
    plt.title('Waveform', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Zero-Crossing Rate
    ax2 = plt.subplot(4, 2, 2)
    plt.plot(times_features, features['zcr'], linewidth=2, color='blue')
    plt.fill_between(times_features, 0, features['zcr'], alpha=0.3, color='blue')
    plt.title('Zero-Crossing Rate', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('ZCR')
    plt.grid(True, alpha=0.3)

    # Short-Time Energy
    ax3 = plt.subplot(4, 2, 3)
    plt.plot(times_features, features['energy'], linewidth=2, color='red')
    plt.fill_between(times_features, 0, features['energy'], alpha=0.3, color='red')
    plt.title('Short-Time Energy', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.grid(True, alpha=0.3)

    # RMS Energy
    ax4 = plt.subplot(4, 2, 4)
    plt.plot(times_features, features['rms'], linewidth=2, color='green')
    plt.fill_between(times_features, 0, features['rms'], alpha=0.3, color='green')
    plt.title('RMS Energy', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS')
    plt.grid(True, alpha=0.3)

    # Energy Entropy
    ax5 = plt.subplot(4, 2, 5)
    plt.plot(times_features, features['entropy'], linewidth=2, color='purple')
    plt.fill_between(times_features, 0, features['entropy'], alpha=0.3, color='purple')
    plt.title('Energy Entropy', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Entropy')
    plt.grid(True, alpha=0.3)

    # Spectral Energy Distribution
    ax6 = plt.subplot(4, 2, 6)
    plt.plot(times_features, features['low_energy'], linewidth=2,
             label='Low Freq (<1kHz)', color='blue')
    plt.plot(times_features, features['high_energy'], linewidth=2,
             label='High Freq (≥1kHz)', color='red')
    plt.title('Spectral Energy Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ZCR vs Energy scatter
    ax7 = plt.subplot(4, 2, 7)
    voiced_idx = features['voiced']
    plt.scatter(features['zcr'][voiced_idx], features['energy'][voiced_idx],
               c='blue', label='Voiced', alpha=0.6, s=30)
    plt.scatter(features['zcr'][~voiced_idx], features['energy'][~voiced_idx],
               c='red', label='Unvoiced', alpha=0.6, s=30)
    plt.title('ZCR vs Energy', fontsize=12, fontweight='bold')
    plt.xlabel('ZCR')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Voice/Unvoiced detection
    ax8 = plt.subplot(4, 2, 8)
    plt.plot(times_audio, audio, linewidth=0.5, alpha=0.5, label='Audio')

    # Highlight voiced segments
    for i in range(len(features['voiced'])):
        if features['voiced'][i]:
            start_time = i * analyzer.hop_length / analyzer.sample_rate
            end_time = (i + 1) * analyzer.hop_length / analyzer.sample_rate
            plt.axvspan(start_time, end_time, alpha=0.3, color='blue')

    plt.title('Voice/Unvoiced Detection (Blue=Voiced)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ZCR/Energy analysis to {save_path}")
    plt.close()


def compare_signal_types():
    """
    Compare ZCR and energy for different signal types.
    """
    sample_rate = 22050
    analyzer = ZCRandEnergyAnalyzer(sample_rate=sample_rate)

    # Generate different signal types
    signals = {
        'Voiced (Vowel)': generate_voiced_segment(duration=1.0, sample_rate=sample_rate),
        'Unvoiced (Fricative)': generate_unvoiced_segment(duration=1.0, sample_rate=sample_rate),
        'White Noise': np.random.randn(sample_rate)
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for idx, (name, audio) in enumerate(signals.items()):
        # Normalize
        audio = audio / np.max(np.abs(audio))

        features = analyzer.extract_all_features(audio)
        times = np.arange(len(features['zcr'])) * analyzer.hop_length / sample_rate

        # Waveform
        time_audio = np.arange(len(audio)) / sample_rate
        axes[idx, 0].plot(time_audio, audio, linewidth=0.5)
        axes[idx, 0].set_title(f'{name} - Waveform', fontweight='bold')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].grid(True, alpha=0.3)

        # ZCR
        axes[idx, 1].plot(times, features['zcr'], linewidth=2, color='blue')
        axes[idx, 1].fill_between(times, 0, features['zcr'], alpha=0.3, color='blue')
        axes[idx, 1].set_title(f'{name} - ZCR', fontweight='bold')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('ZCR')
        axes[idx, 1].grid(True, alpha=0.3)

        # Energy
        axes[idx, 2].plot(times, features['energy'], linewidth=2, color='red')
        axes[idx, 2].fill_between(times, 0, features['energy'], alpha=0.3, color='red')
        axes[idx, 2].set_title(f'{name} - Energy', fontweight='bold')
        axes[idx, 2].set_xlabel('Time (s)')
        axes[idx, 2].set_ylabel('Energy')
        axes[idx, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('signal_type_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved signal type comparison")
    plt.close()


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("Zero-Crossing Rate and Energy Features Analysis")
    print("=" * 70)

    # Set random seed
    np.random.seed(42)

    # Parameters
    sample_rate = 22050

    # Initialize analyzer
    print("\n1. Initializing analyzer...")
    analyzer = ZCRandEnergyAnalyzer(
        sample_rate=sample_rate,
        frame_size=2048,
        hop_length=512
    )

    # Generate complex signal
    print("\n2. Generating complex audio signal...")
    audio, labels = generate_complex_signal(sample_rate=sample_rate)
    print(f"   - Audio duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"   - Segments: {', '.join(labels)}")

    # Extract features
    print("\n3. Extracting features...")
    features = analyzer.extract_all_features(audio)

    print("\n   Feature Statistics:")
    print("   " + "-" * 60)
    print(f"   ZCR:              Mean={np.mean(features['zcr']):.4f}, Std={np.std(features['zcr']):.4f}")
    print(f"   Energy:           Mean={np.mean(features['energy']):.4f}, Std={np.std(features['energy']):.4f}")
    print(f"   RMS:              Mean={np.mean(features['rms']):.4f}, Std={np.std(features['rms']):.4f}")
    print(f"   Entropy:          Mean={np.mean(features['entropy']):.4f}, Std={np.std(features['entropy']):.4f}")
    print(f"   Voiced frames:    {np.sum(features['voiced'])} / {len(features['voiced'])}")

    # Visualize
    print("\n4. Creating visualizations...")
    visualize_zcr_energy(audio, analyzer)

    # Compare signal types
    print("\n5. Comparing signal types...")
    compare_signal_types()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
