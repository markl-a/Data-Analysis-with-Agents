"""
MFCC (Mel-Frequency Cepstral Coefficients) Extraction and Analysis
===================================================================

This solution demonstrates comprehensive MFCC extraction and analysis for audio signals,
including feature computation, delta/delta-delta features, and visualization.

Dataset: Synthetic audio signals and speech samples
Techniques:
- MFCC computation with multiple configurations
- Delta and delta-delta features
- Mel filterbank analysis
- Feature normalization and standardization
- Time-series visualization
- Statistical analysis of MFCCs

Author: Data Science Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MFCCExtractor:
    """
    Comprehensive MFCC extraction and analysis class.
    """

    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048,
                 hop_length=512, n_mels=40, fmin=0, fmax=None):
        """
        Initialize MFCC extractor.

        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate
        n_mfcc : int
            Number of MFCCs to extract
        n_fft : int
            FFT window size
        hop_length : int
            Number of samples between frames
        n_mels : int
            Number of Mel bands
        fmin : float
            Minimum frequency
        fmax : float
            Maximum frequency
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2

        # Create mel filterbank
        self.mel_filters = self._create_mel_filterbank()

    def hz_to_mel(self, hz):
        """Convert Hz to Mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(self, mel):
        """Convert Mel to Hz scale."""
        return 700 * (10**(mel / 2595) - 1)

    def _create_mel_filterbank(self):
        """
        Create Mel-scale filterbank.

        Returns:
        --------
        mel_filters : ndarray
            Mel filterbank matrix
        """
        # Create Mel-spaced frequencies
        mel_min = self.hz_to_mel(self.fmin)
        mel_max = self.hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)

        # Convert to FFT bin numbers
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Create filterbank
        mel_filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for i in range(1, self.n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            # Rising slope
            for j in range(left, center):
                mel_filters[i - 1, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                mel_filters[i - 1, j] = (right - j) / (right - center)

        return mel_filters

    def compute_stft(self, audio):
        """
        Compute Short-Time Fourier Transform.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        stft : ndarray
            STFT magnitude spectrogram
        """
        # Apply window
        window = signal.get_window('hann', self.n_fft)

        # Compute number of frames
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length

        # Initialize STFT matrix
        stft = np.zeros((self.n_fft // 2 + 1, n_frames))

        # Compute STFT
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            # Apply window and FFT
            windowed = frame * window
            spectrum = np.abs(fft(windowed))[:self.n_fft // 2 + 1]
            stft[:, i] = spectrum

        return stft

    def compute_mel_spectrogram(self, audio):
        """
        Compute Mel spectrogram.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        mel_spec : ndarray
            Mel spectrogram
        """
        # Compute STFT
        stft = self.compute_stft(audio)

        # Apply Mel filterbank
        mel_spec = np.dot(self.mel_filters, stft)

        # Convert to log scale
        mel_spec = np.log(mel_spec + 1e-10)

        return mel_spec

    def dct(self, mel_spec):
        """
        Apply Discrete Cosine Transform.

        Parameters:
        -----------
        mel_spec : ndarray
            Mel spectrogram

        Returns:
        --------
        dct_coeffs : ndarray
            DCT coefficients (MFCCs)
        """
        from scipy.fftpack import dct as scipy_dct
        return scipy_dct(mel_spec, type=2, axis=0, norm='ortho')[:self.n_mfcc]

    def extract_mfcc(self, audio):
        """
        Extract MFCC features.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        mfcc : ndarray
            MFCC features
        """
        mel_spec = self.compute_mel_spectrogram(audio)
        mfcc = self.dct(mel_spec)
        return mfcc

    def compute_delta(self, features, width=9):
        """
        Compute delta features.

        Parameters:
        -----------
        features : ndarray
            Input features
        width : int
            Delta window width

        Returns:
        --------
        delta : ndarray
            Delta features
        """
        delta = np.zeros_like(features)

        for t in range(features.shape[1]):
            # Create window
            start = max(0, t - width // 2)
            end = min(features.shape[1], t + width // 2 + 1)

            # Compute weighted sum
            denominator = sum([(i - t)**2 for i in range(start, end)])
            if denominator > 0:
                numerator = sum([(i - t) * features[:, i] for i in range(start, end)])
                delta[:, t] = numerator / denominator

        return delta

    def extract_features(self, audio):
        """
        Extract MFCC, delta, and delta-delta features.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        features : dict
            Dictionary containing MFCC, delta, and delta-delta features
        """
        mfcc = self.extract_mfcc(audio)
        delta = self.compute_delta(mfcc)
        delta_delta = self.compute_delta(delta)

        return {
            'mfcc': mfcc,
            'delta': delta,
            'delta_delta': delta_delta,
            'combined': np.vstack([mfcc, delta, delta_delta])
        }


def generate_synthetic_speech(duration=2.0, sample_rate=22050):
    """
    Generate synthetic speech-like signal.

    Parameters:
    -----------
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate

    Returns:
    --------
    audio : ndarray
        Synthetic audio signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)

    # Add formants (resonant frequencies in speech)
    formants = [700, 1220, 2600, 3500]  # Typical vowel formants

    for f in formants:
        # Add fundamental and harmonics
        audio += np.sin(2 * np.pi * f * t) * np.exp(-5 * t)
        audio += 0.5 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-7 * t)

    # Add pitch modulation
    pitch = 100 + 50 * np.sin(2 * np.pi * 3 * t)
    audio += 0.3 * np.sin(2 * np.pi * pitch * t)

    # Add noise
    audio += 0.05 * np.random.randn(len(t))

    # Apply envelope
    envelope = np.exp(-2 * t) * (1 - np.exp(-10 * t))
    audio *= envelope

    return audio / np.max(np.abs(audio))


def generate_vowel_sound(vowel='a', duration=1.0, sample_rate=22050):
    """
    Generate synthetic vowel sound with specific formants.

    Parameters:
    -----------
    vowel : str
        Vowel to generate ('a', 'e', 'i', 'o', 'u')
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate

    Returns:
    --------
    audio : ndarray
        Synthetic vowel audio
    """
    # Formant frequencies for different vowels
    formant_map = {
        'a': [730, 1090, 2440],
        'e': [530, 1840, 2480],
        'i': [270, 2290, 3010],
        'o': [570, 840, 2410],
        'u': [440, 1020, 2240]
    }

    formants = formant_map.get(vowel.lower(), formant_map['a'])
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate fundamental frequency (pitch)
    f0 = 120  # Hz
    audio = np.zeros_like(t)

    # Generate harmonics at formant frequencies
    for i, formant in enumerate(formants):
        bandwidth = 50 + i * 30
        audio += np.sin(2 * np.pi * formant * t) * np.exp(-bandwidth * t)

    # Add pitch
    audio += 0.5 * np.sin(2 * np.pi * f0 * t)

    # Normalize
    return audio / np.max(np.abs(audio))


def visualize_mfcc_analysis(audio, mfcc_extractor, save_path='mfcc_analysis.png'):
    """
    Create comprehensive MFCC visualization.

    Parameters:
    -----------
    audio : ndarray
        Audio signal
    mfcc_extractor : MFCCExtractor
        MFCC extractor instance
    save_path : str
        Path to save figure
    """
    features = mfcc_extractor.extract_features(audio)
    mel_spec = mfcc_extractor.compute_mel_spectrogram(audio)

    fig = plt.figure(figsize=(16, 12))

    # Waveform
    ax1 = plt.subplot(3, 2, 1)
    time = np.arange(len(audio)) / mfcc_extractor.sample_rate
    plt.plot(time, audio, linewidth=0.5)
    plt.title('Waveform', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Mel Spectrogram
    ax2 = plt.subplot(3, 2, 2)
    times = np.arange(mel_spec.shape[1]) * mfcc_extractor.hop_length / mfcc_extractor.sample_rate
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, times[-1], 0, mfcc_extractor.n_mels])
    plt.colorbar(label='Log Power')
    plt.title('Mel Spectrogram', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Band')

    # MFCC
    ax3 = plt.subplot(3, 2, 3)
    plt.imshow(features['mfcc'], aspect='auto', origin='lower', cmap='coolwarm',
               extent=[0, times[-1], 1, mfcc_extractor.n_mfcc])
    plt.colorbar(label='MFCC Value')
    plt.title('MFCC Features', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficient')

    # Delta MFCC
    ax4 = plt.subplot(3, 2, 4)
    plt.imshow(features['delta'], aspect='auto', origin='lower', cmap='coolwarm',
               extent=[0, times[-1], 1, mfcc_extractor.n_mfcc])
    plt.colorbar(label='Delta Value')
    plt.title('Delta MFCC', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficient')

    # Delta-Delta MFCC
    ax5 = plt.subplot(3, 2, 5)
    plt.imshow(features['delta_delta'], aspect='auto', origin='lower', cmap='coolwarm',
               extent=[0, times[-1], 1, mfcc_extractor.n_mfcc])
    plt.colorbar(label='Delta-Delta Value')
    plt.title('Delta-Delta MFCC', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficient')

    # MFCC Statistics
    ax6 = plt.subplot(3, 2, 6)
    mfcc_mean = np.mean(features['mfcc'], axis=1)
    mfcc_std = np.std(features['mfcc'], axis=1)
    x = np.arange(1, mfcc_extractor.n_mfcc + 1)
    plt.bar(x, mfcc_mean, yerr=mfcc_std, alpha=0.7, capsize=5)
    plt.title('MFCC Statistics (Mean ± Std)', fontsize=12, fontweight='bold')
    plt.xlabel('MFCC Coefficient')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved MFCC analysis to {save_path}")
    plt.close()


def compare_vowel_mfccs(sample_rate=22050):
    """
    Compare MFCC features across different vowels.
    """
    vowels = ['a', 'e', 'i', 'o', 'u']
    mfcc_extractor = MFCCExtractor(sample_rate=sample_rate)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    all_mfccs = []

    for idx, vowel in enumerate(vowels):
        audio = generate_vowel_sound(vowel, duration=0.5, sample_rate=sample_rate)
        mfcc = mfcc_extractor.extract_mfcc(audio)
        all_mfccs.append(mfcc)

        # Plot MFCC for each vowel
        times = np.arange(mfcc.shape[1]) * mfcc_extractor.hop_length / sample_rate
        im = axes[idx].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis',
                             extent=[0, times[-1], 1, mfcc_extractor.n_mfcc])
        axes[idx].set_title(f'Vowel: {vowel.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('MFCC Coefficient')
        plt.colorbar(im, ax=axes[idx])

    # Plot average MFCC comparison
    axes[5].remove()
    ax_new = fig.add_subplot(2, 3, 6)

    for idx, vowel in enumerate(vowels):
        mfcc_mean = np.mean(all_mfccs[idx], axis=1)
        ax_new.plot(range(1, len(mfcc_mean) + 1), mfcc_mean,
                   marker='o', label=vowel.upper(), linewidth=2)

    ax_new.set_title('Average MFCC Comparison', fontsize=12, fontweight='bold')
    ax_new.set_xlabel('MFCC Coefficient')
    ax_new.set_ylabel('Mean Value')
    ax_new.legend()
    ax_new.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vowel_mfcc_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved vowel MFCC comparison")
    plt.close()


def analyze_mfcc_temporal_dynamics():
    """
    Analyze temporal dynamics of MFCC features.
    """
    # Generate speech with varying characteristics
    sample_rate = 22050
    mfcc_extractor = MFCCExtractor(sample_rate=sample_rate)

    # Create concatenated vowel sequence
    vowels = ['a', 'e', 'i', 'o', 'u']
    audio_segments = []

    for vowel in vowels:
        segment = generate_vowel_sound(vowel, duration=0.3, sample_rate=sample_rate)
        audio_segments.append(segment)

    audio = np.concatenate(audio_segments)
    features = mfcc_extractor.extract_features(audio)

    # Visualize temporal dynamics
    fig = plt.figure(figsize=(16, 10))

    # Plot first 3 MFCC coefficients over time
    ax1 = plt.subplot(3, 1, 1)
    times = np.arange(features['mfcc'].shape[1]) * mfcc_extractor.hop_length / sample_rate

    for i in range(3):
        plt.plot(times, features['mfcc'][i, :], label=f'MFCC {i+1}', linewidth=2)

    plt.title('MFCC Temporal Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add vowel boundaries
    boundary_times = np.cumsum([0.3] * 5)
    for t in boundary_times[:-1]:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)

    # Plot delta features
    ax2 = plt.subplot(3, 1, 2)
    for i in range(3):
        plt.plot(times, features['delta'][i, :], label=f'Delta {i+1}', linewidth=2)

    plt.title('Delta MFCC Temporal Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Delta Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    for t in boundary_times[:-1]:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)

    # Plot combined feature heatmap
    ax3 = plt.subplot(3, 1, 3)
    plt.imshow(features['combined'], aspect='auto', origin='lower', cmap='RdBu_r',
               extent=[0, times[-1], 1, features['combined'].shape[0]])
    plt.colorbar(label='Feature Value')
    plt.title('Combined MFCC Features (MFCC + Delta + Delta-Delta)',
             fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Feature Index')

    for t in boundary_times[:-1]:
        plt.axvline(x=t, color='yellow', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('mfcc_temporal_dynamics.png', dpi=300, bbox_inches='tight')
    print("Saved temporal dynamics analysis")
    plt.close()


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("MFCC Extraction and Analysis")
    print("=" * 70)

    # Set random seed
    np.random.seed(42)

    # Parameters
    sample_rate = 22050

    # Initialize MFCC extractor
    print("\n1. Initializing MFCC extractor...")
    mfcc_extractor = MFCCExtractor(
        sample_rate=sample_rate,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        n_mels=40
    )
    print(f"   - Sample rate: {sample_rate} Hz")
    print(f"   - Number of MFCCs: {mfcc_extractor.n_mfcc}")
    print(f"   - FFT size: {mfcc_extractor.n_fft}")
    print(f"   - Hop length: {mfcc_extractor.hop_length}")
    print(f"   - Number of Mel bands: {mfcc_extractor.n_mels}")

    # Generate synthetic speech
    print("\n2. Generating synthetic speech signal...")
    audio = generate_synthetic_speech(duration=2.0, sample_rate=sample_rate)
    print(f"   - Audio duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"   - Audio samples: {len(audio)}")

    # Extract MFCC features
    print("\n3. Extracting MFCC features...")
    features = mfcc_extractor.extract_features(audio)
    print(f"   - MFCC shape: {features['mfcc'].shape}")
    print(f"   - Delta shape: {features['delta'].shape}")
    print(f"   - Delta-Delta shape: {features['delta_delta'].shape}")
    print(f"   - Combined features: {features['combined'].shape}")

    # Compute statistics
    print("\n4. Computing MFCC statistics...")
    mfcc_mean = np.mean(features['mfcc'], axis=1)
    mfcc_std = np.std(features['mfcc'], axis=1)
    mfcc_min = np.min(features['mfcc'], axis=1)
    mfcc_max = np.max(features['mfcc'], axis=1)

    print("\n   MFCC Statistics:")
    print("   " + "-" * 60)
    print(f"   {'Coeff':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("   " + "-" * 60)
    for i in range(mfcc_extractor.n_mfcc):
        print(f"   {i+1:<8} {mfcc_mean[i]:<12.4f} {mfcc_std[i]:<12.4f} "
              f"{mfcc_min[i]:<12.4f} {mfcc_max[i]:<12.4f}")

    # Visualize analysis
    print("\n5. Creating visualizations...")
    visualize_mfcc_analysis(audio, mfcc_extractor)

    # Compare vowel MFCCs
    print("\n6. Comparing vowel MFCCs...")
    compare_vowel_mfccs(sample_rate)

    # Analyze temporal dynamics
    print("\n7. Analyzing temporal dynamics...")
    analyze_mfcc_temporal_dynamics()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - mfcc_analysis.png")
    print("  - vowel_mfcc_comparison.png")
    print("  - mfcc_temporal_dynamics.png")


if __name__ == "__main__":
    main()
