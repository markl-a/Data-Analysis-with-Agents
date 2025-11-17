"""
Spectral Features Analysis (Centroid, Rolloff, Flux, Bandwidth)
================================================================

This solution demonstrates comprehensive spectral feature extraction and analysis
for audio signals, including centroid, rolloff, flux, and bandwidth computation.

Dataset: Synthetic audio signals and music samples
Techniques:
- Spectral centroid computation
- Spectral rolloff analysis
- Spectral flux measurement
- Spectral bandwidth calculation
- Time-frequency representations
- Statistical analysis of spectral features

Author: Data Science Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SpectralAnalyzer:
    """
    Comprehensive spectral feature extraction and analysis.
    """

    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512):
        """
        Initialize spectral analyzer.

        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate
        n_fft : int
            FFT window size
        hop_length : int
            Number of samples between frames
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freqs = rfftfreq(n_fft, 1/sample_rate)

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
        phase : ndarray
            STFT phase spectrogram
        """
        # Apply window
        window = signal.get_window('hann', self.n_fft)

        # Compute number of frames
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length

        # Initialize STFT matrices
        stft_complex = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=complex)

        # Compute STFT
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            # Apply window and FFT
            windowed = frame * window
            spectrum = rfft(windowed)
            stft_complex[:, i] = spectrum

        magnitude = np.abs(stft_complex)
        phase = np.angle(stft_complex)

        return magnitude, phase

    def spectral_centroid(self, audio):
        """
        Compute spectral centroid.

        The spectral centroid indicates where the "center of mass" of the spectrum is.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        centroids : ndarray
            Spectral centroid for each frame (in Hz)
        """
        stft, _ = self.compute_stft(audio)
        centroids = np.zeros(stft.shape[1])

        for i in range(stft.shape[1]):
            spectrum = stft[:, i]
            if np.sum(spectrum) > 0:
                centroids[i] = np.sum(self.freqs * spectrum) / np.sum(spectrum)
            else:
                centroids[i] = 0

        return centroids

    def spectral_rolloff(self, audio, rolloff_percent=0.85):
        """
        Compute spectral rolloff.

        The frequency below which a specified percentage of the total spectral energy is contained.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        rolloff_percent : float
            Percentage threshold (default: 0.85 = 85%)

        Returns:
        --------
        rolloffs : ndarray
            Spectral rolloff frequency for each frame (in Hz)
        """
        stft, _ = self.compute_stft(audio)
        rolloffs = np.zeros(stft.shape[1])

        for i in range(stft.shape[1]):
            spectrum = stft[:, i]
            cumsum = np.cumsum(spectrum)
            total_energy = cumsum[-1]

            if total_energy > 0:
                threshold = rolloff_percent * total_energy
                idx = np.where(cumsum >= threshold)[0]
                if len(idx) > 0:
                    rolloffs[i] = self.freqs[idx[0]]
            else:
                rolloffs[i] = 0

        return rolloffs

    def spectral_flux(self, audio):
        """
        Compute spectral flux.

        Measures the rate of change of the power spectrum.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        flux : ndarray
            Spectral flux for each frame
        """
        stft, _ = self.compute_stft(audio)
        flux = np.zeros(stft.shape[1])

        # Normalize spectra
        stft_norm = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-10)

        for i in range(1, stft.shape[1]):
            # Compute difference between consecutive frames
            diff = stft_norm[:, i] - stft_norm[:, i-1]
            # Sum of squared differences (only positive changes)
            flux[i] = np.sum(np.maximum(diff, 0) ** 2)

        return flux

    def spectral_bandwidth(self, audio, p=2):
        """
        Compute spectral bandwidth.

        Measures the width of the spectrum.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        p : int
            Power (default: 2 for variance-based bandwidth)

        Returns:
        --------
        bandwidths : ndarray
            Spectral bandwidth for each frame (in Hz)
        """
        stft, _ = self.compute_stft(audio)
        centroids = self.spectral_centroid(audio)
        bandwidths = np.zeros(stft.shape[1])

        for i in range(stft.shape[1]):
            spectrum = stft[:, i]
            if np.sum(spectrum) > 0:
                deviation = np.abs(self.freqs - centroids[i]) ** p
                bandwidths[i] = (np.sum(deviation * spectrum) / np.sum(spectrum)) ** (1/p)
            else:
                bandwidths[i] = 0

        return bandwidths

    def spectral_contrast(self, audio, n_bands=6):
        """
        Compute spectral contrast.

        Measures the difference in amplitude between peaks and valleys in the spectrum.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        n_bands : int
            Number of frequency bands

        Returns:
        --------
        contrast : ndarray
            Spectral contrast features
        """
        stft, _ = self.compute_stft(audio)

        # Create frequency bands (octave spacing)
        freq_bands = np.logspace(np.log10(200), np.log10(self.sample_rate/2),
                                 n_bands + 1)
        band_indices = np.searchsorted(self.freqs, freq_bands)

        contrast = np.zeros((n_bands, stft.shape[1]))

        for i in range(n_bands):
            start_idx = band_indices[i]
            end_idx = band_indices[i + 1]

            for j in range(stft.shape[1]):
                band_spectrum = stft[start_idx:end_idx, j]
                if len(band_spectrum) > 0:
                    peak = np.percentile(band_spectrum, 90)
                    valley = np.percentile(band_spectrum, 10)
                    contrast[i, j] = peak - valley

        return contrast

    def spectral_flatness(self, audio):
        """
        Compute spectral flatness (Wiener entropy).

        Measures how noise-like vs. tone-like a sound is.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        flatness : ndarray
            Spectral flatness for each frame
        """
        stft, _ = self.compute_stft(audio)
        flatness = np.zeros(stft.shape[1])

        for i in range(stft.shape[1]):
            spectrum = stft[:, i] + 1e-10
            geometric_mean = np.exp(np.mean(np.log(spectrum)))
            arithmetic_mean = np.mean(spectrum)
            flatness[i] = geometric_mean / arithmetic_mean

        return flatness

    def extract_all_features(self, audio):
        """
        Extract all spectral features.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        features : dict
            Dictionary of all spectral features
        """
        return {
            'centroid': self.spectral_centroid(audio),
            'rolloff': self.spectral_rolloff(audio),
            'flux': self.spectral_flux(audio),
            'bandwidth': self.spectral_bandwidth(audio),
            'contrast': self.spectral_contrast(audio),
            'flatness': self.spectral_flatness(audio)
        }


def generate_musical_note(frequency, duration=1.0, sample_rate=22050):
    """
    Generate a musical note with harmonics.

    Parameters:
    -----------
    frequency : float
        Fundamental frequency in Hz
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

    # Add fundamental and harmonics
    harmonics = [1, 2, 3, 4, 5, 6]
    amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]

    for harm, amp in zip(harmonics, amplitudes):
        audio += amp * np.sin(2 * np.pi * frequency * harm * t)

    # Apply ADSR envelope
    attack = int(0.1 * len(t))
    decay = int(0.2 * len(t))
    release = int(0.3 * len(t))

    envelope = np.ones_like(t)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
    envelope[-release:] = np.linspace(0.7, 0, release)

    audio *= envelope
    return audio / np.max(np.abs(audio))


def generate_complex_signal(sample_rate=22050, duration=3.0):
    """
    Generate a complex audio signal with varying spectral characteristics.

    Parameters:
    -----------
    sample_rate : int
        Sample rate
    duration : float
        Duration in seconds

    Returns:
    --------
    audio : ndarray
        Complex synthetic audio signal
    """
    # Generate musical scale
    notes = [261.63, 293.66, 329.63, 349.23, 392.00]  # C, D, E, F, G
    segment_duration = duration / len(notes)

    segments = []
    for note in notes:
        segment = generate_musical_note(note, duration=segment_duration,
                                       sample_rate=sample_rate)
        segments.append(segment)

    audio = np.concatenate(segments)

    # Add noise burst in the middle
    noise_start = len(audio) // 2
    noise_duration = int(0.2 * sample_rate)
    audio[noise_start:noise_start + noise_duration] += 0.3 * np.random.randn(noise_duration)

    return audio / np.max(np.abs(audio))


def visualize_spectral_features(audio, analyzer, save_path='spectral_features.png'):
    """
    Create comprehensive spectral features visualization.

    Parameters:
    -----------
    audio : ndarray
        Audio signal
    analyzer : SpectralAnalyzer
        Spectral analyzer instance
    save_path : str
        Path to save figure
    """
    features = analyzer.extract_all_features(audio)
    stft, _ = analyzer.compute_stft(audio)

    fig = plt.figure(figsize=(16, 14))

    # Waveform
    ax1 = plt.subplot(4, 2, 1)
    time = np.arange(len(audio)) / analyzer.sample_rate
    plt.plot(time, audio, linewidth=0.5)
    plt.title('Waveform', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Spectrogram
    ax2 = plt.subplot(4, 2, 2)
    times = np.arange(stft.shape[1]) * analyzer.hop_length / analyzer.sample_rate
    plt.imshow(20 * np.log10(stft + 1e-10), aspect='auto', origin='lower',
               cmap='viridis', extent=[0, times[-1], 0, analyzer.sample_rate/2])
    plt.colorbar(label='Power (dB)')
    plt.title('Spectrogram', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Spectral Centroid
    ax3 = plt.subplot(4, 2, 3)
    plt.plot(times, features['centroid'], linewidth=2, color='red')
    plt.fill_between(times, 0, features['centroid'], alpha=0.3, color='red')
    plt.title('Spectral Centroid', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)

    # Spectral Rolloff
    ax4 = plt.subplot(4, 2, 4)
    plt.plot(times, features['rolloff'], linewidth=2, color='blue')
    plt.fill_between(times, 0, features['rolloff'], alpha=0.3, color='blue')
    plt.title('Spectral Rolloff (85%)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)

    # Spectral Flux
    ax5 = plt.subplot(4, 2, 5)
    plt.plot(times, features['flux'], linewidth=2, color='green')
    plt.fill_between(times, 0, features['flux'], alpha=0.3, color='green')
    plt.title('Spectral Flux', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Flux')
    plt.grid(True, alpha=0.3)

    # Spectral Bandwidth
    ax6 = plt.subplot(4, 2, 6)
    plt.plot(times, features['bandwidth'], linewidth=2, color='purple')
    plt.fill_between(times, 0, features['bandwidth'], alpha=0.3, color='purple')
    plt.title('Spectral Bandwidth', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Bandwidth (Hz)')
    plt.grid(True, alpha=0.3)

    # Spectral Contrast
    ax7 = plt.subplot(4, 2, 7)
    plt.imshow(features['contrast'], aspect='auto', origin='lower',
               cmap='coolwarm', extent=[0, times[-1], 1, features['contrast'].shape[0]])
    plt.colorbar(label='Contrast')
    plt.title('Spectral Contrast', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Band')

    # Spectral Flatness
    ax8 = plt.subplot(4, 2, 8)
    plt.plot(times, features['flatness'], linewidth=2, color='orange')
    plt.fill_between(times, 0, features['flatness'], alpha=0.3, color='orange')
    plt.title('Spectral Flatness', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Flatness')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved spectral features visualization to {save_path}")
    plt.close()


def compare_instruments():
    """
    Compare spectral features across different instrument-like sounds.
    """
    sample_rate = 22050
    analyzer = SpectralAnalyzer(sample_rate=sample_rate)

    # Generate different instrument sounds
    instruments = {
        'Flute': generate_musical_note(523.25, duration=1.0),  # Pure tone
        'Violin': generate_musical_note(440.00, duration=1.0),  # Rich harmonics
        'Noise': np.random.randn(sample_rate),  # White noise
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    for idx, (name, audio) in enumerate(instruments.items()):
        features = analyzer.extract_all_features(audio)
        times = np.arange(len(features['centroid'])) * analyzer.hop_length / sample_rate

        # Centroid
        axes[idx, 0].plot(times, features['centroid'], linewidth=2)
        axes[idx, 0].set_title(f'{name} - Spectral Centroid', fontweight='bold')
        axes[idx, 0].set_xlabel('Time (s)')
        axes[idx, 0].set_ylabel('Frequency (Hz)')
        axes[idx, 0].grid(True, alpha=0.3)

        # Flatness
        axes[idx, 1].plot(times, features['flatness'], linewidth=2, color='orange')
        axes[idx, 1].set_title(f'{name} - Spectral Flatness', fontweight='bold')
        axes[idx, 1].set_xlabel('Time (s)')
        axes[idx, 1].set_ylabel('Flatness')
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('instrument_spectral_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved instrument spectral comparison")
    plt.close()


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("Spectral Features Analysis")
    print("=" * 70)

    # Set random seed
    np.random.seed(42)

    # Parameters
    sample_rate = 22050

    # Initialize analyzer
    print("\n1. Initializing spectral analyzer...")
    analyzer = SpectralAnalyzer(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512
    )
    print(f"   - Sample rate: {sample_rate} Hz")
    print(f"   - FFT size: {analyzer.n_fft}")
    print(f"   - Hop length: {analyzer.hop_length}")

    # Generate complex signal
    print("\n2. Generating complex audio signal...")
    audio = generate_complex_signal(sample_rate=sample_rate, duration=3.0)
    print(f"   - Audio duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"   - Audio samples: {len(audio)}")

    # Extract all features
    print("\n3. Extracting spectral features...")
    features = analyzer.extract_all_features(audio)

    print("\n   Feature Statistics:")
    print("   " + "-" * 60)
    for name, values in features.items():
        if len(values.shape) == 1:
            print(f"   {name.capitalize():<20} Mean: {np.mean(values):.2f}  "
                  f"Std: {np.std(values):.2f}")
        else:
            print(f"   {name.capitalize():<20} Shape: {values.shape}")

    # Visualize features
    print("\n4. Creating visualizations...")
    visualize_spectral_features(audio, analyzer)

    # Compare instruments
    print("\n5. Comparing instrument-like sounds...")
    compare_instruments()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
