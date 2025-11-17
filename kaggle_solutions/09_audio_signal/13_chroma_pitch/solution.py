"""
Chroma Features and Pitch Detection
====================================

This solution demonstrates comprehensive chroma feature extraction and pitch detection
for audio signals, including chromagram computation, pitch tracking, and harmonic analysis.

Dataset: Synthetic musical signals
Techniques:
- Chroma feature extraction
- Chromagram visualization
- Pitch detection using autocorrelation
- YIN pitch detection algorithm
- Harmonic product spectrum
- Constant-Q transform for pitch analysis
- Musical key detection

Author: Data Science Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, rfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ChromaAnalyzer:
    """
    Comprehensive chroma feature extraction and pitch detection.
    """

    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_chroma=12):
        """
        Initialize chroma analyzer.

        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate
        n_fft : int
            FFT window size
        hop_length : int
            Number of samples between frames
        n_chroma : int
            Number of chroma bins (typically 12 for musical pitches)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                          'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Create chroma filterbank
        self.chroma_filters = self._create_chroma_filterbank()

    def hz_to_midi(self, hz):
        """Convert frequency in Hz to MIDI note number."""
        return 12 * np.log2(hz / 440.0) + 69

    def midi_to_hz(self, midi):
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * 2**((midi - 69) / 12)

    def _create_chroma_filterbank(self):
        """
        Create chroma filterbank.

        Returns:
        --------
        chroma_filters : ndarray
            Chroma filterbank matrix
        """
        freqs = rfftfreq(self.n_fft, 1/self.sample_rate)
        chroma_filters = np.zeros((self.n_chroma, len(freqs)))

        # Reference frequency for C0
        c0_hz = self.midi_to_hz(12)

        for i, freq in enumerate(freqs):
            if freq > 0:
                # Convert to MIDI
                midi = self.hz_to_midi(freq)
                # Get chroma bin (0-11)
                chroma_bin = int(np.round(midi)) % 12
                # Add to corresponding chroma
                chroma_filters[chroma_bin, i] += 1

        # Normalize
        chroma_filters = chroma_filters / (np.sum(chroma_filters, axis=0, keepdims=True) + 1e-10)

        return chroma_filters

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
        window = signal.get_window('hann', self.n_fft)
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        stft = np.zeros((self.n_fft // 2 + 1, n_frames))

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            windowed = frame * window
            spectrum = np.abs(rfft(windowed))
            stft[:, i] = spectrum

        return stft

    def extract_chroma(self, audio):
        """
        Extract chroma features.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        chroma : ndarray
            Chroma features (n_chroma x n_frames)
        """
        stft = self.compute_stft(audio)
        chroma = np.dot(self.chroma_filters, stft)

        # Normalize each frame
        chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + 1e-10)

        return chroma

    def autocorrelation_pitch(self, audio, min_pitch=80, max_pitch=400):
        """
        Detect pitch using autocorrelation method.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        min_pitch : float
            Minimum pitch in Hz
        max_pitch : float
            Maximum pitch in Hz

        Returns:
        --------
        pitches : ndarray
            Detected pitch for each frame
        confidences : ndarray
            Confidence for each pitch estimate
        """
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        pitches = np.zeros(n_frames)
        confidences = np.zeros(n_frames)

        # Convert pitch range to lag range
        min_lag = int(self.sample_rate / max_pitch)
        max_lag = int(self.sample_rate / min_pitch)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            # Compute autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks in valid range
            if max_lag < len(autocorr):
                valid_autocorr = autocorr[min_lag:max_lag]
                if len(valid_autocorr) > 0:
                    peak_idx = np.argmax(valid_autocorr)
                    lag = peak_idx + min_lag

                    # Compute pitch
                    pitches[i] = self.sample_rate / lag

                    # Compute confidence
                    if autocorr[0] > 0:
                        confidences[i] = autocorr[lag] / autocorr[0]

        return pitches, confidences

    def yin_pitch(self, audio, min_pitch=80, max_pitch=400, threshold=0.1):
        """
        YIN pitch detection algorithm.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        min_pitch : float
            Minimum pitch in Hz
        max_pitch : float
            Maximum pitch in Hz
        threshold : float
            Threshold for pitch detection

        Returns:
        --------
        pitches : ndarray
            Detected pitch for each frame
        """
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        pitches = np.zeros(n_frames)

        min_lag = int(self.sample_rate / max_pitch)
        max_lag = int(self.sample_rate / min_pitch)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            # Compute difference function
            diff = np.zeros(max_lag)
            for tau in range(1, max_lag):
                diff[tau] = np.sum((frame[:-tau] - frame[tau:])**2)

            # Cumulative mean normalized difference
            cmnd = np.ones(max_lag)
            cumsum = 0
            for tau in range(1, max_lag):
                cumsum += diff[tau]
                if cumsum > 0:
                    cmnd[tau] = diff[tau] * tau / cumsum

            # Find first minimum below threshold
            tau = min_lag
            while tau < max_lag:
                if cmnd[tau] < threshold:
                    # Parabolic interpolation
                    if tau < max_lag - 1 and tau > 0:
                        better_tau = tau + (cmnd[tau+1] - cmnd[tau-1]) / (2 * (2*cmnd[tau] - cmnd[tau+1] - cmnd[tau-1]))
                        pitches[i] = self.sample_rate / better_tau
                    else:
                        pitches[i] = self.sample_rate / tau
                    break
                tau += 1

        return pitches

    def harmonic_product_spectrum(self, audio, n_harmonics=5):
        """
        Harmonic Product Spectrum for pitch detection.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        n_harmonics : int
            Number of harmonics to consider

        Returns:
        --------
        pitches : ndarray
            Detected pitch for each frame
        """
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        pitches = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]

            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))

            # Compute spectrum
            spectrum = np.abs(rfft(frame))
            freqs = rfftfreq(len(frame), 1/self.sample_rate)

            # Compute HPS
            hps = spectrum.copy()
            for h in range(2, n_harmonics + 1):
                # Downsample spectrum
                downsampled = signal.resample(spectrum, len(spectrum) // h)
                # Multiply with original
                min_len = min(len(hps), len(downsampled))
                hps[:min_len] *= downsampled[:min_len]

            # Find peak
            if len(hps) > 0:
                peak_idx = np.argmax(hps[:len(hps)//2])  # Only search lower half
                if peak_idx < len(freqs):
                    pitches[i] = freqs[peak_idx]

        return pitches

    def detect_key(self, chroma):
        """
        Detect musical key from chroma features.

        Parameters:
        -----------
        chroma : ndarray
            Chroma features

        Returns:
        --------
        key : str
            Detected key (e.g., 'C major', 'A minor')
        """
        # Average chroma over time
        avg_chroma = np.mean(chroma, axis=1)

        # Major and minor key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)

        # Try all possible keys
        max_corr = -1
        best_key = None

        for shift in range(12):
            # Major key
            shifted_major = np.roll(major_profile, shift)
            corr = np.corrcoef(avg_chroma, shifted_major)[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_key = f"{self.note_names[shift]} major"

            # Minor key
            shifted_minor = np.roll(minor_profile, shift)
            corr = np.corrcoef(avg_chroma, shifted_minor)[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_key = f"{self.note_names[shift]} minor"

        return best_key


def generate_chord(notes, duration=1.0, sample_rate=22050):
    """
    Generate a musical chord.

    Parameters:
    -----------
    notes : list
        List of MIDI note numbers
    duration : float
        Duration in seconds
    sample_rate : int
        Sample rate

    Returns:
    --------
    audio : ndarray
        Synthesized chord
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)

    for note in notes:
        freq = 440.0 * 2**((note - 69) / 12)
        audio += np.sin(2 * np.pi * freq * t)

        # Add harmonics
        audio += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        audio += 0.15 * np.sin(2 * np.pi * 3 * freq * t)

    # Apply envelope
    envelope = np.exp(-1.5 * t) * (1 - np.exp(-20 * t))
    audio *= envelope

    return audio / np.max(np.abs(audio))


def generate_melody(sample_rate=22050):
    """
    Generate a simple melody.

    Returns:
    --------
    audio : ndarray
        Synthesized melody
    """
    # C major scale melody
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
    segments = []

    for note in notes:
        freq = 440.0 * 2**((note - 69) / 12)
        segment = generate_chord([note], duration=0.3, sample_rate=sample_rate)
        segments.append(segment)

    return np.concatenate(segments)


def visualize_chroma_pitch(audio, analyzer, save_path='chroma_pitch_analysis.png'):
    """
    Visualize chroma and pitch features.

    Parameters:
    -----------
    audio : ndarray
        Audio signal
    analyzer : ChromaAnalyzer
        Chroma analyzer instance
    save_path : str
        Path to save figure
    """
    # Extract features
    chroma = analyzer.extract_chroma(audio)
    pitches_autocorr, confidences = analyzer.autocorrelation_pitch(audio)
    pitches_yin = analyzer.yin_pitch(audio)
    pitches_hps = analyzer.harmonic_product_spectrum(audio)

    fig = plt.figure(figsize=(16, 12))

    # Waveform
    ax1 = plt.subplot(3, 2, 1)
    time = np.arange(len(audio)) / analyzer.sample_rate
    plt.plot(time, audio, linewidth=0.5)
    plt.title('Waveform', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Chromagram
    ax2 = plt.subplot(3, 2, 2)
    times = np.arange(chroma.shape[1]) * analyzer.hop_length / analyzer.sample_rate
    plt.imshow(chroma, aspect='auto', origin='lower', cmap='hot',
               extent=[0, times[-1], 0, analyzer.n_chroma])
    plt.colorbar(label='Chroma Intensity')
    plt.yticks(np.arange(analyzer.n_chroma) + 0.5, analyzer.note_names)
    plt.title('Chromagram', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Class')

    # Autocorrelation pitch
    ax3 = plt.subplot(3, 2, 3)
    plt.plot(times, pitches_autocorr, linewidth=2, label='Pitch')
    plt.scatter(times, pitches_autocorr, c=confidences, cmap='viridis',
               s=20, alpha=0.6, label='Confidence')
    plt.colorbar(label='Confidence')
    plt.title('Pitch Detection (Autocorrelation)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # YIN pitch
    ax4 = plt.subplot(3, 2, 4)
    valid_pitches = pitches_yin[pitches_yin > 0]
    valid_times = times[pitches_yin > 0]
    plt.plot(valid_times, valid_pitches, linewidth=2, color='red')
    plt.scatter(valid_times, valid_pitches, color='red', s=20, alpha=0.6)
    plt.title('Pitch Detection (YIN Algorithm)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)

    # HPS pitch
    ax5 = plt.subplot(3, 2, 5)
    valid_hps = pitches_hps[pitches_hps > 0]
    valid_times_hps = times[pitches_hps > 0]
    plt.plot(valid_times_hps, valid_hps, linewidth=2, color='green')
    plt.scatter(valid_times_hps, valid_hps, color='green', s=20, alpha=0.6)
    plt.title('Pitch Detection (Harmonic Product Spectrum)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)

    # Average chroma
    ax6 = plt.subplot(3, 2, 6)
    avg_chroma = np.mean(chroma, axis=1)
    bars = plt.bar(analyzer.note_names, avg_chroma, color='steelblue', alpha=0.7)
    plt.title('Average Chroma Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Pitch Class')
    plt.ylabel('Average Intensity')
    plt.grid(True, alpha=0.3, axis='y')

    # Highlight dominant notes
    max_idx = np.argmax(avg_chroma)
    bars[max_idx].set_color('red')
    bars[max_idx].set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved chroma/pitch analysis to {save_path}")
    plt.close()


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("Chroma Features and Pitch Detection")
    print("=" * 70)

    # Set random seed
    np.random.seed(42)

    # Parameters
    sample_rate = 22050

    # Initialize analyzer
    print("\n1. Initializing chroma analyzer...")
    analyzer = ChromaAnalyzer(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_chroma=12
    )
    print(f"   - Sample rate: {sample_rate} Hz")
    print(f"   - FFT size: {analyzer.n_fft}")
    print(f"   - Hop length: {analyzer.hop_length}")
    print(f"   - Chroma bins: {analyzer.n_chroma}")

    # Generate melody
    print("\n2. Generating musical melody...")
    audio = generate_melody(sample_rate=sample_rate)
    print(f"   - Audio duration: {len(audio) / sample_rate:.2f} seconds")

    # Extract chroma
    print("\n3. Extracting chroma features...")
    chroma = analyzer.extract_chroma(audio)
    print(f"   - Chroma shape: {chroma.shape}")

    # Detect key
    detected_key = analyzer.detect_key(chroma)
    print(f"   - Detected key: {detected_key}")

    # Detect pitch
    print("\n4. Detecting pitch...")
    pitches_autocorr, confidences = analyzer.autocorrelation_pitch(audio)
    pitches_yin = analyzer.yin_pitch(audio)
    pitches_hps = analyzer.harmonic_product_spectrum(audio)

    print(f"   - Autocorrelation: Mean pitch = {np.mean(pitches_autocorr[pitches_autocorr>0]):.2f} Hz")
    print(f"   - YIN: Mean pitch = {np.mean(pitches_yin[pitches_yin>0]):.2f} Hz")
    print(f"   - HPS: Mean pitch = {np.mean(pitches_hps[pitches_hps>0]):.2f} Hz")

    # Visualize
    print("\n5. Creating visualizations...")
    visualize_chroma_pitch(audio, analyzer)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
