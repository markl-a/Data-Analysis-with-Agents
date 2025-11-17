"""
Voice Activity Detection (VAD)
================================

This solution demonstrates comprehensive voice activity detection algorithms
for identifying speech segments in audio signals.

Dataset: Synthetic audio with speech and non-speech segments
Techniques:
- Energy-based VAD
- Zero-crossing rate VAD
- Spectral entropy VAD
- Statistical model-based VAD
- Deep learning features for VAD
- Multi-feature fusion

Author: Data Science Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VoiceActivityDetector:
    """
    Comprehensive Voice Activity Detection system.
    """

    def __init__(self, sample_rate=16000, frame_size=512, hop_length=256):
        """
        Initialize VAD system.

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

    def frame_audio(self, audio):
        """
        Split audio into overlapping frames.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        frames : ndarray
            Framed audio (n_frames x frame_size)
        """
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_length
        frames = np.zeros((n_frames, self.frame_size))

        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            frames[i] = frame

        return frames

    def energy_vad(self, audio, threshold_factor=0.03):
        """
        Energy-based VAD.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        threshold_factor : float
            Threshold as fraction of max energy

        Returns:
        --------
        vad : ndarray
            Boolean array indicating voice activity
        energy : ndarray
            Frame energies
        """
        frames = self.frame_audio(audio)
        energy = np.sum(frames ** 2, axis=1) / self.frame_size

        # Adaptive threshold
        max_energy = np.max(energy)
        threshold = threshold_factor * max_energy

        vad = energy > threshold
        return vad, energy

    def zcr_vad(self, audio, threshold_low=0.05, threshold_high=0.15):
        """
        Zero-Crossing Rate based VAD.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        threshold_low : float
            Lower ZCR threshold
        threshold_high : float
            Upper ZCR threshold

        Returns:
        --------
        vad : ndarray
            Boolean array indicating voice activity
        zcr : ndarray
            Zero-crossing rates
        """
        frames = self.frame_audio(audio)
        zcr = np.zeros(len(frames))

        for i, frame in enumerate(frames):
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            zcr[i] = zero_crossings / len(frame)

        # Voice typically has moderate ZCR
        vad = (zcr > threshold_low) & (zcr < threshold_high)
        return vad, zcr

    def spectral_entropy_vad(self, audio, threshold=0.7):
        """
        Spectral entropy based VAD.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        threshold : float
            Entropy threshold

        Returns:
        --------
        vad : ndarray
            Boolean array indicating voice activity
        entropy : ndarray
            Spectral entropies
        """
        frames = self.frame_audio(audio)
        entropy = np.zeros(len(frames))

        for i, frame in enumerate(frames):
            # Compute power spectrum
            spectrum = np.abs(rfft(frame)) ** 2
            spectrum = spectrum / (np.sum(spectrum) + 1e-10)

            # Compute entropy
            entropy[i] = -np.sum(spectrum * np.log2(spectrum + 1e-10))

        # Normalize
        entropy = entropy / np.log2(len(spectrum))

        # Speech has lower entropy than noise
        vad = entropy < threshold
        return vad, entropy

    def statistical_model_vad(self, audio, hangover=5):
        """
        Statistical model-based VAD using noise estimation.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal
        hangover : int
            Number of frames to extend speech segments

        Returns:
        --------
        vad : ndarray
            Boolean array indicating voice activity
        """
        frames = self.frame_audio(audio)
        energy = np.sum(frames ** 2, axis=1) / self.frame_size

        # Estimate noise statistics from initial frames
        n_init_frames = min(10, len(frames))
        noise_mean = np.mean(energy[:n_init_frames])
        noise_std = np.std(energy[:n_init_frames])

        # Adaptive threshold
        threshold = noise_mean + 3 * noise_std

        # Initial VAD decision
        vad = energy > threshold

        # Apply hangover (extend speech segments)
        vad_extended = vad.copy()
        speech_counter = 0

        for i in range(len(vad)):
            if vad[i]:
                speech_counter = hangover
            elif speech_counter > 0:
                vad_extended[i] = True
                speech_counter -= 1

        # Update noise estimate during non-speech
        for i in range(len(frames)):
            if not vad_extended[i]:
                # Update noise statistics
                alpha = 0.1
                noise_mean = (1 - alpha) * noise_mean + alpha * energy[i]
                noise_std = (1 - alpha) * noise_std + alpha * np.abs(energy[i] - noise_mean)
                threshold = noise_mean + 3 * noise_std

        return vad_extended

    def multi_feature_vad(self, audio):
        """
        Multi-feature fusion VAD.

        Combines energy, ZCR, and spectral entropy.

        Parameters:
        -----------
        audio : ndarray
            Input audio signal

        Returns:
        --------
        vad : ndarray
            Boolean array indicating voice activity
        features : dict
            Dictionary of features
        """
        # Get individual VAD decisions
        vad_energy, energy = self.energy_vad(audio)
        vad_zcr, zcr = self.zcr_vad(audio)
        vad_entropy, entropy = self.spectral_entropy_vad(audio)

        # Majority voting
        votes = vad_energy.astype(int) + vad_zcr.astype(int) + vad_entropy.astype(int)
        vad = votes >= 2

        features = {
            'energy': energy,
            'zcr': zcr,
            'entropy': entropy,
            'vad_energy': vad_energy,
            'vad_zcr': vad_zcr,
            'vad_entropy': vad_entropy
        }

        return vad, features

    def smooth_vad(self, vad, min_speech_duration=5, min_silence_duration=3):
        """
        Smooth VAD decisions by removing short segments.

        Parameters:
        -----------
        vad : ndarray
            Raw VAD decisions
        min_speech_duration : int
            Minimum speech segment duration (frames)
        min_silence_duration : int
            Minimum silence segment duration (frames)

        Returns:
        --------
        vad_smooth : ndarray
            Smoothed VAD decisions
        """
        vad_smooth = vad.copy()

        # Find speech segments
        diff = np.diff(np.concatenate([[0], vad.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Remove short speech segments
        for start, end in zip(starts, ends):
            if end - start < min_speech_duration:
                vad_smooth[start:end] = False

        # Remove short silence segments
        diff = np.diff(np.concatenate([[0], vad_smooth.astype(int), [0]]))
        starts = np.where(diff == -1)[0]
        ends = np.where(diff == 1)[0]

        for start, end in zip(starts, ends):
            if end - start < min_silence_duration:
                vad_smooth[start:end] = True

        return vad_smooth


def generate_speech_with_noise(speech_duration=1.0, silence_duration=0.5,
                               sample_rate=16000, snr_db=10):
    """
    Generate synthetic speech with noise and silence.

    Parameters:
    -----------
    speech_duration : float
        Duration of speech segment
    silence_duration : float
        Duration of silence segment
    sample_rate : int
        Sample rate
    snr_db : float
        Signal-to-noise ratio in dB

    Returns:
    --------
    audio : ndarray
        Audio with speech and silence
    ground_truth : ndarray
        Ground truth VAD labels (per sample)
    """
    # Generate speech-like signal (formants)
    t_speech = np.linspace(0, speech_duration, int(sample_rate * speech_duration))
    speech = np.zeros_like(t_speech)

    # Add formants
    formants = [500, 1500, 2500]
    for f in formants:
        speech += np.sin(2 * np.pi * f * t_speech)

    # Add pitch variation
    f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t_speech)
    speech += 2 * np.sin(2 * np.pi * f0 * t_speech)

    # Apply envelope
    envelope = np.exp(-t_speech) * (1 - np.exp(-10 * t_speech))
    speech *= envelope

    # Generate silence
    silence = np.zeros(int(sample_rate * silence_duration))

    # Concatenate segments
    audio_clean = np.concatenate([speech, silence, speech, silence, speech])

    # Add noise
    noise = np.random.randn(len(audio_clean))
    signal_power = np.mean(audio_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
    audio = audio_clean + noise

    # Ground truth labels
    ground_truth = np.concatenate([
        np.ones(len(speech)),
        np.zeros(len(silence)),
        np.ones(len(speech)),
        np.zeros(len(silence)),
        np.ones(len(speech))
    ]).astype(bool)

    return audio, ground_truth


def visualize_vad(audio, vad_results, ground_truth, detector, save_path='vad_analysis.png'):
    """
    Visualize VAD results.

    Parameters:
    -----------
    audio : ndarray
        Audio signal
    vad_results : dict
        VAD results from different methods
    ground_truth : ndarray
        Ground truth labels
    detector : VoiceActivityDetector
        VAD detector instance
    save_path : str
        Path to save figure
    """
    fig = plt.figure(figsize=(16, 14))

    times_audio = np.arange(len(audio)) / detector.sample_rate
    times_frames = np.arange(len(vad_results['energy'])) * detector.hop_length / detector.sample_rate

    # Waveform with ground truth
    ax1 = plt.subplot(5, 2, 1)
    plt.plot(times_audio, audio, linewidth=0.5, alpha=0.7)
    # Highlight speech regions
    for i in range(len(ground_truth)):
        if ground_truth[i]:
            start = i / detector.sample_rate
            end = (i + 1) / detector.sample_rate
            plt.axvspan(start, end, alpha=0.2, color='green')
    plt.title('Waveform with Ground Truth (Green=Speech)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Energy-based VAD
    ax2 = plt.subplot(5, 2, 2)
    plt.plot(times_frames, vad_results['energy'], linewidth=2)
    plt.fill_between(times_frames, 0, vad_results['energy'],
                     where=vad_results['vad_energy'], alpha=0.3, color='red', label='Speech')
    plt.title('Energy-based VAD', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ZCR-based VAD
    ax3 = plt.subplot(5, 2, 3)
    plt.plot(times_frames, vad_results['zcr'], linewidth=2)
    plt.fill_between(times_frames, 0, vad_results['zcr'],
                     where=vad_results['vad_zcr'], alpha=0.3, color='blue', label='Speech')
    plt.title('ZCR-based VAD', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('ZCR')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Entropy-based VAD
    ax4 = plt.subplot(5, 2, 4)
    plt.plot(times_frames, vad_results['entropy'], linewidth=2)
    plt.fill_between(times_frames, 0, vad_results['entropy'],
                     where=vad_results['vad_entropy'], alpha=0.3, color='green', label='Speech')
    plt.title('Spectral Entropy-based VAD', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Multi-feature VAD
    ax5 = plt.subplot(5, 2, 5)
    plt.plot(times_audio, audio, linewidth=0.5, alpha=0.5)
    for i in range(len(vad_results['vad_multi'])):
        if vad_results['vad_multi'][i]:
            start = i * detector.hop_length / detector.sample_rate
            end = (i + 1) * detector.hop_length / detector.sample_rate
            plt.axvspan(start, end, alpha=0.3, color='purple')
    plt.title('Multi-Feature Fusion VAD (Purple=Detected Speech)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Statistical VAD
    ax6 = plt.subplot(5, 2, 6)
    plt.plot(times_audio, audio, linewidth=0.5, alpha=0.5)
    for i in range(len(vad_results['vad_statistical'])):
        if vad_results['vad_statistical'][i]:
            start = i * detector.hop_length / detector.sample_rate
            end = (i + 1) * detector.hop_length / detector.sample_rate
            plt.axvspan(start, end, alpha=0.3, color='orange')
    plt.title('Statistical Model VAD (Orange=Detected Speech)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Feature scatter plots
    ax7 = plt.subplot(5, 2, 7)
    speech_idx = vad_results['vad_multi']
    plt.scatter(vad_results['energy'][speech_idx], vad_results['zcr'][speech_idx],
               c='red', label='Detected Speech', alpha=0.6, s=30)
    plt.scatter(vad_results['energy'][~speech_idx], vad_results['zcr'][~speech_idx],
               c='blue', label='Silence', alpha=0.6, s=30)
    plt.title('Energy vs ZCR Feature Space', fontsize=12, fontweight='bold')
    plt.xlabel('Energy')
    plt.ylabel('ZCR')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Performance comparison
    ax8 = plt.subplot(5, 2, 8)
    # Downsample ground truth to frame level
    ground_truth_frames = np.array([
        np.mean(ground_truth[i*detector.hop_length:(i+1)*detector.hop_length])
        for i in range(len(vad_results['vad_energy']))
    ]) > 0.5

    methods = ['Energy', 'ZCR', 'Entropy', 'Multi-Feature', 'Statistical']
    vad_methods = [
        vad_results['vad_energy'],
        vad_results['vad_zcr'],
        vad_results['vad_entropy'],
        vad_results['vad_multi'],
        vad_results['vad_statistical']
    ]

    accuracies = []
    for vad_method in vad_methods:
        accuracy = np.mean(vad_method == ground_truth_frames) * 100
        accuracies.append(accuracy)

    bars = plt.bar(methods, accuracies, color='steelblue', alpha=0.7)
    plt.title('VAD Method Accuracy Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Method')
    plt.ylabel('Accuracy (%)')
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Smoothed VAD
    ax9 = plt.subplot(5, 2, 9)
    plt.plot(times_audio, audio, linewidth=0.5, alpha=0.5)
    for i in range(len(vad_results['vad_smoothed'])):
        if vad_results['vad_smoothed'][i]:
            start = i * detector.hop_length / detector.sample_rate
            end = (i + 1) * detector.hop_length / detector.sample_rate
            plt.axvspan(start, end, alpha=0.3, color='cyan')
    plt.title('Smoothed Multi-Feature VAD (Cyan=Speech)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved VAD analysis to {save_path}")
    plt.close()


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("Voice Activity Detection (VAD)")
    print("=" * 70)

    # Set random seed
    np.random.seed(42)

    # Parameters
    sample_rate = 16000

    # Initialize detector
    print("\n1. Initializing VAD system...")
    detector = VoiceActivityDetector(
        sample_rate=sample_rate,
        frame_size=512,
        hop_length=256
    )

    # Generate test audio
    print("\n2. Generating test audio with speech and silence...")
    audio, ground_truth = generate_speech_with_noise(
        speech_duration=1.0,
        silence_duration=0.5,
        sample_rate=sample_rate,
        snr_db=10
    )
    print(f"   - Audio duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"   - SNR: 10 dB")

    # Run VAD methods
    print("\n3. Running VAD detection methods...")
    vad_energy, energy = detector.energy_vad(audio)
    vad_zcr, zcr = detector.zcr_vad(audio)
    vad_entropy, entropy = detector.spectral_entropy_vad(audio)
    vad_multi, features = detector.multi_feature_vad(audio)
    vad_statistical = detector.statistical_model_vad(audio)
    vad_smoothed = detector.smooth_vad(vad_multi)

    vad_results = {
        'energy': energy,
        'zcr': zcr,
        'entropy': entropy,
        'vad_energy': vad_energy,
        'vad_zcr': vad_zcr,
        'vad_entropy': vad_entropy,
        'vad_multi': vad_multi,
        'vad_statistical': vad_statistical,
        'vad_smoothed': vad_smoothed
    }

    # Compute accuracy
    print("\n4. Computing VAD accuracy...")
    ground_truth_frames = np.array([
        np.mean(ground_truth[i*detector.hop_length:(i+1)*detector.hop_length])
        for i in range(len(vad_energy))
    ]) > 0.5

    print("   " + "-" * 60)
    for name, vad in [('Energy', vad_energy), ('ZCR', vad_zcr),
                      ('Entropy', vad_entropy), ('Multi-Feature', vad_multi),
                      ('Statistical', vad_statistical), ('Smoothed', vad_smoothed)]:
        accuracy = np.mean(vad == ground_truth_frames) * 100
        print(f"   {name:<20} Accuracy: {accuracy:.2f}%")

    # Visualize
    print("\n5. Creating visualizations...")
    visualize_vad(audio, vad_results, ground_truth, detector)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
