"""
Audio Noise Reduction
=====================

This solution demonstrates noise reduction techniques for audio signals.
We generate clean audio with various types of noise and implement
multiple denoising algorithms.

Author: Kaggle Solutions Team
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft
import warnings
warnings.filterwarnings('ignore')


class AudioDenoiser:
    """Audio noise reduction using signal processing techniques"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_types = ['white', 'pink', 'hum', 'click']

    def generate_clean_signal(self, duration=3.0, signal_type='speech'):
        """Generate clean audio signal"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        if signal_type == 'speech':
            # Speech-like signal with formants
            audio = np.zeros_like(t)
            f0 = 150  # Fundamental frequency

            # Add harmonics
            for harmonic in range(1, 8):
                freq = f0 * harmonic
                audio += (1/harmonic) * np.sin(2 * np.pi * freq * t)

            # Modulate with speech-like pattern
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
            audio *= modulation

            # Apply envelope
            envelope = np.exp(-0.5 * np.sin(2 * np.pi * 0.5 * t))
            audio *= envelope

        elif signal_type == 'music':
            # Musical signal with chords
            frequencies = [262, 330, 392]  # C major chord
            audio = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

            # Add rhythm
            rhythm = 1 + 0.5 * signal.square(2 * np.pi * 2 * t)
            audio *= rhythm

        else:  # tone
            audio = np.sin(2 * np.pi * 440 * t)

        return audio / np.max(np.abs(audio))

    def add_noise(self, clean_audio, noise_type, snr_db=10):
        """Add specific type of noise to clean audio"""
        # Calculate noise power based on desired SNR
        signal_power = np.mean(clean_audio**2)
        noise_power = signal_power / (10**(snr_db / 10))

        if noise_type == 'white':
            # White noise (flat spectrum)
            noise = np.random.normal(0, np.sqrt(noise_power), len(clean_audio))

        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            white = np.random.randn(len(clean_audio))
            fft_white = fft(white)
            freqs = np.fft.fftfreq(len(clean_audio))

            # Apply 1/f filter
            pink_filter = 1 / np.sqrt(np.abs(freqs) + 1e-10)
            fft_pink = fft_white * pink_filter

            noise = np.real(ifft(fft_pink))
            # Adjust to desired power
            noise = noise * np.sqrt(noise_power / np.mean(noise**2))

        elif noise_type == 'hum':
            # Power line hum (50/60 Hz and harmonics)
            t = np.linspace(0, len(clean_audio)/self.sample_rate, len(clean_audio))
            noise = np.zeros_like(clean_audio)

            for harmonic in [1, 2, 3]:
                noise += np.sin(2 * np.pi * 60 * harmonic * t)

            noise = noise * np.sqrt(noise_power / np.mean(noise**2))

        elif noise_type == 'click':
            # Random clicks/pops
            noise = np.zeros_like(clean_audio)
            n_clicks = int(len(clean_audio) / self.sample_rate * 5)  # 5 clicks per second

            for _ in range(n_clicks):
                click_pos = np.random.randint(0, len(clean_audio))
                click_width = np.random.randint(1, 10)
                if click_pos + click_width < len(clean_audio):
                    noise[click_pos:click_pos+click_width] = np.random.uniform(-1, 1, click_width)

            noise = noise * np.sqrt(noise_power / np.mean(noise**2 + 1e-10))

        noisy_audio = clean_audio + noise
        return noisy_audio, noise

    def spectral_subtraction(self, noisy_audio, noise_estimate=None):
        """Spectral subtraction denoising"""
        # If no noise estimate provided, use beginning of signal
        if noise_estimate is None:
            noise_estimate = noisy_audio[:int(0.5 * self.sample_rate)]

        # Compute noise spectrum
        noise_fft = fft(noise_estimate)
        noise_magnitude = np.abs(noise_fft)
        avg_noise_magnitude = np.mean(noise_magnitude)

        # Process noisy signal
        noisy_fft = fft(noisy_audio)
        noisy_magnitude = np.abs(noisy_fft)
        noisy_phase = np.angle(noisy_fft)

        # Subtract noise spectrum
        clean_magnitude = np.maximum(noisy_magnitude - avg_noise_magnitude, 0)

        # Reconstruct signal
        clean_fft = clean_magnitude * np.exp(1j * noisy_phase)
        clean_audio = np.real(ifft(clean_fft))

        return clean_audio

    def wiener_filter(self, noisy_audio, noise_estimate=None):
        """Wiener filtering denoising"""
        if noise_estimate is None:
            noise_estimate = noisy_audio[:int(0.5 * self.sample_rate)]

        # Estimate noise power spectrum
        noise_fft = fft(noise_estimate)
        noise_power = np.abs(noise_fft)**2

        # Noisy signal spectrum
        noisy_fft = fft(noisy_audio)
        noisy_power = np.abs(noisy_fft)**2

        # Wiener filter
        wiener_gain = np.maximum(1 - noise_power / (noisy_power + 1e-10), 0)

        # Apply filter
        clean_fft = noisy_fft * wiener_gain
        clean_audio = np.real(ifft(clean_fft))

        return clean_audio

    def median_filter(self, noisy_audio, kernel_size=5):
        """Median filtering (good for click removal)"""
        return signal.medfilt(noisy_audio, kernel_size=kernel_size)

    def lowpass_filter(self, noisy_audio, cutoff=4000):
        """Low-pass filter denoising"""
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        sos = signal.butter(5, normal_cutoff, btype='low', output='sos')
        return signal.sosfilt(sos, noisy_audio)

    def notch_filter(self, noisy_audio, freq=60, Q=30):
        """Notch filter for hum removal"""
        sos = signal.iirnotch(freq, Q, self.sample_rate)
        # Apply to fundamental and harmonics
        filtered = noisy_audio.copy()
        for harmonic in [1, 2, 3]:
            sos = signal.iirnotch(freq * harmonic, Q, self.sample_rate)
            filtered = signal.sosfilt(sos, filtered)
        return filtered


def calculate_snr(clean, noisy):
    """Calculate Signal-to-Noise Ratio"""
    noise = noisy - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')
    return snr


def visualize_denoising(denoiser, clean, noisy, denoised, noise_type):
    """Visualize denoising results"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    t = np.linspace(0, len(clean)/denoiser.sample_rate, len(clean))

    # Time domain plots
    axes[0, 0].plot(t, clean, linewidth=0.5, color='green', alpha=0.7)
    axes[0, 0].set_title('Clean Signal', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(t, noisy, linewidth=0.5, color='red', alpha=0.7)
    axes[1, 0].set_title(f'Noisy Signal ({noise_type.capitalize()} Noise)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(t, denoised, linewidth=0.5, color='blue', alpha=0.7)
    axes[2, 0].set_title('Denoised Signal', fontweight='bold')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].grid(True, alpha=0.3)

    # Frequency domain plots
    for idx, (audio, title, color) in enumerate([
        (clean, 'Clean Spectrum', 'green'),
        (noisy, 'Noisy Spectrum', 'red'),
        (denoised, 'Denoised Spectrum', 'blue')
    ]):
        fft_vals = fft(audio)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = np.fft.fftfreq(len(audio), 1/denoiser.sample_rate)[:len(audio)//2]

        axes[idx, 1].plot(freqs, 20*np.log10(fft_mag + 1e-10),
                         linewidth=0.7, color=color, alpha=0.7)
        axes[idx, 1].set_title(title, fontweight='bold')
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude (dB)')
        axes[idx, 1].set_xlim([0, 5000])
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'noise_reduction_{noise_type}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: noise_reduction_{noise_type}.png")
    plt.close()


def compare_methods(denoiser, clean, noisy, noise_type):
    """Compare different denoising methods"""
    methods = {
        'Spectral Subtraction': denoiser.spectral_subtraction(noisy),
        'Wiener Filter': denoiser.wiener_filter(noisy),
        'Median Filter': denoiser.median_filter(noisy),
        'Lowpass Filter': denoiser.lowpass_filter(noisy)
    }

    if noise_type == 'hum':
        methods['Notch Filter'] = denoiser.notch_filter(noisy)

    # Calculate SNR for each method
    results = {}
    for method_name, denoised in methods.items():
        # Ensure same length
        min_len = min(len(clean), len(denoised))
        snr_improvement = calculate_snr(clean[:min_len], denoised[:min_len]) - \
                         calculate_snr(clean[:min_len], noisy[:min_len])
        results[method_name] = snr_improvement

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    method_names = list(results.keys())
    improvements = list(results.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(method_names)))

    bars = ax.barh(method_names, improvements, color=colors, alpha=0.7)
    ax.set_xlabel('SNR Improvement (dB)', fontsize=12)
    ax.set_title(f'Denoising Method Comparison - {noise_type.capitalize()} Noise',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val, i, f' {val:.2f} dB', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'method_comparison_{noise_type}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: method_comparison_{noise_type}.png")
    plt.close()

    return results


def main():
    """Main execution function"""
    print("=" * 60)
    print("Audio Noise Reduction")
    print("=" * 60)

    # Initialize denoiser
    denoiser = AudioDenoiser(sample_rate=16000)

    # Generate clean signal
    print("\n1. Generating clean audio signal...")
    clean_audio = denoiser.generate_clean_signal(duration=3.0, signal_type='speech')
    print(f"   Generated {len(clean_audio)/denoiser.sample_rate:.1f}s clean audio")

    # Test different noise types
    noise_types = ['white', 'hum', 'click']

    for noise_type in noise_types:
        print(f"\n2. Adding {noise_type} noise...")
        noisy_audio, noise = denoiser.add_noise(clean_audio, noise_type, snr_db=10)

        original_snr = calculate_snr(clean_audio, noisy_audio)
        print(f"   Original SNR: {original_snr:.2f} dB")

        # Apply denoising
        print(f"\n3. Applying denoising for {noise_type} noise...")

        if noise_type == 'white' or noise_type == 'pink':
            denoised = denoiser.wiener_filter(noisy_audio)
        elif noise_type == 'hum':
            denoised = denoiser.notch_filter(noisy_audio)
        elif noise_type == 'click':
            denoised = denoiser.median_filter(noisy_audio)

        # Calculate improvement
        min_len = min(len(clean_audio), len(denoised))
        improved_snr = calculate_snr(clean_audio[:min_len], denoised[:min_len])
        improvement = improved_snr - original_snr

        print(f"   Denoised SNR: {improved_snr:.2f} dB")
        print(f"   Improvement: {improvement:.2f} dB")

        # Visualizations
        print(f"\n4. Creating visualizations for {noise_type} noise...")
        visualize_denoising(denoiser, clean_audio, noisy_audio, denoised, noise_type)

        # Compare methods
        print(f"\n5. Comparing denoising methods for {noise_type} noise...")
        results = compare_methods(denoiser, clean_audio, noisy_audio, noise_type)

        print("\n   Method comparison:")
        for method, improvement in results.items():
            print(f"   {method:20s}: {improvement:+.2f} dB")

        print("\n" + "-" * 60)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
