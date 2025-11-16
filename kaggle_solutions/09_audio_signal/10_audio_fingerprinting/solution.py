"""
Audio Fingerprinting System
============================

This solution demonstrates audio fingerprinting for music identification.
We generate unique audio fingerprints that can identify songs even with
noise and distortion.

Author: Kaggle Solutions Team
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AudioFingerprinter:
    """Audio fingerprinting for song identification"""

    def __init__(self, sample_rate=11025):
        self.sample_rate = sample_rate
        self.database = {}  # song_id -> list of fingerprints

    def generate_test_song(self, song_id, duration=10.0):
        """Generate a unique test song"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)

        # Each song has unique frequencies based on its ID
        np.random.seed(song_id)
        base_freq = 200 + song_id * 50

        # Create unique harmonic structure
        for i in range(6):
            freq = base_freq * (i + 1) + np.random.uniform(-10, 10)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = 1 / (i + 1)
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)

        # Add rhythm pattern unique to song
        rhythm_freq = 2 + song_id * 0.3
        rhythm = 1 + 0.5 * signal.square(2 * np.pi * rhythm_freq * t)
        audio *= rhythm

        # Add melodic variation
        melody_freq = 0.5
        melody = 1 + 0.3 * np.sin(2 * np.pi * melody_freq * t)
        audio *= melody

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio

    def compute_spectrogram(self, audio, window_size=512, overlap=256):
        """Compute spectrogram of audio"""
        noverlap = window_size - overlap
        f, t, Sxx = signal.spectrogram(audio, self.sample_rate,
                                       nperseg=window_size,
                                       noverlap=noverlap)
        return f, t, Sxx

    def find_peaks(self, Sxx, f, t):
        """Find spectral peaks in spectrogram"""
        peaks = []

        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Find peaks in each time frame
        for time_idx in range(Sxx.shape[1]):
            spectrum = Sxx_db[:, time_idx]

            # Find local maxima
            peak_indices, properties = signal.find_peaks(
                spectrum,
                height=np.max(spectrum) - 30,  # Within 30dB of max
                distance=5  # At least 5 bins apart
            )

            # Store peak information (time, frequency, magnitude)
            for peak_idx in peak_indices:
                peaks.append({
                    'time': t[time_idx],
                    'freq': f[peak_idx],
                    'mag': spectrum[peak_idx]
                })

        return peaks

    def generate_fingerprints(self, peaks, fan_value=5):
        """Generate constellation map fingerprints from peaks"""
        fingerprints = []

        # Sort peaks by time
        peaks_sorted = sorted(peaks, key=lambda x: x['time'])

        # Create fingerprints using peak pairs (constellation map)
        for i, peak1 in enumerate(peaks_sorted):
            # Look at next few peaks (fan_value)
            for j in range(i + 1, min(i + fan_value + 1, len(peaks_sorted))):
                peak2 = peaks_sorted[j]

                # Time difference
                time_delta = peak2['time'] - peak1['time']

                # Only consider peaks within reasonable time window
                if 0 < time_delta < 0.5:  # Within 0.5 seconds
                    # Create hash from peak pair
                    freq1 = int(peak1['freq'])
                    freq2 = int(peak2['freq'])
                    time_diff = int(time_delta * 1000)  # milliseconds

                    # Fingerprint: (freq1, freq2, time_delta) -> absolute_time
                    fingerprint_hash = f"{freq1}_{freq2}_{time_diff}"
                    fingerprint = {
                        'hash': fingerprint_hash,
                        'time_offset': peak1['time']
                    }
                    fingerprints.append(fingerprint)

        return fingerprints

    def add_to_database(self, song_id, audio):
        """Add song to fingerprint database"""
        # Compute spectrogram
        f, t, Sxx = self.compute_spectrogram(audio)

        # Find peaks
        peaks = self.find_peaks(Sxx, f, t)

        # Generate fingerprints
        fingerprints = self.generate_fingerprints(peaks)

        # Store in database
        self.database[song_id] = fingerprints

        return len(fingerprints)

    def match(self, audio_sample):
        """Match audio sample against database"""
        # Generate fingerprints for sample
        f, t, Sxx = self.compute_spectrogram(audio_sample)
        peaks = self.find_peaks(Sxx, f, t)
        sample_fingerprints = self.generate_fingerprints(peaks)

        # Match against database
        matches = {song_id: [] for song_id in self.database.keys()}

        for sample_fp in sample_fingerprints:
            sample_hash = sample_fp['hash']
            sample_time = sample_fp['time_offset']

            # Check each song in database
            for song_id, song_fingerprints in self.database.items():
                for song_fp in song_fingerprints:
                    if song_fp['hash'] == sample_hash:
                        # Calculate time offset difference
                        time_diff = sample_time - song_fp['time_offset']
                        matches[song_id].append(time_diff)

        # Find song with most consistent matches
        best_match = None
        best_score = 0

        for song_id, time_diffs in matches.items():
            if len(time_diffs) == 0:
                continue

            # Cluster time differences (songs should have consistent offset)
            time_diffs = np.array(time_diffs)
            # Find mode (most common offset)
            hist, bins = np.histogram(time_diffs, bins=50)
            max_count = np.max(hist)

            if max_count > best_score:
                best_score = max_count
                best_match = song_id

        return best_match, best_score

    def add_noise(self, audio, snr_db=20):
        """Add noise to audio"""
        signal_power = np.mean(audio**2)
        noise_power = signal_power / (10**(snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise


def visualize_fingerprinting(fingerprinter, audio, song_id):
    """Visualize audio fingerprinting process"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Waveform
    t = np.linspace(0, len(audio)/fingerprinter.sample_rate, len(audio))
    axes[0].plot(t, audio, linewidth=0.5, color='blue', alpha=0.7)
    axes[0].set_title(f'Song {song_id} - Waveform', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Spectrogram
    f, t_spec, Sxx = fingerprinter.compute_spectrogram(audio)
    axes[1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                      shading='gouraud', cmap='viridis')
    axes[1].set_title('Spectrogram', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim([0, 3000])

    # Constellation map (peaks)
    peaks = fingerprinter.find_peaks(Sxx, f, t_spec)
    peak_times = [p['time'] for p in peaks]
    peak_freqs = [p['freq'] for p in peaks]

    axes[2].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                      shading='gouraud', cmap='gray', alpha=0.3)
    axes[2].scatter(peak_times, peak_freqs, c='red', s=10, alpha=0.6, label='Peaks')
    axes[2].set_title('Constellation Map (Fingerprint Peaks)', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_ylim([0, 3000])
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'audio_fingerprint_song_{song_id}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: audio_fingerprint_song_{song_id}.png")
    plt.close()


def visualize_matching_results(results):
    """Visualize fingerprint matching results"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Accuracy for different noise levels
    noise_levels = sorted(results.keys())
    accuracies = [results[snr]['accuracy'] for snr in noise_levels]

    axes[0].plot(noise_levels, accuracies, marker='o', linewidth=2,
                markersize=8, color='steelblue')
    axes[0].set_title('Fingerprinting Accuracy vs Noise Level', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[0].legend()

    # Match confidence scores
    snr = 20  # Example SNR
    if snr in results:
        songs = results[snr]['songs']
        confidences = results[snr]['confidences']

        colors = ['green' if results[snr]['correct'][i] else 'red'
                 for i in range(len(songs))]

        axes[1].bar(range(len(songs)), confidences, color=colors, alpha=0.7)
        axes[1].set_title(f'Match Confidence (SNR={snr}dB)', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Test Sample', fontsize=12)
        axes[1].set_ylabel('Number of Matching Fingerprints', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Correct Match'),
            Patch(facecolor='red', alpha=0.7, label='Incorrect Match')
        ]
        axes[1].legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('fingerprint_matching_results.png', dpi=300, bbox_inches='tight')
    print("Saved: fingerprint_matching_results.png")
    plt.close()


def main():
    """Main execution function"""
    print("=" * 60)
    print("Audio Fingerprinting System")
    print("=" * 60)

    # Initialize fingerprinter
    fingerprinter = AudioFingerprinter(sample_rate=11025)

    # Generate database of songs
    print("\n1. Generating song database...")
    n_songs = 10
    song_database = {}

    for song_id in range(n_songs):
        audio = fingerprinter.generate_test_song(song_id, duration=10.0)
        song_database[song_id] = audio
        n_fingerprints = fingerprinter.add_to_database(song_id, audio)
        print(f"   Song {song_id}: {n_fingerprints} fingerprints generated")

    print(f"\n   Total songs in database: {len(fingerprinter.database)}")

    # Visualize fingerprinting for one song
    print("\n2. Visualizing fingerprinting process...")
    visualize_fingerprinting(fingerprinter, song_database[0], 0)

    # Test matching with different noise levels
    print("\n3. Testing fingerprint matching...")
    noise_levels = [40, 30, 20, 15, 10, 5]  # SNR in dB
    results = {}

    for snr in noise_levels:
        print(f"\n   Testing with SNR = {snr} dB...")
        correct = 0
        total = n_songs

        songs_tested = []
        confidences_list = []
        correct_list = []

        for song_id in range(n_songs):
            # Take a sample from the song
            full_audio = song_database[song_id]
            sample_start = int(len(full_audio) * 0.3)  # Start at 30%
            sample_length = int(fingerprinter.sample_rate * 3)  # 3 seconds
            sample = full_audio[sample_start:sample_start+sample_length]

            # Add noise
            noisy_sample = fingerprinter.add_noise(sample, snr_db=snr)

            # Match
            matched_id, confidence = fingerprinter.match(noisy_sample)

            songs_tested.append(song_id)
            confidences_list.append(confidence)
            correct_list.append(matched_id == song_id)

            if matched_id == song_id:
                correct += 1
                print(f"      Song {song_id}: Matched correctly (confidence: {confidence})")
            else:
                print(f"      Song {song_id}: Matched as {matched_id} (confidence: {confidence})")

        accuracy = correct / total
        print(f"\n   Accuracy at SNR {snr}dB: {accuracy:.2%} ({correct}/{total})")

        results[snr] = {
            'accuracy': accuracy,
            'songs': songs_tested,
            'confidences': confidences_list,
            'correct': correct_list
        }

    # Visualize results
    print("\n4. Creating result visualizations...")
    visualize_matching_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    for snr in sorted(results.keys(), reverse=True):
        acc = results[snr]['accuracy']
        print(f"SNR {snr:3d} dB: {acc:6.1%} accuracy")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    # Technical details
    print("\nTechnical Details:")
    print(f"- Sample rate: {fingerprinter.sample_rate} Hz")
    print(f"- Songs in database: {len(fingerprinter.database)}")
    avg_fingerprints = np.mean([len(fps) for fps in fingerprinter.database.values()])
    print(f"- Average fingerprints per song: {avg_fingerprints:.0f}")
    print(f"- Test sample duration: 3 seconds")
    print(f"- Fingerprinting method: Constellation map")


if __name__ == "__main__":
    main()
