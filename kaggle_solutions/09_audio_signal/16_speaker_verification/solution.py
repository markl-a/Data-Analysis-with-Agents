"""
Speaker Recognition and Verification
=====================================

This solution demonstrates speaker recognition and verification using
acoustic features and statistical modeling.

Dataset: Synthetic speaker audio samples
Techniques:
- Speaker feature extraction (i-vectors, x-vectors concept)
- Gaussian Mixture Models (GMM)
- Cosine similarity scoring
- Equal Error Rate (EER) computation
- Speaker enrollment and verification
- ROC curves and performance metrics

Author: Data Science Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SpeakerVerifier:
    """
    Speaker verification system using GMM-UBM approach.
    """

    def __init__(self, sample_rate=16000, n_mfcc=13, n_components=16):
        """
        Initialize speaker verifier.
        
        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate
        n_mfcc : int
            Number of MFCC features
        n_components : int
            Number of GMM components
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_components = n_components
        self.ubm = None  # Universal Background Model
        self.speaker_models = {}

    def extract_mfcc(self, audio, n_fft=512, hop_length=256):
        """
        Extract MFCC features.
        
        Parameters:
        -----------
        audio : ndarray
            Input audio signal
            
        Returns:
        --------
        mfcc : ndarray
            MFCC features (n_frames x n_mfcc)
        """
        # Simple MFCC implementation
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        n_mels = 40
        
        # Create mel filterbank
        mel_filters = self._create_mel_filterbank(n_fft, n_mels)
        
        # Extract features
        mfcc_features = []
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft]
            
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            
            # Apply window
            windowed = frame * np.hanning(len(frame))
            
            # Compute spectrum
            spectrum = np.abs(rfft(windowed))
            
            # Apply mel filterbank
            mel_spec = np.dot(mel_filters, spectrum)
            
            # Log and DCT
            log_mel = np.log(mel_spec + 1e-10)
            from scipy.fftpack import dct
            mfcc = dct(log_mel, type=2, norm='ortho')[:self.n_mfcc]
            
            mfcc_features.append(mfcc)
        
        return np.array(mfcc_features)

    def _create_mel_filterbank(self, n_fft, n_mels):
        """Create mel filterbank."""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        bin_points = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        mel_filters = np.zeros((n_mels, n_fft // 2 + 1))
        
        for i in range(1, n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            for j in range(left, center):
                mel_filters[i - 1, j] = (j - left) / (center - left)
            
            for j in range(center, right):
                mel_filters[i - 1, j] = (right - j) / (right - center)
        
        return mel_filters

    def train_ubm(self, background_data):
        """
        Train Universal Background Model.
        
        Parameters:
        -----------
        background_data : list of ndarray
            List of audio samples for UBM training
        """
        print("   Training UBM...")
        all_features = []
        
        for audio in background_data:
            mfcc = self.extract_mfcc(audio)
            all_features.append(mfcc)
        
        all_features = np.vstack(all_features)
        
        # Train GMM
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=100,
            random_state=42
        )
        self.ubm.fit(all_features)
        print(f"   UBM trained with {len(all_features)} feature vectors")

    def enroll_speaker(self, speaker_id, enrollment_data):
        """
        Enroll a speaker by adapting UBM.
        
        Parameters:
        -----------
        speaker_id : str
            Speaker identifier
        enrollment_data : list of ndarray
            List of audio samples for enrollment
        """
        all_features = []
        
        for audio in enrollment_data:
            mfcc = self.extract_mfcc(audio)
            all_features.append(mfcc)
        
        all_features = np.vstack(all_features)
        
        # Train speaker-specific GMM
        speaker_gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=100,
            random_state=42
        )
        speaker_gmm.fit(all_features)
        
        self.speaker_models[speaker_id] = speaker_gmm
        print(f"   Enrolled speaker '{speaker_id}' with {len(all_features)} features")

    def verify_speaker(self, test_audio, claimed_speaker_id):
        """
        Verify if test audio belongs to claimed speaker.
        
        Parameters:
        -----------
        test_audio : ndarray
            Test audio sample
        claimed_speaker_id : str
            Claimed speaker identity
            
        Returns:
        --------
        score : float
            Verification score (log-likelihood ratio)
        """
        mfcc = self.extract_mfcc(test_audio)
        
        # Compute log-likelihood with speaker model
        speaker_ll = self.speaker_models[claimed_speaker_id].score(mfcc)
        
        # Compute log-likelihood with UBM
        ubm_ll = self.ubm.score(mfcc)
        
        # Log-likelihood ratio
        score = speaker_ll - ubm_ll
        
        return score


def generate_speaker_audio(speaker_id, num_samples=3, duration=2.0, sample_rate=16000):
    """
    Generate synthetic speaker audio with unique characteristics.
    
    Parameters:
    -----------
    speaker_id : int
        Speaker identifier
    num_samples : int
        Number of audio samples to generate
    duration : float
        Duration of each sample
    sample_rate : int
        Sample rate
        
    Returns:
    --------
    samples : list of ndarray
        List of audio samples
    """
    np.random.seed(speaker_id)
    samples = []
    
    # Each speaker has unique pitch and formant characteristics
    base_pitch = 100 + speaker_id * 20  # Hz
    formant_shift = speaker_id * 50  # Hz
    
    for _ in range(num_samples):
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Add pitch
        pitch = base_pitch + 10 * np.sin(2 * np.pi * 3 * t)
        audio += 1.5 * np.sin(2 * np.pi * pitch * t)
        
        # Add formants
        formants = [500 + formant_shift, 1500 + formant_shift, 2500 + formant_shift]
        for f in formants:
            audio += 0.5 * np.sin(2 * np.pi * f * t)
        
        # Add noise (speaker-specific)
        audio += 0.05 * np.random.randn(len(t))
        
        # Apply envelope
        envelope = np.exp(-t) * (1 - np.exp(-10 * t))
        audio *= envelope
        
        samples.append(audio / np.max(np.abs(audio)))
    
    return samples


def visualize_speaker_verification(verifier, save_path='speaker_verification.png'):
    """
    Visualize speaker verification results.
    """
    # Generate data for 3 speakers
    num_speakers = 3
    speakers_data = {}
    
    for i in range(num_speakers):
        speakers_data[f'Speaker_{i}'] = generate_speaker_audio(i, num_samples=5)
    
    # Background data
    background_data = []
    for i in range(5):
        background_data.extend(generate_speaker_audio(10 + i, num_samples=2))
    
    # Train UBM
    verifier.train_ubm(background_data)
    
    # Enroll speakers
    for speaker_id, data in speakers_data.items():
        verifier.enroll_speaker(speaker_id, data[:3])
    
    # Test verification
    genuine_scores = []
    impostor_scores = []
    
    for speaker_id, data in speakers_data.items():
        # Genuine attempts (same speaker)
        for test_sample in data[3:]:
            score = verifier.verify_speaker(test_sample, speaker_id)
            genuine_scores.append(score)
        
        # Impostor attempts (different speakers)
        for other_id, other_data in speakers_data.items():
            if other_id != speaker_id:
                for test_sample in other_data[3:]:
                    score = verifier.verify_speaker(test_sample, speaker_id)
                    impostor_scores.append(score)
    
    # Visualization
    fig = plt.figure(figsize=(14, 10))
    
    # Score distributions
    ax1 = plt.subplot(2, 2, 1)
    plt.hist(genuine_scores, bins=20, alpha=0.6, label='Genuine', color='green')
    plt.hist(impostor_scores, bins=20, alpha=0.6, label='Impostor', color='red')
    plt.xlabel('Verification Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ROC Curve
    ax2 = plt.subplot(2, 2, 2)
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DET Curve
    ax3 = plt.subplot(2, 2, 3)
    fnr = 1 - tpr
    plt.plot(fpr * 100, fnr * 100, linewidth=2, color='purple')
    
    # Find EER
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    plt.plot(eer * 100, eer * 100, 'ro', markersize=10, label=f'EER = {eer*100:.2f}%')
    
    plt.xlabel('False Acceptance Rate (%)')
    plt.ylabel('False Rejection Rate (%)')
    plt.title('DET Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    
    # Performance metrics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    metrics_text = f"""
    Speaker Verification Performance
    {'='*40}
    
    Number of Speakers: {num_speakers}
    
    Genuine Attempts: {len(genuine_scores)}
    Impostor Attempts: {len(impostor_scores)}
    
    Genuine Score: {np.mean(genuine_scores):.3f} ± {np.std(genuine_scores):.3f}
    Impostor Score: {np.mean(impostor_scores):.3f} ± {np.std(impostor_scores):.3f}
    
    Equal Error Rate (EER): {eer*100:.2f}%
    AUC: {roc_auc:.3f}
    
    At EER threshold ({thresholds[eer_idx]:.3f}):
      - FAR: {fpr[eer_idx]*100:.2f}%
      - FRR: {fnr[eer_idx]*100:.2f}%
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved speaker verification results to {save_path}")
    plt.close()
    
    return eer, roc_auc


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("Speaker Recognition and Verification")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Initialize verifier
    print("\n1. Initializing speaker verification system...")
    verifier = SpeakerVerifier(
        sample_rate=16000,
        n_mfcc=13,
        n_components=16
    )
    
    # Run verification
    print("\n2. Running speaker verification experiment...")
    eer, auc_score = visualize_speaker_verification(verifier)
    
    print("\n" + "=" * 70)
    print(f"EER: {eer*100:.2f}%")
    print(f"AUC: {auc_score:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
