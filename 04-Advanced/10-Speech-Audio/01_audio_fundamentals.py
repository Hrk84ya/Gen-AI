"""
Audio & Speech Fundamentals
=============================
Covers: Waveform handling, STFT spectrograms, Mel spectrograms,
MFCCs, and SpecAugment data augmentation.

All core transforms are implemented from scratch in NumPy/PyTorch
so you understand what torchaudio does under the hood.
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Waveform Utilities
# ---------------------------------------------------------------------------

def generate_sine_wave(freq: float = 440.0, duration: float = 1.0,
                       sample_rate: int = 16000) -> np.ndarray:
    """Generate a pure sine tone (useful for testing)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_composite_wave(freqs: list, duration: float = 1.0,
                            sample_rate: int = 16000) -> np.ndarray:
    """Sum of multiple sine waves — simulates a richer sound."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (wave / len(freqs)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Short-Time Fourier Transform (STFT)
# ---------------------------------------------------------------------------

def stft(signal: np.ndarray, n_fft: int = 512, hop_length: int = 160,
         window: str = "hann") -> np.ndarray:
    """
    Compute the magnitude spectrogram via STFT.

    Args:
        signal: 1-D waveform array
        n_fft: FFT window size
        hop_length: stride between frames
        window: window function name

    Returns:
        magnitude spectrogram of shape (n_fft//2 + 1, n_frames)
    """
    if window == "hann":
        win = np.hanning(n_fft).astype(np.float32)
    else:
        win = np.ones(n_fft, dtype=np.float32)

    # Pad signal so we don't lose the tail
    pad_len = n_fft // 2
    padded = np.pad(signal, (pad_len, pad_len), mode="reflect")

    n_frames = 1 + (len(padded) - n_fft) // hop_length
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start:start + n_fft] * win
        fft_out = np.fft.rfft(frame)
        spec[:, i] = np.abs(fft_out)

    return spec


# ---------------------------------------------------------------------------
# 3. Mel Spectrogram
# ---------------------------------------------------------------------------

def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = 64, n_fft: int = 512,
                   sample_rate: int = 16000) -> np.ndarray:
    """
    Build a Mel-scale triangular filterbank matrix.

    Returns:
        filters: (n_mels, n_fft//2 + 1)
    """
    n_freqs = n_fft // 2 + 1
    low_mel, high_mel = hz_to_mel(0), hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            if center != left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filters[i, j] = (right - j) / (right - center)
    return filters


def mel_spectrogram(signal: np.ndarray, n_fft: int = 512,
                    hop_length: int = 160, n_mels: int = 64,
                    sample_rate: int = 16000) -> np.ndarray:
    """
    Compute a log-Mel spectrogram.

    Returns:
        log_mel: (n_mels, n_frames)
    """
    spec = stft(signal, n_fft=n_fft, hop_length=hop_length)
    fb = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate)
    mel = fb @ (spec ** 2)  # power spectrogram through filterbank
    log_mel = np.log(mel + 1e-9)
    return log_mel


# ---------------------------------------------------------------------------
# 4. MFCCs (Mel-Frequency Cepstral Coefficients)
# ---------------------------------------------------------------------------

def mfcc(signal: np.ndarray, n_mfcc: int = 13, n_mels: int = 64,
         n_fft: int = 512, hop_length: int = 160,
         sample_rate: int = 16000) -> np.ndarray:
    """
    Compute MFCCs via DCT of the log-Mel spectrogram.

    Returns:
        coeffs: (n_mfcc, n_frames)
    """
    log_mel = mel_spectrogram(signal, n_fft, hop_length, n_mels, sample_rate)
    # Type-II DCT (scipy-free implementation)
    n = log_mel.shape[0]
    dct_matrix = np.zeros((n_mfcc, n), dtype=np.float32)
    for k in range(n_mfcc):
        for i in range(n):
            dct_matrix[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    coeffs = dct_matrix @ log_mel
    return coeffs


# ---------------------------------------------------------------------------
# 5. SpecAugment — Park et al. 2019
# ---------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """
    SpecAugment: simple spectrogram augmentation.

    Applies:
        1. Frequency masking — zero out a random band of Mel bins
        2. Time masking — zero out a random span of time frames

    Applied only during training.
    """

    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 20,
                 n_freq_masks: int = 1, n_time_masks: int = 1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (batch, n_mels, time) or (n_mels, time)
        """
        if not self.training:
            return spec
        cloned = spec.clone()
        squeezed = cloned.dim() == 2
        if squeezed:
            cloned = cloned.unsqueeze(0)

        _, n_mels, n_time = cloned.shape

        for _ in range(self.n_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(n_mels - f, 1), (1,)).item()
            cloned[:, f0:f0 + f, :] = 0

        for _ in range(self.n_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(n_time - t, 1), (1,)).item()
            cloned[:, :, t0:t0 + t] = 0

        return cloned.squeeze(0) if squeezed else cloned


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sr = 16000

    # Generate a test signal (chord: 261 Hz + 329 Hz + 392 Hz ≈ C major)
    wave = generate_composite_wave([261.63, 329.63, 392.00], duration=1.0,
                                   sample_rate=sr)
    print(f"Waveform: {wave.shape[0]} samples @ {sr} Hz")

    # STFT spectrogram
    spec = stft(wave, n_fft=512, hop_length=160)
    print(f"STFT spectrogram: {spec.shape}  (freq_bins × frames)")

    # Mel spectrogram
    log_mel = mel_spectrogram(wave, n_fft=512, hop_length=160, n_mels=64,
                              sample_rate=sr)
    print(f"Log-Mel spectrogram: {log_mel.shape}  (n_mels × frames)")

    # MFCCs
    coeffs = mfcc(wave, n_mfcc=13, n_mels=64, sample_rate=sr)
    print(f"MFCCs: {coeffs.shape}  (n_mfcc × frames)")

    # SpecAugment
    aug = SpecAugment(freq_mask_param=8, time_mask_param=20)
    aug.train()
    mel_tensor = torch.tensor(log_mel)
    augmented = aug(mel_tensor)
    n_masked = (augmented == 0).sum().item()
    print(f"SpecAugment: {n_masked} values masked out of {augmented.numel()}")
