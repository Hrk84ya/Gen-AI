# 🎙️ Speech & Audio Processing

## Overview

This module covers deep learning approaches to speech and audio, from
fundamental signal processing (spectrograms, MFCCs) through modern
neural architectures for speech recognition, synthesis, and audio
classification.

## Contents

| File | Topics |
|------|--------|
| `01_audio_fundamentals.py` | Waveform I/O, spectrograms, Mel-filterbanks, MFCCs, data augmentation (SpecAugment) |
| `02_speech_models.py` | Audio classification CNN, CTC-based speech recognition, simple TTS (Tacotron-style encoder) |

## Prerequisites

- PyTorch basics (Level 2)
- Sequence models / RNNs (Section 2.5)
- `torchaudio` (added to requirements.txt)

## Learning Objectives

1. Understand time-domain vs frequency-domain audio representations
2. Extract and visualize spectrograms and MFCCs
3. Build a CNN-based audio classifier
4. Implement CTC loss for speech recognition
5. Understand encoder-decoder TTS pipelines

## Key Concepts

- **Spectrogram**: 2-D time×frequency representation of audio via STFT
- **Mel scale**: Perceptually-motivated frequency warping
- **MFCC**: Compact spectral features widely used in speech systems
- **CTC loss**: Alignment-free loss for sequence-to-sequence speech recognition
- **SpecAugment**: Simple but effective audio data augmentation

## References

- Hannun et al., *Deep Speech* (2014)
- Park et al., *SpecAugment* (2019)
- Shen et al., *Tacotron 2* (2018)
