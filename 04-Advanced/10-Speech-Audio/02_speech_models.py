"""
Speech & Audio Neural Models
===============================
Covers: Audio classification CNN, CTC-based speech recognition,
and a simplified Tacotron-style TTS encoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. Audio Classification CNN
# ---------------------------------------------------------------------------

class AudioClassifier(nn.Module):
    """
    Simple CNN that classifies Mel spectrogram frames.

    Input:  (batch, 1, n_mels, time_frames)  — like a 1-channel image
    Output: (batch, num_classes)

    Architecture mirrors a small VGG-style net:
        Conv→BN→ReLU→Pool  ×3  →  AdaptivePool → FC
    """

    def __init__(self, n_mels: int = 64, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# 2. CTC-Based Speech Recognition (Simplified)
# ---------------------------------------------------------------------------

class CTCSpeechRecognizer(nn.Module):
    """
    Minimal CTC speech recognizer.

    Pipeline:
        Mel spectrogram → Conv feature extractor → BiLSTM → Linear → CTC loss

    CTC (Connectionist Temporal Classification) lets us train without
    explicit frame-level alignment between audio and text.
    """

    def __init__(self, n_mels: int = 64, hidden: int = 128,
                 vocab_size: int = 28):
        """
        Args:
            vocab_size: number of output tokens (26 letters + space + blank)
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, time)
        Returns:
            log_probs: (time, batch, vocab_size) — ready for CTC loss
        """
        x = self.conv(x)                    # (B, hidden, T)
        x = x.permute(0, 2, 1)              # (B, T, hidden)
        x, _ = self.rnn(x)                  # (B, T, hidden*2)
        x = self.fc(x)                      # (B, T, vocab)
        return F.log_softmax(x, dim=-1).permute(1, 0, 2)  # (T, B, vocab)


# ---------------------------------------------------------------------------
# 3. Simplified Tacotron-Style TTS Encoder
# ---------------------------------------------------------------------------

class TTSEncoder(nn.Module):
    """
    Text encoder inspired by Tacotron 2.

    Converts a sequence of character embeddings into a context
    representation that a decoder would use to produce Mel frames.

    Pipeline:
        Embedding → 3×Conv1D → BiLSTM → encoder outputs
    """

    def __init__(self, vocab_size: int = 28, embed_dim: int = 64,
                 hidden: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.rnn = nn.LSTM(hidden, hidden // 2, batch_first=True,
                           bidirectional=True)

    def forward(self, text_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_indices: (batch, seq_len) — integer character indices
        Returns:
            encoder_out: (batch, seq_len, hidden)
        """
        x = self.embedding(text_indices)       # (B, S, embed)
        x = self.convs(x.permute(0, 2, 1))    # (B, hidden, S)
        x = x.permute(0, 2, 1)                # (B, S, hidden)
        x, _ = self.rnn(x)                    # (B, S, hidden)
        return x


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Audio Classification ---
    print("=== Audio Classifier ===")
    clf = AudioClassifier(n_mels=64, num_classes=10)
    dummy_mel = torch.randn(4, 1, 64, 100)  # batch of 4 spectrograms
    logits = clf(dummy_mel)
    print(f"Input: {dummy_mel.shape} → Output: {logits.shape}")

    # --- CTC Speech Recognition ---
    print("\n=== CTC Speech Recognizer ===")
    asr = CTCSpeechRecognizer(n_mels=64, hidden=128, vocab_size=28)
    mel_input = torch.randn(2, 64, 150)  # batch=2, 64 mels, 150 frames
    log_probs = asr(mel_input)
    print(f"Input: {mel_input.shape} → Log-probs: {log_probs.shape}")

    # Quick CTC loss demo
    input_lengths = torch.full((2,), log_probs.size(0), dtype=torch.long)
    targets = torch.randint(1, 28, (2, 10))  # random target sequences
    target_lengths = torch.full((2,), 10, dtype=torch.long)
    ctc_loss = nn.CTCLoss(blank=0)
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss.item():.4f}")

    # --- TTS Encoder ---
    print("\n=== TTS Encoder ===")
    tts_enc = TTSEncoder(vocab_size=28, embed_dim=64, hidden=128)
    text = torch.randint(0, 28, (2, 20))  # batch=2, 20 characters
    enc_out = tts_enc(text)
    print(f"Input: {text.shape} → Encoder output: {enc_out.shape}")
