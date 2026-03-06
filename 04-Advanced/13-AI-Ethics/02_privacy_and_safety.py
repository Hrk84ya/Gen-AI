"""
Privacy & Safety in AI
========================
Covers: Differential Privacy (DP-SGD), membership inference attacks,
toxicity filtering, and safety guardrails.
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. Differential Privacy — DP-SGD (Abadi et al. 2016)
# ---------------------------------------------------------------------------

class DPSGD:
    """
    Differentially-Private Stochastic Gradient Descent.

    Two key operations per mini-batch:
        1. Clip each per-sample gradient to max_norm
        2. Add Gaussian noise scaled to (noise_multiplier × max_norm)

    The (ε, δ)-privacy guarantee depends on noise_multiplier,
    max_norm, number of steps, and dataset size.
    """

    def __init__(self, model: nn.Module, lr: float = 0.01,
                 max_norm: float = 1.0, noise_multiplier: float = 1.0):
        self.model = model
        self.lr = lr
        self.max_norm = max_norm
        self.noise_multiplier = noise_multiplier

    def _clip_and_noise(self, grads: torch.Tensor,
                        batch_size: int) -> torch.Tensor:
        """Clip per-sample gradient norms and add Gaussian noise."""
        # grads: (batch_size, param_dim) — one gradient vector per sample
        norms = grads.norm(dim=1, keepdim=True).clamp(min=1e-8)
        clip_factor = (self.max_norm / norms).clamp(max=1.0)
        clipped = grads * clip_factor

        # Sum clipped gradients
        summed = clipped.sum(dim=0)

        # Add noise
        noise = torch.randn_like(summed) * self.noise_multiplier * self.max_norm
        return (summed + noise) / batch_size

    def step(self, X: torch.Tensor, y: torch.Tensor):
        """
        One DP-SGD step.

        For simplicity we compute per-sample gradients by looping.
        (In production, use torch.vmap or Opacus.)
        """
        batch_size = X.size(0)
        param_shapes = [p.shape for p in self.model.parameters()]
        total_params = sum(p.numel() for p in self.model.parameters())

        per_sample_grads = []
        for i in range(batch_size):
            self.model.zero_grad()
            loss = F.cross_entropy(self.model(X[i:i+1]), y[i:i+1])
            loss.backward()
            grad_vec = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
            per_sample_grads.append(grad_vec)

        grads = torch.stack(per_sample_grads)  # (B, P)
        noisy_grad = self._clip_and_noise(grads, batch_size)

        # Apply to parameters
        offset = 0
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                p -= self.lr * noisy_grad[offset:offset + numel].view_as(p)
                offset += numel


# ---------------------------------------------------------------------------
# 2. Membership Inference Attack — Shokri et al. 2017
# ---------------------------------------------------------------------------

class MembershipInferenceAttack:
    """
    Simple threshold-based membership inference.

    Intuition: models tend to be more confident (lower loss) on
    training samples than on unseen samples.  An attacker trains
    a classifier on (loss, confidence) features to distinguish
    members from non-members.

    This is a privacy audit tool — use it to test your own models.
    """

    def __init__(self, target_model: nn.Module):
        self.target_model = target_model
        self.attack_model = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _extract_features(self, X: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor:
        """Extract (loss, max_confidence) per sample."""
        self.target_model.eval()
        with torch.no_grad():
            logits = self.target_model(X)
            probs = F.softmax(logits, dim=-1)
            losses = F.cross_entropy(logits, y, reduction="none")
            max_conf = probs.max(dim=1).values
        return torch.stack([losses, max_conf], dim=1)

    def train_attack(self, X_member: torch.Tensor, y_member: torch.Tensor,
                     X_nonmember: torch.Tensor, y_nonmember: torch.Tensor,
                     epochs: int = 200):
        """Train the attack model on member vs non-member features."""
        feat_m = self._extract_features(X_member, y_member)
        feat_n = self._extract_features(X_nonmember, y_nonmember)
        X_attack = torch.cat([feat_m, feat_n])
        y_attack = torch.cat([torch.ones(len(feat_m)),
                              torch.zeros(len(feat_n))])

        opt = optim.Adam(self.attack_model.parameters(), lr=0.01)
        for _ in range(epochs):
            logits = self.attack_model(X_attack).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, y_attack)
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def attack(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Predict membership probability for each sample."""
        feats = self._extract_features(X, y)
        return torch.sigmoid(self.attack_model(feats).squeeze()).numpy()


# ---------------------------------------------------------------------------
# 3. Toxicity Filter (Keyword + Learned)
# ---------------------------------------------------------------------------

class ToxicityFilter:
    """
    Two-stage toxicity filter:
        1. Fast keyword blocklist check
        2. Learned classifier on text embeddings (simulated here)

    In production, replace the learned classifier with a fine-tuned
    language model (e.g., Perspective API, OpenAI moderation endpoint).
    """

    def __init__(self, blocklist: list = None):
        self.blocklist = set(w.lower() for w in (blocklist or []))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def keyword_check(self, text: str) -> bool:
        """Returns True if text contains a blocked keyword."""
        tokens = set(text.lower().split())
        return bool(tokens & self.blocklist)

    @torch.no_grad()
    def classify_embedding(self, embedding: torch.Tensor) -> float:
        """Score an embedding for toxicity (0 = safe, 1 = toxic)."""
        self.classifier.eval()
        return torch.sigmoid(self.classifier(embedding)).item()

    def is_toxic(self, text: str,
                 embedding: torch.Tensor = None,
                 threshold: float = 0.5) -> bool:
        """Combined check: keyword OR classifier above threshold."""
        if self.keyword_check(text):
            return True
        if embedding is not None:
            return self.classify_embedding(embedding) > threshold
        return False


# ---------------------------------------------------------------------------
# 4. Safety Guardrails
# ---------------------------------------------------------------------------

class SafetyGuardrail:
    """
    Configurable guardrail that wraps a model's output.

    Checks:
        - Output confidence thresholding (refuse if uncertain)
        - Input validation (reject out-of-distribution inputs)
        - Rate limiting (simple counter-based)

    Returns a structured response with the decision and reasoning.
    """

    def __init__(self, model: nn.Module, confidence_threshold: float = 0.7,
                 max_input_norm: float = 10.0, max_calls: int = 1000):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_input_norm = max_input_norm
        self.max_calls = max_calls
        self.call_count = 0

    def __call__(self, x: torch.Tensor) -> dict:
        self.call_count += 1

        # Rate limit
        if self.call_count > self.max_calls:
            return {"allowed": False, "reason": "rate_limit_exceeded",
                    "prediction": None}

        # Input validation
        input_norm = x.norm().item()
        if input_norm > self.max_input_norm:
            return {"allowed": False, "reason": "input_out_of_distribution",
                    "input_norm": input_norm, "prediction": None}

        # Run model
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_class = probs.max(dim=-1)

        # Confidence check
        if confidence.item() < self.confidence_threshold:
            return {"allowed": False, "reason": "low_confidence",
                    "confidence": round(confidence.item(), 4),
                    "prediction": pred_class.item()}

        return {"allowed": True, "prediction": pred_class.item(),
                "confidence": round(confidence.item(), 4)}


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    N, D, C = 400, 8, 3
    X = torch.randn(N, D)
    y = torch.randint(0, C, (N,))

    # --- DP-SGD ---
    print("=== DP-SGD Training ===")
    model_dp = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
    dp_opt = DPSGD(model_dp, lr=0.05, max_norm=1.0, noise_multiplier=0.8)
    for epoch in range(20):
        # Mini-batch
        idx = torch.randperm(N)[:32]
        dp_opt.step(X[idx], y[idx])
    with torch.no_grad():
        acc = (model_dp(X).argmax(1) == y).float().mean()
    print(f"DP model accuracy: {acc:.3f}")

    # --- Membership Inference ---
    print("\n=== Membership Inference Attack ===")
    target = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, C))
    opt = optim.Adam(target.parameters(), lr=0.01)
    # Overfit on first half (members)
    for _ in range(300):
        loss = F.cross_entropy(target(X[:200]), y[:200])
        opt.zero_grad(); loss.backward(); opt.step()

    attack = MembershipInferenceAttack(target)
    attack.train_attack(X[:200], y[:200], X[200:], y[200:])
    member_scores = attack.attack(X[:50], y[:50])
    nonmember_scores = attack.attack(X[200:250], y[200:250])
    print(f"Mean member score:     {member_scores.mean():.3f}")
    print(f"Mean non-member score: {nonmember_scores.mean():.3f}")

    # --- Toxicity Filter ---
    print("\n=== Toxicity Filter ===")
    filt = ToxicityFilter(blocklist=["harmful", "dangerous"])
    print(f"'hello world' toxic? {filt.is_toxic('hello world')}")
    print(f"'this is harmful' toxic? {filt.is_toxic('this is harmful')}")

    # --- Safety Guardrail ---
    print("\n=== Safety Guardrail ===")
    guard = SafetyGuardrail(target, confidence_threshold=0.6,
                            max_input_norm=10.0)
    normal = guard(torch.randn(1, D))
    print(f"Normal input: {normal}")
    ood = guard(torch.randn(1, D) * 100)
    print(f"OOD input:    {ood}")
