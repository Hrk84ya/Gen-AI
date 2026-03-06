"""
Advanced Bayesian Deep Learning
=================================
Covers: Deep Ensembles, SWAG (Stochastic Weight Averaging Gaussian),
Evidential Deep Learning, and uncertainty-aware active learning.
"""

import copy
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. Deep Ensembles — Lakshminarayanan et al. 2017
# ---------------------------------------------------------------------------

class DeepEnsemble:
    """
    Train M independent networks and average their predictions.

    Ensemble disagreement is a strong signal of epistemic uncertainty
    without any Bayesian machinery.
    """

    def __init__(self, base_model_fn, n_models: int = 5):
        """
        Args:
            base_model_fn: callable that returns a fresh nn.Module
            n_models: ensemble size
        """
        self.models = [base_model_fn() for _ in range(n_models)]

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 200,
            lr: float = 0.01):
        for i, model in enumerate(self.models):
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                model.train()
                loss = F.cross_entropy(model(X), y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_probs: (N, C) averaged softmax
            uncertainty: (N, C) variance across ensemble members
        """
        preds = []
        for m in self.models:
            m.eval()
            preds.append(F.softmax(m(X), dim=-1))
        preds = torch.stack(preds)  # (M, N, C)
        return preds.mean(0), preds.var(0)


# ---------------------------------------------------------------------------
# 2. SWAG — Maddox et al. 2019
# ---------------------------------------------------------------------------

class SWAG:
    """
    Stochastic Weight Averaging Gaussian.

    After standard training, continue for K epochs collecting weight
    snapshots. Fit a low-rank + diagonal Gaussian to the weight
    trajectory. Sample from this Gaussian at test time for uncertainty.

    Steps:
        1. Train model normally
        2. Continue training, collecting weight snapshots every C steps
        3. Compute running mean and second moment of weights
        4. At test time, sample weights from the fitted Gaussian
    """

    def __init__(self, model: nn.Module, max_rank: int = 20):
        self.model = model
        self.max_rank = max_rank
        self.mean = None
        self.sq_mean = None
        self.deviations: List[torch.Tensor] = []
        self.n_snapshots = 0

    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def _load_params(self, flat: torch.Tensor):
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view_as(p))
            offset += numel

    def collect(self):
        """Collect a weight snapshot (call after each SWA epoch)."""
        flat = self._flatten_params()
        if self.mean is None:
            self.mean = flat.clone()
            self.sq_mean = flat ** 2
        else:
            self.mean = (self.n_snapshots * self.mean + flat) / (self.n_snapshots + 1)
            self.sq_mean = (self.n_snapshots * self.sq_mean + flat ** 2) / (self.n_snapshots + 1)

        # Store low-rank deviation
        if len(self.deviations) < self.max_rank:
            self.deviations.append(flat - self.mean)
        self.n_snapshots += 1

    def sample_and_load(self):
        """Sample weights from the SWAG posterior and load into model."""
        diag_var = (self.sq_mean - self.mean ** 2).clamp(min=1e-8)
        z1 = torch.randn_like(self.mean)
        sample = self.mean + torch.sqrt(diag_var) * z1 * (1.0 / np.sqrt(2))

        if self.deviations:
            D = torch.stack(self.deviations)  # (K, P)
            z2 = torch.randn(D.size(0))
            sample += (D.T @ z2) * (1.0 / np.sqrt(2 * (D.size(0) - 1)))

        self._load_params(sample)

    @torch.no_grad()
    def predict(self, X: torch.Tensor, n_samples: int = 30
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = []
        for _ in range(n_samples):
            self.sample_and_load()
            self.model.eval()
            preds.append(F.softmax(self.model(X), dim=-1))
        # Restore mean weights
        self._load_params(self.mean)
        preds = torch.stack(preds)
        return preds.mean(0), preds.var(0)


# ---------------------------------------------------------------------------
# 3. Evidential Deep Learning — Sensoy et al. 2018
# ---------------------------------------------------------------------------

class EvidentialClassifier(nn.Module):
    """
    Instead of softmax probabilities, output Dirichlet concentration
    parameters α. Uncertainty comes from the Dirichlet distribution.

    Loss = Bayes risk of cross-entropy under Dir(α) + KL regularizer.

    High total evidence (Σα) → confident.
    Low total evidence → uncertain (close to uniform Dirichlet).
    """

    def __init__(self, in_features: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Dirichlet concentration α = softplus(logits) + 1."""
        return F.softplus(self.net(x)) + 1.0

    def expected_probability(self, alpha: torch.Tensor) -> torch.Tensor:
        """E[p] under Dir(α) = α / S, where S = Σα."""
        S = alpha.sum(dim=-1, keepdim=True)
        return alpha / S

    def uncertainty(self, alpha: torch.Tensor) -> torch.Tensor:
        """Epistemic uncertainty = C / S (number of classes / total evidence)."""
        S = alpha.sum(dim=-1)
        return self.num_classes / S

    @staticmethod
    def evidential_loss(alpha: torch.Tensor, y: torch.Tensor,
                        kl_weight: float = 0.01) -> torch.Tensor:
        """
        Bayes risk of cross-entropy + KL(Dir(α̃) || Dir(1)).
        """
        S = alpha.sum(dim=-1, keepdim=True)
        C = alpha.size(-1)
        one_hot = F.one_hot(y, C).float()

        # Bayes risk term
        loss = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        # KL regularizer (remove evidence for correct class)
        alpha_tilde = alpha * (1 - one_hot) + one_hot  # set correct class α to 1
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
        kl = (torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(C)))
              - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
              + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde)
                 - torch.digamma(S_tilde))).sum(dim=-1, keepdim=True))

        return (loss + kl_weight * kl.squeeze()).mean()


# ---------------------------------------------------------------------------
# 4. Uncertainty-Aware Active Learning
# ---------------------------------------------------------------------------

class UncertaintyActiveLearner:
    """
    Pool-based active learning that selects the most uncertain
    samples for labeling.

    Strategies:
        - max_entropy: pick samples with highest predictive entropy
        - bald: Bayesian Active Learning by Disagreement
    """

    def __init__(self, model: nn.Module, strategy: str = "max_entropy"):
        self.model = model
        self.strategy = strategy

    @torch.no_grad()
    def score(self, X: torch.Tensor, n_mc: int = 30) -> np.ndarray:
        """
        Score each sample by uncertainty.

        Returns:
            scores: (N,) — higher means more uncertain
        """
        self.model.train()  # MC dropout
        preds = torch.stack([F.softmax(self.model(X), dim=-1)
                             for _ in range(n_mc)])  # (T, N, C)

        if self.strategy == "max_entropy":
            mean_p = preds.mean(0)
            entropy = -(mean_p * torch.log(mean_p + 1e-8)).sum(dim=-1)
            return entropy.numpy()
        elif self.strategy == "bald":
            # BALD = H[y|x] - E_w[H[y|x,w]]
            mean_p = preds.mean(0)
            total_entropy = -(mean_p * torch.log(mean_p + 1e-8)).sum(dim=-1)
            per_sample_entropy = -(preds * torch.log(preds + 1e-8)).sum(dim=-1)
            expected_entropy = per_sample_entropy.mean(0)
            return (total_entropy - expected_entropy).numpy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def select(self, X: torch.Tensor, n_select: int = 10,
               n_mc: int = 30) -> np.ndarray:
        """Return indices of the n_select most uncertain samples."""
        scores = self.score(X, n_mc)
        return np.argsort(scores)[-n_select:][::-1]


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    N, D, C = 500, 10, 3
    X = torch.randn(N, D)
    W_true = torch.randn(D, C)
    y = (X @ W_true + 0.3 * torch.randn(N, C)).argmax(dim=1)

    def make_mlp():
        return nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, C),
        )

    # --- Deep Ensemble ---
    print("=== Deep Ensemble ===")
    ens = DeepEnsemble(make_mlp, n_models=3)
    ens.fit(X, y, epochs=150)
    mean_p, var_p = ens.predict(X[:5])
    print(f"Predictions: {mean_p.argmax(1).tolist()}")
    print(f"Mean variance: {var_p.mean(1).tolist()}")

    # --- Evidential Deep Learning ---
    print("\n=== Evidential Deep Learning ===")
    edl = EvidentialClassifier(D, 64, C)
    opt = optim.Adam(edl.parameters(), lr=0.005)
    for epoch in range(300):
        alpha = edl(X)
        loss = EvidentialClassifier.evidential_loss(alpha, y, kl_weight=0.01)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 100 == 0:
            probs = edl.expected_probability(alpha)
            acc = (probs.argmax(1) == y).float().mean()
            unc = edl.uncertainty(alpha).mean()
            print(f"Epoch {epoch+1} | Loss: {loss:.3f} | Acc: {acc:.3f} | "
                  f"Mean uncertainty: {unc:.3f}")

    # Test on OOD data (should show higher uncertainty)
    ood = torch.randn(20, D) * 5  # far from training distribution
    alpha_ood = edl(ood)
    print(f"In-distribution uncertainty:  {edl.uncertainty(edl(X)).mean():.3f}")
    print(f"Out-of-distribution uncertainty: {edl.uncertainty(alpha_ood).mean():.3f}")

    # --- Active Learning ---
    print("\n=== Uncertainty Active Learning ===")
    al_model = make_mlp()
    opt = optim.Adam(al_model.parameters(), lr=0.01)
    for _ in range(100):
        loss = F.cross_entropy(al_model(X[:50]), y[:50])  # train on small subset
        opt.zero_grad(); loss.backward(); opt.step()

    al = UncertaintyActiveLearner(al_model, strategy="bald")
    selected = al.select(X[50:], n_select=5)
    print(f"Most uncertain sample indices (from pool): {selected.tolist()}")
