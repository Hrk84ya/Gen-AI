"""
Bayesian Deep Learning Fundamentals
======================================
Covers: MC Dropout, Concrete Dropout, Bayes-by-Backprop,
calibration metrics (ECE).
"""

import math
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. MC Dropout — Gal & Ghahramani 2016
# ---------------------------------------------------------------------------

class MCDropoutNet(nn.Module):
    """
    Standard MLP that keeps dropout ON at test time.

    By running T forward passes and averaging, we get an approximate
    posterior predictive distribution.  The variance across passes
    estimates epistemic uncertainty.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int,
                 drop_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(drop_p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop_p),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run n_samples stochastic forward passes (dropout active).

        Returns:
            mean_pred: averaged softmax probabilities  (batch, classes)
            uncertainty: predictive variance             (batch, classes)
        """
        self.train()  # keep dropout on
        preds = torch.stack([F.softmax(self(x), dim=-1) for _ in range(n_samples)])
        mean_pred = preds.mean(dim=0)
        uncertainty = preds.var(dim=0)
        return mean_pred, uncertainty


# ---------------------------------------------------------------------------
# 2. Concrete Dropout — Gal et al. 2017
# ---------------------------------------------------------------------------

class ConcreteDropout(nn.Module):
    """
    Learns the dropout probability via gradient descent.

    Instead of hand-tuning p, we parameterize it as σ(logit_p)
    and add a regularization term that balances model complexity
    against data fit.
    """

    def __init__(self, layer: nn.Module, weight_reg: float = 1e-6,
                 drop_reg: float = 1e-3):
        super().__init__()
        self.layer = layer
        self.weight_reg = weight_reg
        self.drop_reg = drop_reg
        # Initialize logit so p ≈ 0.1
        self.logit_p = nn.Parameter(torch.tensor(-2.2))

    @property
    def p(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        # Concrete relaxation of Bernoulli dropout
        if self.training:
            u = torch.rand_like(x).clamp(1e-6, 1 - 1e-6)
            drop_mask = torch.sigmoid((torch.log(u) - torch.log(1 - u) +
                                       self.logit_p) / 0.1)
            x = x * drop_mask / (1 - p)
        return self.layer(x)

    def regularization(self) -> torch.Tensor:
        """KL-divergence-style regularization for the dropout rate."""
        p = self.p
        # Entropy of Bernoulli(p)
        entropy = -p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
        # Weight regularization scaled by (1-p)
        weight_sq = sum((w ** 2).sum() for w in self.layer.parameters())
        return self.weight_reg * weight_sq / (1 - p) - self.drop_reg * entropy


# ---------------------------------------------------------------------------
# 3. Bayes-by-Backprop — Blundell et al. 2015
# ---------------------------------------------------------------------------

class BayesLinear(nn.Module):
    """
    Linear layer with Gaussian weight posterior q(w) = N(μ, σ²).

    Uses the reparameterization trick:  w = μ + σ ⊙ ε,  ε ~ N(0,1)

    The ELBO loss = NLL + KL(q(w) || p(w)) is computed per-layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_rho = nn.Parameter(torch.empty(out_features))

        # Prior
        self.prior_sigma = prior_sigma

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        nn.init.constant_(self.w_rho, -3.0)  # σ ≈ log(1+exp(-3)) ≈ 0.05
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_mu, -bound, bound)
        nn.init.constant_(self.b_rho, -3.0)

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_sigma = self._softplus(self.w_rho)
        b_sigma = self._softplus(self.b_rho)

        if self.training:
            w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
            b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            w, b = self.w_mu, self.b_mu

        return F.linear(x, w, b)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(w) || N(0, prior_σ²)) in closed form for Gaussians."""
        w_sigma = self._softplus(self.w_rho)
        b_sigma = self._softplus(self.b_rho)

        prior_var = self.prior_sigma ** 2

        kl_w = 0.5 * (w_sigma ** 2 / prior_var + self.w_mu ** 2 / prior_var
                       - 1 - 2 * torch.log(w_sigma / self.prior_sigma)).sum()
        kl_b = 0.5 * (b_sigma ** 2 / prior_var + self.b_mu ** 2 / prior_var
                       - 1 - 2 * torch.log(b_sigma / self.prior_sigma)).sum()
        return kl_w + kl_b


class BayesNet(nn.Module):
    """MLP with BayesLinear layers."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.l1 = BayesLinear(in_features, hidden)
        self.l2 = BayesLinear(hidden, hidden)
        self.l3 = BayesLinear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def kl_divergence(self) -> torch.Tensor:
        return self.l1.kl_divergence() + self.l2.kl_divergence() + self.l3.kl_divergence()


# ---------------------------------------------------------------------------
# 4. Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 15) -> float:
    """
    Compute ECE — measures how well predicted probabilities match
    actual accuracy.

    Args:
        probs: predicted class probabilities (N, C)
        labels: true class indices (N,)
        n_bins: number of confidence bins

    Returns:
        ece: scalar in [0, 1]
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(labels)


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Synthetic classification data ---
    N, D, C = 500, 10, 3
    X = torch.randn(N, D)
    W_true = torch.randn(D, C)
    y = (X @ W_true + 0.3 * torch.randn(N, C)).argmax(dim=1)

    # --- MC Dropout ---
    print("=== MC Dropout ===")
    mc_model = MCDropoutNet(D, 64, C, drop_p=0.2)
    opt = optim.Adam(mc_model.parameters(), lr=0.01)
    for epoch in range(200):
        mc_model.train()
        loss = F.cross_entropy(mc_model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    mean_p, unc = mc_model.predict_with_uncertainty(X[:5], n_samples=50)
    print(f"Predictions: {mean_p.argmax(1).tolist()}")
    print(f"Mean uncertainty: {unc.mean(1).tolist()}")

    # --- Bayes-by-Backprop ---
    print("\n=== Bayes-by-Backprop ===")
    bnn = BayesNet(D, 64, C)
    opt = optim.Adam(bnn.parameters(), lr=0.005)
    for epoch in range(300):
        bnn.train()
        logits = bnn(X)
        nll = F.cross_entropy(logits, y)
        kl = bnn.kl_divergence() / N  # scale KL by dataset size
        loss = nll + kl
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 100 == 0:
            acc = (logits.argmax(1) == y).float().mean()
            print(f"Epoch {epoch+1} | NLL: {nll:.3f} | KL: {kl:.4f} | Acc: {acc:.3f}")

    # --- Calibration ---
    print("\n=== Calibration ===")
    bnn.eval()
    with torch.no_grad():
        probs = F.softmax(bnn(X), dim=-1).numpy()
    ece = expected_calibration_error(probs, y.numpy())
    print(f"ECE (Bayes net): {ece:.4f}")
