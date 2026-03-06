"""
Advanced Explainability & Interpretability
============================================
Covers: LIME (local surrogate), SHAP (Shapley-value approximation),
concept-based explanations (TCAV-style), and fairness metrics.
"""

import numpy as np
from typing import Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. LIME — Ribeiro et al. 2016
# ---------------------------------------------------------------------------

class LIME:
    """
    Local Interpretable Model-agnostic Explanations.

    For a single prediction, perturb the input, query the black-box
    model, and fit a weighted linear model on the neighborhood.
    The linear coefficients are the feature importances.
    """

    def __init__(self, predict_fn: Callable, n_samples: int = 500,
                 kernel_width: float = 0.75):
        """
        Args:
            predict_fn: callable(X) → probabilities (N, C)
            n_samples: number of perturbed samples
            kernel_width: RBF kernel width for locality weighting
        """
        self.predict_fn = predict_fn
        self.n_samples = n_samples
        self.kernel_width = kernel_width

    def explain(self, x: np.ndarray,
                target_class: Optional[int] = None) -> np.ndarray:
        """
        Args:
            x: single input (D,)
            target_class: class to explain; if None uses argmax

        Returns:
            importances: (D,) linear coefficients
        """
        D = x.shape[0]

        # Generate binary perturbation masks
        masks = np.random.binomial(1, 0.5, size=(self.n_samples, D)).astype(np.float32)
        # Create perturbed inputs (mask=1 keeps original, mask=0 replaces with 0)
        perturbed = masks * x[np.newaxis, :]

        # Get model predictions
        probs = self.predict_fn(perturbed)  # (n_samples, C)
        if target_class is None:
            target_class = self.predict_fn(x[np.newaxis, :]).argmax(axis=1)[0]
        target_probs = probs[:, target_class]

        # Compute locality weights (RBF kernel on mask distance)
        distances = np.sqrt(((1 - masks) ** 2).sum(axis=1))
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # Weighted least squares: (M^T W M)^{-1} M^T W y
        W = np.diag(weights)
        M = np.hstack([masks, np.ones((self.n_samples, 1))])  # add bias
        try:
            theta = np.linalg.solve(M.T @ W @ M + 1e-6 * np.eye(D + 1),
                                    M.T @ W @ target_probs)
        except np.linalg.LinAlgError:
            theta = np.zeros(D + 1)
        return theta[:D]  # drop bias term


# ---------------------------------------------------------------------------
# 2. SHAP (Simplified Kernel SHAP) — Lundberg & Lee 2017
# ---------------------------------------------------------------------------

class KernelSHAP:
    """
    Approximate Shapley values using weighted linear regression
    on coalition samples (Kernel SHAP).

    Shapley values satisfy desirable axioms: efficiency, symmetry,
    dummy, and additivity.
    """

    def __init__(self, predict_fn: Callable, n_samples: int = 500):
        self.predict_fn = predict_fn
        self.n_samples = n_samples

    @staticmethod
    def _shapley_kernel_weight(D: int, n_ones: int) -> float:
        """SHAP kernel weight for a coalition of size n_ones out of D."""
        if n_ones == 0 or n_ones == D:
            return 1e6  # large weight for full/empty coalitions
        from math import comb
        return (D - 1) / (comb(D, n_ones) * n_ones * (D - n_ones))

    def explain(self, x: np.ndarray, baseline: Optional[np.ndarray] = None,
                target_class: Optional[int] = None) -> np.ndarray:
        """
        Args:
            x: single input (D,)
            baseline: reference input; defaults to zeros
            target_class: class to explain

        Returns:
            shap_values: (D,) feature attributions
        """
        D = x.shape[0]
        if baseline is None:
            baseline = np.zeros_like(x)

        if target_class is None:
            target_class = self.predict_fn(x[np.newaxis, :]).argmax(axis=1)[0]

        # Sample coalitions
        coalitions = np.random.binomial(1, 0.5, size=(self.n_samples, D)).astype(np.float32)
        # Build inputs: coalition=1 → use x, coalition=0 → use baseline
        inputs = coalitions * x + (1 - coalitions) * baseline
        probs = self.predict_fn(inputs)[:, target_class]

        # Compute SHAP kernel weights
        weights = np.array([
            self._shapley_kernel_weight(D, int(c.sum()))
            for c in coalitions
        ])

        # Weighted least squares
        W = np.diag(weights)
        M = np.hstack([coalitions, np.ones((self.n_samples, 1))])
        try:
            theta = np.linalg.solve(M.T @ W @ M + 1e-6 * np.eye(D + 1),
                                    M.T @ W @ probs)
        except np.linalg.LinAlgError:
            theta = np.zeros(D + 1)
        return theta[:D]


# ---------------------------------------------------------------------------
# 3. Concept-Based Explanations (TCAV-style) — Kim et al. 2018
# ---------------------------------------------------------------------------

class ConceptTester:
    """
    Simplified Testing with Concept Activation Vectors (TCAV).

    Idea: define a "concept" by a set of positive/negative examples,
    train a linear probe on intermediate activations, then measure
    how sensitive the model's predictions are to that concept direction.
    """

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self._activations = None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._activations = output.detach()

    def get_activations(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            self.model(X)
        return self._activations.view(X.size(0), -1)

    def train_concept_vector(self, pos_examples: torch.Tensor,
                              neg_examples: torch.Tensor,
                              lr: float = 0.01,
                              epochs: int = 100) -> torch.Tensor:
        """
        Train a linear classifier to separate concept-positive from
        concept-negative activations.

        Returns:
            cav: concept activation vector (unit normal to decision boundary)
        """
        pos_act = self.get_activations(pos_examples)
        neg_act = self.get_activations(neg_examples)
        X = torch.cat([pos_act, neg_act])
        y = torch.cat([torch.ones(len(pos_act)), torch.zeros(len(neg_act))])

        D = X.size(1)
        probe = nn.Linear(D, 1)
        opt = optim.Adam(probe.parameters(), lr=lr)
        for _ in range(epochs):
            loss = F.binary_cross_entropy_with_logits(probe(X).squeeze(), y)
            opt.zero_grad(); loss.backward(); opt.step()

        cav = probe.weight.data.squeeze()
        return cav / cav.norm()

    def tcav_score(self, X: torch.Tensor, cav: torch.Tensor,
                   target_class: int) -> float:
        """
        Fraction of inputs for which the directional derivative
        along the CAV is positive (concept increases target class score).
        """
        acts = self.get_activations(X)
        acts.requires_grad_(True)

        # Recompute forward from activations (simplified: use a linear head)
        # In practice you'd hook into the real model's later layers.
        # Here we approximate with the dot product of activations and CAV.
        scores = (acts * cav.unsqueeze(0)).sum(dim=1)
        return (scores > 0).float().mean().item()


# ---------------------------------------------------------------------------
# 4. Fairness Metrics
# ---------------------------------------------------------------------------

def demographic_parity(predictions: np.ndarray,
                       sensitive_attr: np.ndarray) -> float:
    """
    Demographic parity difference.

    |P(ŷ=1 | A=0) - P(ŷ=1 | A=1)|

    Lower is fairer. Zero means the positive prediction rate is
    equal across groups.
    """
    groups = np.unique(sensitive_attr)
    rates = []
    for g in groups:
        mask = sensitive_attr == g
        rates.append(predictions[mask].mean())
    return float(abs(rates[0] - rates[1]))


def equalized_odds_diff(predictions: np.ndarray, labels: np.ndarray,
                         sensitive_attr: np.ndarray) -> float:
    """
    Equalized odds difference (max over TPR and FPR gaps).

    Measures whether the model has equal error rates across groups.
    """
    groups = np.unique(sensitive_attr)
    tpr, fpr = [], []
    for g in groups:
        mask = sensitive_attr == g
        pos = labels[mask] == 1
        neg = labels[mask] == 0
        tpr.append(predictions[mask][pos].mean() if pos.sum() > 0 else 0.0)
        fpr.append(predictions[mask][neg].mean() if neg.sum() > 0 else 0.0)
    return float(max(abs(tpr[0] - tpr[1]), abs(fpr[0] - fpr[1])))


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Simple model for demos ---
    D, C = 10, 3
    model = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
    opt = optim.Adam(model.parameters(), lr=0.01)
    X = np.random.randn(200, D).astype(np.float32)
    y_true = np.random.randint(0, C, 200)
    X_t, y_t = torch.tensor(X), torch.tensor(y_true)
    for _ in range(200):
        loss = F.cross_entropy(model(X_t), y_t)
        opt.zero_grad(); loss.backward(); opt.step()

    def predict_fn(x):
        with torch.no_grad():
            return F.softmax(model(torch.tensor(x, dtype=torch.float32)), dim=-1).numpy()

    # --- LIME ---
    print("=== LIME ===")
    lime = LIME(predict_fn, n_samples=500)
    imp = lime.explain(X[0])
    top3 = np.argsort(np.abs(imp))[-3:][::-1]
    print(f"Top-3 features: {top3.tolist()}, importances: {imp[top3].tolist()}")

    # --- Kernel SHAP ---
    print("\n=== Kernel SHAP ===")
    shap = KernelSHAP(predict_fn, n_samples=500)
    sv = shap.explain(X[0])
    top3 = np.argsort(np.abs(sv))[-3:][::-1]
    print(f"Top-3 features: {top3.tolist()}, SHAP values: {sv[top3].tolist()}")

    # --- Fairness ---
    print("\n=== Fairness Metrics ===")
    preds = predict_fn(X).argmax(axis=1)
    binary_preds = (preds == 1).astype(float)
    sensitive = np.random.binomial(1, 0.5, 200)
    dp = demographic_parity(binary_preds, sensitive)
    eo = equalized_odds_diff(binary_preds, (y_true == 1).astype(float), sensitive)
    print(f"Demographic Parity Diff: {dp:.4f}")
    print(f"Equalized Odds Diff:     {eo:.4f}")
