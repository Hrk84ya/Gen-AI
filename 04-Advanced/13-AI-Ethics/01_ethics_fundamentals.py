"""
AI Ethics Fundamentals
========================
Covers: Bias detection & measurement, fairness-constrained training,
dataset auditing, and model card generation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. Bias & Fairness Metrics
# ---------------------------------------------------------------------------

def disparate_impact_ratio(predictions: np.ndarray,
                           sensitive: np.ndarray) -> float:
    """
    Disparate Impact Ratio = P(ŷ=1 | A=0) / P(ŷ=1 | A=1).

    The 80% rule: a ratio below 0.8 suggests disparate impact.
    Values close to 1.0 indicate parity.
    """
    g0 = predictions[sensitive == 0].mean()
    g1 = predictions[sensitive == 1].mean()
    if g1 == 0:
        return float("inf")
    return float(g0 / g1)


def equal_opportunity_diff(predictions: np.ndarray, labels: np.ndarray,
                           sensitive: np.ndarray) -> float:
    """
    |TPR(A=0) - TPR(A=1)|

    Equal opportunity requires equal true-positive rates across groups.
    """
    tprs = []
    for g in [0, 1]:
        mask = (sensitive == g) & (labels == 1)
        if mask.sum() == 0:
            tprs.append(0.0)
        else:
            tprs.append(predictions[mask].mean())
    return float(abs(tprs[0] - tprs[1]))


def demographic_parity_diff(predictions: np.ndarray,
                            sensitive: np.ndarray) -> float:
    """
    |P(ŷ=1 | A=0) - P(ŷ=1 | A=1)|
    """
    rates = [predictions[sensitive == g].mean() for g in [0, 1]]
    return float(abs(rates[0] - rates[1]))


def calibration_by_group(probs: np.ndarray, labels: np.ndarray,
                         sensitive: np.ndarray,
                         n_bins: int = 10) -> Dict[int, float]:
    """
    Per-group Expected Calibration Error.

    Returns dict mapping group → ECE.
    """
    result = {}
    for g in np.unique(sensitive):
        mask = sensitive == g
        p, y = probs[mask], labels[mask]
        confidences = p.max(axis=1) if p.ndim > 1 else p
        preds = (p.argmax(axis=1) if p.ndim > 1
                 else (p > 0.5).astype(int))
        correct = (preds == y).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if in_bin.sum() == 0:
                continue
            ece += in_bin.sum() * abs(correct[in_bin].mean() -
                                      confidences[in_bin].mean())
        result[int(g)] = ece / len(y)
    return result


# ---------------------------------------------------------------------------
# 2. Dataset Auditor
# ---------------------------------------------------------------------------

class DatasetAuditor:
    """
    Audit a dataset for representation imbalance and label skew
    across sensitive groups.

    Usage:
        auditor = DatasetAuditor(labels, sensitive_attr)
        report = auditor.audit()
    """

    def __init__(self, labels: np.ndarray, sensitive: np.ndarray,
                 group_names: Optional[List[str]] = None):
        self.labels = labels
        self.sensitive = sensitive
        self.groups = np.unique(sensitive)
        self.group_names = group_names or [str(g) for g in self.groups]

    def audit(self) -> Dict:
        """Return a structured audit report."""
        total = len(self.labels)
        report = {"total_samples": total, "groups": {}}

        for g, name in zip(self.groups, self.group_names):
            mask = self.sensitive == g
            count = mask.sum()
            label_dist = {}
            for lbl in np.unique(self.labels):
                label_dist[int(lbl)] = int((self.labels[mask] == lbl).sum())
            report["groups"][name] = {
                "count": int(count),
                "fraction": round(count / total, 4),
                "label_distribution": label_dist,
            }

        # Flag imbalance
        fracs = [v["fraction"] for v in report["groups"].values()]
        report["max_imbalance_ratio"] = round(max(fracs) / max(min(fracs), 1e-9), 2)
        return report


# ---------------------------------------------------------------------------
# 3. Fairness-Constrained Training
# ---------------------------------------------------------------------------

class FairClassifier(nn.Module):
    """
    Binary classifier trained with a fairness penalty.

    Loss = BCE + λ · fairness_penalty

    The penalty encourages equal positive-prediction rates across
    groups (demographic parity).  Uses a differentiable relaxation:
    penalty = (mean(σ(logits) | A=0) - mean(σ(logits) | A=1))²
    """

    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def fairness_loss(logits: torch.Tensor, sensitive: torch.Tensor,
                      lam: float = 1.0) -> torch.Tensor:
        """Differentiable demographic-parity penalty."""
        probs = torch.sigmoid(logits)
        g0 = probs[sensitive == 0].mean()
        g1 = probs[sensitive == 1].mean()
        return lam * (g0 - g1) ** 2


def train_fair_classifier(X: torch.Tensor, y: torch.Tensor,
                          sensitive: torch.Tensor,
                          lam: float = 1.0, epochs: int = 300,
                          lr: float = 0.01):
    """Train with and without fairness constraint for comparison."""
    results = {}
    for name, use_fair in [("unconstrained", False), ("fair", True)]:
        model = FairClassifier(X.size(1))
        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            logits = model(X)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            if use_fair:
                loss = loss + FairClassifier.fairness_loss(logits, sensitive, lam)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            preds = (torch.sigmoid(model(X)) > 0.5).numpy().astype(float)
        dp = demographic_parity_diff(preds, sensitive.numpy())
        acc = (preds == y.numpy()).mean()
        results[name] = {"accuracy": round(acc, 4), "dp_diff": round(dp, 4)}
    return results


# ---------------------------------------------------------------------------
# 4. Model Card Generator
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """
    Structured model documentation following Mitchell et al. 2019.

    Captures intended use, limitations, evaluation metrics, and
    ethical considerations in a standardized format.
    """
    model_name: str = ""
    version: str = "1.0"
    description: str = ""
    intended_use: str = ""
    out_of_scope: str = ""
    training_data: str = ""
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Model Card: {self.model_name}",
            f"**Version**: {self.version}\n",
            f"## Description\n{self.description}\n",
            f"## Intended Use\n{self.intended_use}\n",
            f"## Out of Scope\n{self.out_of_scope}\n",
            f"## Training Data\n{self.training_data}\n",
            "## Evaluation Metrics",
        ]
        for k, v in self.eval_metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("\n## Fairness Metrics")
        for k, v in self.fairness_metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("\n## Limitations")
        for item in self.limitations:
            lines.append(f"- {item}")
        lines.append("\n## Ethical Considerations")
        for item in self.ethical_considerations:
            lines.append(f"- {item}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Synthetic biased dataset
    N = 600
    sensitive = np.random.binomial(1, 0.5, N)
    X = np.random.randn(N, 8).astype(np.float32)
    # Inject bias: group 1 gets a feature boost that correlates with label
    X[sensitive == 1, 0] += 1.0
    true_w = np.random.randn(8).astype(np.float32)
    y = ((X @ true_w + 0.5 * np.random.randn(N)) > 0).astype(np.float32)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    s_t = torch.tensor(sensitive)

    # --- Bias metrics ---
    print("=== Dataset Audit ===")
    auditor = DatasetAuditor(y.astype(int), sensitive,
                             group_names=["Group A", "Group B"])
    report = auditor.audit()
    for name, info in report["groups"].items():
        print(f"  {name}: {info['count']} samples ({info['fraction']:.1%}), "
              f"labels={info['label_distribution']}")
    print(f"  Imbalance ratio: {report['max_imbalance_ratio']}")

    # --- Fair vs unfair training ---
    print("\n=== Fairness-Constrained Training ===")
    results = train_fair_classifier(X_t, y_t, s_t, lam=2.0, epochs=300)
    for name, metrics in results.items():
        print(f"  {name}: acc={metrics['accuracy']}, "
              f"demographic_parity_diff={metrics['dp_diff']}")

    # --- Model card ---
    print("\n=== Model Card ===")
    card = ModelCard(
        model_name="FairBinaryClassifier",
        description="Binary classifier trained with demographic parity constraint.",
        intended_use="Demonstration of fairness-aware training.",
        out_of_scope="Production deployment without further validation.",
        training_data="Synthetic Gaussian data with injected group bias.",
        eval_metrics={"accuracy": results["fair"]["accuracy"]},
        fairness_metrics={"demographic_parity_diff": results["fair"]["dp_diff"]},
        limitations=["Synthetic data only", "Binary sensitive attribute"],
        ethical_considerations=[
            "Fairness constraint may reduce overall accuracy",
            "Single metric does not capture all fairness dimensions",
        ],
    )
    print(card.to_markdown()[:500] + "\n...")
