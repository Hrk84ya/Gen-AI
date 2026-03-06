"""
Explainability & Interpretability — Fundamentals
===================================================
Covers: Vanilla saliency maps, Grad-CAM, Integrated Gradients,
and occlusion sensitivity.

All methods work with any PyTorch classifier.
"""

import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Vanilla Saliency Map — Simonyan et al. 2014
# ---------------------------------------------------------------------------

def saliency_map(model: nn.Module, x: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
    """
    Compute the gradient of the target class score w.r.t. the input.

    Args:
        model: classifier (outputs logits)
        x: single input tensor (1, C, H, W) or (1, features)
        target_class: class index; if None uses argmax

    Returns:
        saliency: absolute gradient, same spatial shape as x (numpy)
    """
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = logits.argmax(dim=-1).item()
    score = logits[0, target_class]
    score.backward()
    sal = x.grad.data.abs().squeeze().cpu().numpy()
    return sal


# ---------------------------------------------------------------------------
# 2. Grad-CAM — Selvaraju et al. 2017
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Highlights which spatial regions of a conv feature map are most
    important for a given class prediction.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(input_image, class_idx=5)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Args:
            x: (1, C, H, W)
            target_class: class index; if None uses argmax

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        logits = self.model(x)
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global-average-pool the gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# ---------------------------------------------------------------------------
# 3. Integrated Gradients — Sundararajan et al. 2017
# ---------------------------------------------------------------------------

def integrated_gradients(model: nn.Module, x: torch.Tensor,
                         target_class: Optional[int] = None,
                         baseline: Optional[torch.Tensor] = None,
                         n_steps: int = 50) -> np.ndarray:
    """
    Axiomatic attribution method.

    Integrates the gradient along a straight-line path from a baseline
    (default: zeros) to the input.

    IG_i = (x_i - x'_i) × ∫₀¹ ∂F/∂x_i(x' + α(x - x')) dα

    Args:
        model: classifier
        x: single input (1, ...)
        target_class: class index
        baseline: reference input (same shape as x); defaults to zeros
        n_steps: number of interpolation steps

    Returns:
        attributions: same shape as x (numpy)
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    # Determine target class
    with torch.no_grad():
        logits = model(x)
    if target_class is None:
        target_class = logits.argmax(dim=-1).item()

    # Interpolate and accumulate gradients
    scaled_inputs = [baseline + (float(i) / n_steps) * (x - baseline)
                     for i in range(n_steps + 1)]
    grads = []
    for inp in scaled_inputs:
        inp = inp.clone().detach().requires_grad_(True)
        out = model(inp)
        out[0, target_class].backward()
        grads.append(inp.grad.data.clone())

    # Trapezoidal approximation of the integral
    avg_grad = torch.stack(grads).mean(dim=0)
    ig = (x - baseline) * avg_grad
    return ig.squeeze().detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 4. Occlusion Sensitivity
# ---------------------------------------------------------------------------

def occlusion_sensitivity(model: nn.Module, x: torch.Tensor,
                          target_class: Optional[int] = None,
                          patch_size: int = 4,
                          stride: int = 2) -> np.ndarray:
    """
    Slide a zero-patch over the input and measure how the target
    class score changes.

    Works for (1, C, H, W) image inputs.

    Returns:
        sensitivity_map: (H, W) numpy array — higher = more important
    """
    model.eval()
    with torch.no_grad():
        base_logits = model(x)
    if target_class is None:
        target_class = base_logits.argmax(dim=-1).item()
    base_score = base_logits[0, target_class].item()

    _, C, H, W = x.shape
    sens = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            occluded = x.clone()
            occluded[:, :, i:i + patch_size, j:j + patch_size] = 0
            with torch.no_grad():
                score = model(occluded)[0, target_class].item()
            drop = base_score - score
            sens[i:i + patch_size, j:j + patch_size] += drop
            counts[i:i + patch_size, j:j + patch_size] += 1

    counts[counts == 0] = 1
    return sens / counts


# ---------------------------------------------------------------------------
# 5. Helper: Simple CNN for demos
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Tiny CNN for demonstration purposes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    model = SimpleCNN(num_classes=10)
    dummy = torch.randn(1, 1, 28, 28)

    print("=== Saliency Map ===")
    sal = saliency_map(model, dummy, target_class=0)
    print(f"Shape: {sal.shape}, range: [{sal.min():.4f}, {sal.max():.4f}]")

    print("\n=== Grad-CAM ===")
    cam = GradCAM(model, target_layer=model.features[3])  # second conv
    heatmap = cam(dummy, target_class=0)
    print(f"Heatmap shape: {heatmap.shape}")

    print("\n=== Integrated Gradients ===")
    ig = integrated_gradients(model, dummy, target_class=0, n_steps=30)
    print(f"Shape: {ig.shape}, range: [{ig.min():.4f}, {ig.max():.4f}]")

    print("\n=== Occlusion Sensitivity ===")
    occ = occlusion_sensitivity(model, dummy, target_class=0,
                                patch_size=4, stride=2)
    print(f"Shape: {occ.shape}, range: [{occ.min():.4f}, {occ.max():.4f}]")
