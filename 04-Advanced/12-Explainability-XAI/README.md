# 🔍 Explainability & Interpretability (XAI)

## Overview

As deep learning models are deployed in high-stakes domains, understanding
*why* a model makes a prediction is critical. This module covers gradient-based,
perturbation-based, and concept-based explanation methods, plus tools for
auditing model fairness.

## Contents

| File | Topics |
|------|--------|
| `01_xai_fundamentals.py` | Saliency maps, Grad-CAM, Integrated Gradients, occlusion sensitivity |
| `02_advanced_xai.py` | LIME (local surrogate), SHAP (Shapley values), concept-based explanations (TCAV-style), fairness metrics |

## Prerequisites

- CNN basics (Section 2.3)
- PyTorch autograd

## Learning Objectives

1. Compute and visualize gradient-based saliency maps
2. Implement Grad-CAM for CNN explanations
3. Use Integrated Gradients for attribution
4. Build a local surrogate (LIME) from scratch
5. Approximate Shapley values for feature importance
6. Understand concept-based testing (TCAV)

## Key Concepts

- **Saliency map**: Gradient of output w.r.t. input pixels
- **Grad-CAM**: Class-discriminative localization using conv feature maps
- **Integrated Gradients**: Axiomatic attribution along a straight-line path
- **LIME**: Fit a simple model on perturbed neighbors of an input
- **SHAP**: Game-theoretic feature importance via Shapley values

## References

- Selvaraju et al., *Grad-CAM* (2017)
- Sundararajan et al., *Axiomatic Attribution for Deep Networks* (2017)
- Ribeiro et al., *"Why Should I Trust You?"* (LIME, 2016)
- Lundberg & Lee, *SHAP* (2017)
