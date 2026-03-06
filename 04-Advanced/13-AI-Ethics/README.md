# ⚖️ AI Ethics

## Overview

This module provides a practical, code-driven approach to AI ethics.
Rather than only discussing principles, we implement bias detection,
fairness-aware training, privacy-preserving techniques, and
transparency tooling so you can build responsible AI systems.

## Contents

| File | Topics |
|------|--------|
| `01_ethics_fundamentals.py` | Bias detection & measurement, fairness constraints in training, dataset auditing, model cards |
| `02_privacy_and_safety.py` | Differential privacy (DP-SGD), membership inference attacks, toxicity filtering, safety guardrails |

## Prerequisites

- Neural network basics (Level 2)
- Familiarity with XAI concepts helps (Section 4.12) but is not required

## Learning Objectives

1. Measure bias in datasets and model predictions
2. Train models under fairness constraints (equalized odds, demographic parity)
3. Implement differential privacy with noisy SGD
4. Detect membership inference vulnerabilities
5. Build basic toxicity / safety filters
6. Generate structured model cards for documentation

## Key Concepts

- **Disparate impact**: When a model's outcomes disproportionately affect a protected group
- **Equalized odds**: Equal TPR and FPR across groups
- **Differential privacy**: Formal guarantee that any single training example has bounded influence
- **Membership inference**: Attack that determines whether a sample was in the training set
- **Model card**: Standardized documentation of a model's intended use, limitations, and evaluation

## References

- Barocas, Hardt & Narayanan, *Fairness and Machine Learning* (2019)
- Abadi et al., *Deep Learning with Differential Privacy* (2016)
- Mitchell et al., *Model Cards for Model Reporting* (2019)
- Shokri et al., *Membership Inference Attacks* (2017)
