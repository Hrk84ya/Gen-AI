# 🎲 Bayesian Deep Learning

## Overview

Standard neural networks produce point estimates and can be
over-confident. Bayesian deep learning places distributions over
weights (or predictions), giving principled uncertainty estimates.
This module covers practical techniques from MC Dropout to full
variational inference.

## Contents

| File | Topics |
|------|--------|
| `01_bayesian_fundamentals.py` | MC Dropout, concrete dropout, Bayes-by-Backprop (weight uncertainty), calibration metrics |
| `02_advanced_bayesian.py` | Deep ensembles, SWAG, evidential deep learning, uncertainty-aware active learning |

## Prerequisites

- Neural network basics (Level 2)
- Probability & statistics (Section 1.3)

## Learning Objectives

1. Distinguish aleatoric vs epistemic uncertainty
2. Use MC Dropout for approximate Bayesian inference
3. Implement Bayes-by-Backprop with the reparameterization trick
4. Build deep ensembles and compare uncertainty quality
5. Apply uncertainty estimates to active learning

## Key Concepts

- **Epistemic uncertainty**: Model uncertainty, reducible with more data
- **Aleatoric uncertainty**: Data noise, irreducible
- **MC Dropout**: Run dropout at test time, average predictions
- **Bayes-by-Backprop**: Learn a distribution q(w) over weights via variational inference
- **Calibration**: How well predicted probabilities match actual frequencies

## References

- Gal & Ghahramani, *Dropout as a Bayesian Approximation* (2016)
- Blundell et al., *Weight Uncertainty in Neural Networks* (2015)
- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty* (2017)
