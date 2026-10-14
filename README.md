# SHAP-guided Robustness Evaluation of Deep Learning-based Smart-home NIDSs

This repository contains the code and experiments for the paper
**"SHAP-guided Robustness Evaluation of Deep Learning-based Smart-home NIDSs"**.

## Overview

Deep learning-based Network Intrusion Detection Systems (NIDSs) achieve strong
detection performance in Smart-home IoT environments, but they remain black-box
models. SHAP explanations reveal which traffic features drive a prediction, yet
the same information can also expose weaknesses an attacker may exploit to evade
detection.

This project proposes a **SHAP-guided robustness evaluation procedure**:
explanation-guided feature perturbations are used as a *diagnostic probe*, not as
a new adversarial attack.

## What it does

- Six-class classification on **ToN-IoT** (benign, dos, infog-scan-port, inject,
  malwr, mitm-arpspoof), with biflows represented by their first 10 packets and
  6 per-packet features (PL, IAT, DIR, WIN, TTL, FLG).
- Comparison of three architectures with different inductive biases:
  - **LopezCNN** (convolutional)
  - **BiflowTransformer** (self-attention)
  - **NetMamba** (selective state-space model)
- Explanations via **SHAP GradientExplainer**, aggregated into class-level
  importance profiles (median of normalized attributions).
- Two threat models:
  - **Untargeted evasion**: any misclassification counts as success.
  - **Benign-targeted evasion**: malicious traffic must be classified as benign.
- Two feature-constraint regimes (**loose** and **strict**) plus a random baseline.
- Metrics: Attack Success Rate (ASR), Average Minimum Successful Perturbation
  (AMSP), and Evasion Vulnerability Score (EVS).

## Key findings

- SHAP-guided feature selection outperforms random selection, increasing ASR
  while reducing the magnitude of successful perturbations.
- Benign-targeted evasion is substantially harder than generic misclassification
  and strongly architecture- and class-dependent.
- **NetMamba** is the most exposed architecture, while **BiflowTransformer** is the
  most robust in the standard benign-targeted setting.
- Successful evasions concentrate on a few features, mainly **Packet Length (PL)**
  and **Time To Live (TTL)**.

## Limitations

The evaluation is performed in feature space with a bounded search budget
(MAX_STEPS = 1000), single-feature non-cumulative perturbations, and no
packet-level validation. A failed attack only means that no successful
perturbation was found within the given budget.

## Future work

Larger search budgets and multi-feature perturbations, packet-level validation,
additional datasets and model families, and defensive uses such as adversarial
training.

## Contributors
* **Matteo D'Avino** M63001724
* **Luigi Fienga** M63001733
* **Andrea Roscigno** M63001778


