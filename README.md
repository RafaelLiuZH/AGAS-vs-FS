# AGAS vs. FraudSquad Repository

This repository implements the adversarial benchmark between **AGAS (Agentic Group Attack System, ICDM 2026)** as a **Black-Box Shilling Attacker** and **FraudSquad** as a **Gray-Box Gated Graph Neural Detector**.

---

## Architectural Workflow

### 1. Attacker: AGAS (Black-Box)
- **Zero Internal Access**: Operates solely by observing platform feedback signals (target rank, acceptance rate, alert triggers).
- **LLM Coordinator & Worker Roles**: Manages worker agents across 8 strategies ($S_1 - S_8$) and assigns roles:
  - `Profiler (PR)`: Posts safe filler ratings to build co-occurrence bridges.
  - `Sniper (SN)`: Pushes payload ratings on target items.
  - `Camouflageur (CA)`: Posts benign reviews to lower platform alert levels.
  - `Inactive (IN)`: Cools down to evade rate-limiting rules.

### 2. Defender: FraudSquad (Gray-Box)
- **BERT Text Feature Extraction**: Extracts dense linguistic embeddings ($X_i$) from review text.
- **Masked Risk Label Embeddings ($Z_i$)**: Maintains trainable embeddings for `{normal, fraud, unknown}`, masked during training to prevent label leakage.
- **Review Graph Construction ($G$)**: Connects review nodes sharing same user, product + rating, or product + month.
- **2-Layer Gated Graph Transformer**: Computes multi-head attention weights ($\\alpha_{ij}$) and uses a gated shortcut to combine local features with aggregated graph context:
  $$\\hat{H}_i = \\text{gate}_i \\odot O_i + (1 - \\text{gate}_i) \\odot \\tilde{H}_i$$
- **Top 3% Percentile Cutoff**: Ranks MLP classification probabilities $\\hat{y}_i \\in [0, 1]$ and flags top suspicious candidates to uncover bot accounts.

---

## Quickstart Guide

### 1. Repository Setup
```bash
git clone <repository-url>
cd agas-vs-fraudsquad
pip install -r requirements.txt
pip install -e .
```

### 2. Run Benchmark Experiment
```bash
python experiments/run_simulation.py
```

### 3. Run Test Suite
```bash
pytest tests/
```
