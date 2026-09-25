# Veto-40M: Energy-Based Decision Transformer

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Veto--40M-FFD21E)](https://huggingface.co/saqiibb/Veto-40M)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Veto-40M** is a 33.7M parameter Energy-Based Decision Transformer designed for joint decision ranking, intent classification, and out-of-domain action rejection.

Instead of applying a Softmax distribution over a fixed set of classes, Veto-40M evaluates arbitrary `(State, Choice)` pairs and projects them into an unnormalized scalar energy landscape. Valid decisions are mapped toward low-energy basins ($E \approx 0$), while invalid or out-of-domain choices yield high energy scores.

---

## Key Features

* **Energy-Based Scalar Head**: Evaluates any state-choice pair independently to output a scalar energy value.
* **Custom Tokenizer**: Built with a Byte-Level BPE tokenizer utilizing special control tokens (`<state>`, `<choice>`).
* **Hybrid Loss Strategy**: Trained with Softmin Cross-Entropy for relative choice ranking alongside $L_2$ positive grounding to prevent energy drift.
* **Modern Transformer Design**: Incorporates RMSNorm, SwiGLU activation functions, and PyTorch Scaled Dot-Product Attention (SDPA).

---

## Performance Benchmarks

Evaluated on the **BANKING77** test dataset across varying candidate choice set sizes ($K$ choices per state):

| Benchmark Metric | Candidate Choices ($K$) | Accuracy (%) |
| :--- | :---: | :---: |
| **Focused Ranking** | 6 Choices | **94.48%** |
| **Extended Ranking** | 20 Choices | **88.31%** |
| **Full Intent Resolution** | 77 Choices | **75.29%** |

---

## Architecture & Hyperparameters

| Configuration Parameter | Value |
| :--- | :--- |
| **Total Parameters** | 33,694,720 (33.7M) |
| **Vocabulary Size** | 16,384 (Byte-Level BPE) |
| **Hidden Dimension ($d_{model}$)** | 512 |
| **Intermediate Dimension ($d_{ffn}$)** | 1,376 (SwiGLU) |
| **Transformer Layers** | 8 |
| **Attention Heads** | 8 |
| **Max Sequence Length** | 256 |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) |
| **Output Head** | Linear Scalar Projection (No Bias) |

---

## Loss Formulation

The network is optimized using a combined loss function ($\mathcal{L}_{\text{total}}$) consisting of softmin ranking and $L_2$ grounding:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ranking}} + \lambda_{\text{ground}} \cdot \mathcal{L}_{\text{grounding}}$$

### 1. Softmin Ranking Loss
Negative energies act as unnormalized logits where lower energy corresponds to higher confidence:

$$\mathcal{L}_{\text{ranking}} = -\log \left( \frac{\exp(-E(s, c^+)/\tau)}{\sum_{i=0}^{N} \exp(-E(s, c_i)/\tau)} \right)$$

### 2. Positive $L_2$ Grounding Loss
Anchors valid choices to low-energy baselines near zero ($\lambda_{\text{ground}} = 0.1$):

$$\mathcal{L}_{\text{grounding}} = \frac{1}{B} \sum_{i=1}^{B} \left( E(s_i, c_i^+) \right)^2$$

---

## Repository Structure

```text
.
├── veto-40m-decision.ipynb   # Full notebook with training, evaluation, and export steps
├── README.md                 # Project documentation
└── veto-40m-local/           # Exported model directory
    ├── pytorch_model.bin     # Model weights
    ├── config.json           # Model configuration
    └── tokenizer.json        # BPE Tokenizer configuration
