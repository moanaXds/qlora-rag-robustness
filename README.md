# Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.36%2B-orange)](https://huggingface.co/transformers/)
[![PEFT QLoRA](https://img.shields.io/badge/PEFT-QLoRA%204--bit-purple)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PubMedQA Benchmark](https://img.shields.io/badge/Benchmark-PubMedQA%20(pqa__labeled)-green)](https://pubmedqa.github.io/)

**Author:** Muanna Hamid (Roll No: 242593)  
**Supervisor / Course Instructor:** Prof. Azka Atiq  
**Department:** Data Science, Faculty of Computing  
**Institution:** National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad  
**Course:** Data Science & Analytics (CS/DS) — Fall 2026  
**Deliverable:** Deliverable 3 (Phase 3: Research Implementation & Experimental Evaluation)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Research Questions & Hypotheses](#2-research-questions--hypotheses)
3. [Key Empirical Findings & Takeaways](#3-key-empirical-findings--takeaways)
4. [Methodological Framework & Architecture](#4-methodological-framework--architecture)
5. [Evidence Perturbation Taxonomy (C1–C4)](#5-evidence-perturbation-taxonomy-c1c4)
6. [Dataset & Data Leakage Guardrails](#6-dataset--data-leakage-guardrails)
7. [BM25 Retrieval & Corpus Indexing](#7-bm25-retrieval--corpus-indexing)
8. [Model Architecture & QLoRA Configuration](#8-model-architecture--qlora-configuration)
9. [Experimental Configurations (E1–E10 + Baselines)](#9-experimental-configurations-e1e10--baselines)
10. [Quantitative Benchmark Results](#10-quantitative-benchmark-results)
11. [Paired Statistical Significance & Holm-Bonferroni Correction](#11-paired-statistical-significance--holm-bonferroni-correction)
12. [Ablation Studies (Context Effect, Degradation & Epochs)](#12-ablation-studies-context-effect-degradation--epochs)
13. [Abstention Vulnerability Analysis (The 0% CAR Paradox)](#13-abstention-vulnerability-analysis-the-0-car-paradox)
14. [Adversarial Vulnerability & Counterfactual Adoption](#14-adversarial-vulnerability--counterfactual-adoption)
15. [Qualitative Grounding Audit & Error Taxonomy](#15-qualitative-grounding-audit--error-taxonomy)
16. [Key Visualizations & Publication Figures](#16-key-visualizations--publication-figures)
17. [Repository File Structure](#17-repository-file-structure)
18. [Installation & Reproduction Guide](#18-installation--reproduction-guide)
19. [Limitations & Threats to Validity](#19-limitations--threats-to-validity)
20. [Citation & Acknowledgments](#20-citation--acknowledgments)

---

## 1. Executive Summary

Retrieval-Augmented Generation (RAG) is widely adopted to mitigate hallucinations in Large Language Models (LLMs) by grounding responses in external documents. Furthermore, parameter-efficient fine-tuning via **4-bit Quantized Low-Rank Adaptation (QLoRA)** has become the standard mechanism for domain-adapting small, deployable language models under strict hardware budgets.

However, real-world biomedical retrieval is imperfect. Retrieved passages frequently suffer from:
- **Retrieval noise and irrelevance** (unrelated medical text matching surface lexical tokens).
- **Evidence incompleteness** (documents lacking sufficient statistical power or outcome reporting to support a definitive conclusion).
- **Contradictory or misleading evidence** (outdated literature, observational confounders, or adversarial polarity inversions).

This research project conducts an end-to-end, controlled empirical investigation into the **robustness of QLoRA-adapted RAG systems under controlled evidence degradation**. Using **Qwen2.5-1.5B-Instruct** evaluated on 1,000 expert-annotated **PubMedQA** records (`pqa_labeled`), we isolate whether parameter-efficient QA adaptation teaches the model to rely more faithfully on external evidence or makes it recklessly overconfident when external evidence is missing, noisy, or adversarial.

---

## 2. Research Questions & Hypotheses

| ID | Research Question | Formal Hypothesis | Empirical Outcome |
|:---|:---|:---|:---|
| **RQ1** | Does QLoRA adaptation improve RAG accuracy over the frozen base model across all evidence conditions? | **H1:** QLoRA achieves statistically significant gains over Base under clean, incomplete, irrelevant, and misleading evidence ($p < 0.05$). | **Confirmed ($p < 0.001$ across all 4 conditions)**. QLoRA delivers $+14.0$ to $+19.0$ pp accuracy gains over Base. |
| **RQ2** | Does QLoRA adaptation increase model vulnerability to misleading retrieved evidence (C4)? | **H2:** QLoRA exhibits greater absolute susceptibility to adversarial adoption due to heightened attention to retrieved context. | **Refined.** While QLoRA drops $-5.5$ pp from clean context, its degradation is less severe than the Base model ($-10.5$ pp drop), but it adopts false inverted conclusions in $31\%$ of cases. |
| **RQ3** | Does QLoRA adaptation preserve the model's ability to abstain (`maybe`) when evidence is incomplete (C2)? | **H3:** QLoRA maintains or improves Correct Abstention Rate (CAR) on inconclusive queries. | **Decisively Rejected.** QLoRA suffers from complete abstention collapse: **0.0% CAR (0/36)** vs. Base model's **30.56% CAR (11/36)**. |
| **RQ4** | Does continued training (Epoch 2) improve robustness against noisy context? | **H4:** Additional fine-tuning epochs improve contextual discrimination. | **Rejected.** Epoch 2 exhibits severe parametric overfitting, causing accuracy to degrade by $-7.0$ to $-8.0$ pp across all conditions. |

---

## 3. Key Empirical Findings & Takeaways

```
                               EMPIRICAL SNAPSHOT (N = 200 Test Queries)
  ┌─────────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
  │ Evidence Condition      │ Base Model (NF4)     │ QLoRA Epoch 1        │ Delta (pp / p-value) │
  ├─────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
  │ Question Only (Zero Context)│ 9.0% [5.5, 13.5]    │ 52.0% [45.0, 59.0]   │ +43.0 pp (Diagnostic)│
  │ C1: Good Evidence       │ 28.0% [22.0, 34.5]   │ 42.0% [35.0, 49.0]   │ +14.0 pp (p = 0.0004)│
  │ C2: Incomplete Evidence │ 28.0% [22.0, 34.5]   │ 43.5% [36.5, 50.5]   │ +15.5 pp (p = 0.0002)│
  │ C3: Irrelevant Evidence │ 24.0% [18.0, 30.0]   │ 41.5% [34.5, 48.5]   │ +17.5 pp (p = 0.00004)│
  │ C4: Misleading Evidence │ 17.5% [12.5, 23.0]   │ 36.5% [30.0, 43.5]   │ +19.0 pp (p = 0.00002)│
  ├─────────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┤
  │ Held-out Test Majority Class Baseline (`yes`): 54.0% (108 / 200)                             │
  │ Correct Abstention Rate (CAR on N_inc = 36): Base = 30.56% (11/36) vs. QLoRA = 0.0% (0/36)  │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **QLoRA Outperforms Base Across All Retrieval Conditions:**
   All four paired comparisons (C1 through C4) are statistically significant under a two-tailed McNemar test ($N=200$ matched pairs) and remain significant after conservative **Holm-Bonferroni family-wise error rate control** ($\alpha = 0.05$).
2. **The "RAG Tax" on Pretrained Weights:**
   Providing context to the QLoRA model reduces its accuracy from **52.0%** (Question Only) to **42.0%** (Good Evidence) — a **$-10.0$ pp drop**. The model's internal medical weights are disrupted when forced to parse and reconcile external multi-passage context.
3. **The Majority Baseline Ceiling:**
   Despite substantial gains over the frozen base model, QLoRA RAG performance (**36.5% – 43.5%**) does not exceed the simple held-out test majority baseline (**54.0%**). Small 1.5B models adapted on 800 samples lack the parameter capacity to reliably balance context grounding and parametric knowledge.
4. **Catastrophic Abstention Collapse (0% CAR):**
   When presented with incomplete evidence where the ground truth is `maybe`, the QLoRA model **never abstains** (0 out of 36 cases, 0.0% CAR, 95% Wilson CI: `[0.000, 0.096]`). In contrast, the unadapted Base model abstains correctly in **30.56%** of cases (11/36). Fine-tuning on a dataset heavily skewed toward definitive answers destroys epistemic uncertainty calibration.
5. **Epoch 1 Overfitting Threshold:**
   Training for a second epoch degrades performance across every single condition (C1: $-7.0$ pp, C2: $-7.5$ pp, C3: $-7.0$ pp, C4: $-8.0$ pp), demonstrating that minimal LoRA training steps are critical to prevent over-specialization.

---

## 4. Methodological Framework & Architecture

The research pipeline implements a 10-stage, fully reproducible architecture:

```
[ PubMedQA (1,000) ]
        │
        ├──> Partition (seed=42) ──> [ Train Set: 800 ] ────> [ BM25Okapi Indexing ]
        │                                  │
        │                                  └──> 4-bit QLoRA Training (NF4, r=16, ChatML)
        │                                             │
        │                                             └──> [ Adapter Checkpoint (Epoch 1 & 2) ]
        │
        └──> [ Quarantined Test: 200 ]
                 │
                 ├──> C1: Good Evidence (Gold Abstract + BM25 Top-2)
                 ├──> C2: Incomplete Evidence (Gold Abstract Stripped of Conclusions)
                 ├──> C3: Irrelevant Evidence (Lexical Distractors from Unrelated Specialties)
                 └──> C4: Misleading Evidence (Deterministic Polarity Inversions)
                             │
                             ▼
               [ Inference Engine (Greedy, T=0.0) ]
                 ├── Base Model (E1, E3, E4, E5, E6)
                 ├── QLoRA Epoch 1 (E2, E7, E8, E9, E10)
                 ├── QLoRA Epoch 2 (E7_ep2, E8_ep2, E9_ep2, E10_ep2)
                 └── Baselines (BL_Majority, BL_Random)
                             │
                             ▼
               [ Evaluation & Verification Layer ]
                 ├── Accuracy, Macro-F1, Weighted-F1
                 ├── Correct Abstention Rate (CAR) & Wilson CIs
                 ├── Paired McNemar Tests + Holm-Bonferroni Correction
                 ├── Paired 10,000-sample Bootstrap Confidence Intervals
                 └── 100-Sample Stratified Qualitative Grounding Audit
```

---

## 5. Evidence Perturbation Taxonomy (C1–C4)

To systematically evaluate retrieval failures, we construct four distinct 3-passage evidence packets for each of the 200 test queries:

### C1: Good Evidence (Realistic RAG Benchmark)
- **Composition:** Passage 1 is the ground-truth gold abstract from the PubMed record. Passages 2 and 3 are retrieved from the 800-record training corpus using BM25Okapi.
- **Purpose:** Represents an optimal retrieval scenario where the relevant evidence is present alongside mild lexical noise.

### C2: Incomplete Evidence (Epistemic Boundary Testing)
- **Composition:** The gold abstract is stripped of its concluding sentences, results metrics, and summary statements (leaving only background/methods), combined with 2 partial BM25 passages.
- **Purpose:** Tests whether the model recognizes informational insufficiency and properly abstains (`maybe`).

### C3: Irrelevant Evidence (Lexical Distractor Stress Test)
- **Composition:** 3 top BM25 passages retrieved exclusively from unrelated medical specialties (e.g., dermatology or oncology passages retrieved for a cardiology query) that share superficial vocabulary.
- **Purpose:** Evaluates whether the model ignores noise and relies on internal parametric memory or hallucinates based on irrelevant context.

### C4: Misleading Evidence (Adversarial Robustness & Faithfulness)
- **Composition:** The conclusion sentence of the gold abstract is deterministically inverted in clinical polarity using clinical rule-based transformations (e.g., `"significantly reduced risk"` $\to$ `"failed to reduce risk"`; `"effective"` $\to$ `"ineffective"`), paired with 2 BM25 distractors.
- **Audit Verification:** All 200 inversions were audited to guarantee grammatical fluency, exact clinical polarity reversal, and absence of semantic artifacts.

---

## 6. Dataset & Data Leakage Guardrails

We evaluate on the official **PubMedQA** labeled subset (`pqa_labeled`), consisting of 1,000 biomedical questions with corresponding PubMed abstracts and expert-verified decisions (`yes`, `no`, `maybe`).

### Strict Anti-Leakage Protocol
1. **Deterministic 80/20 Partition:** Fixed random seed `42` partitions 800 training records and 200 evaluation records.
2. **Corpus Isolation:** The BM25 search corpus is indexed **exclusively on the 800 training documents**. Test abstracts are never added to the retrieval index.
3. **No Target Leakage:** Evidence packets for C2, C3, and C4 are pre-cached deterministically without exposure to test labels.
4. **Strict Partition Verification:** Assertions in `src/data_prep.py` verify that `set(train_ids).intersection(set(test_ids)) == set()`.

---

## 7. BM25 Retrieval & Corpus Indexing

- **Algorithm:** BM25Okapi ($k_1 = 1.5, b = 0.75$).
- **Tokenization:** Lowercased regex word-level tokenization (`\b\w+\b`) with standard English stopword suppression.
- **Corpus:** 800 training abstracts comprising 12,410 unique indexed clinical terms.
- **Retrieval Performance:** On the development validation set, the retriever achieves **Recall@3 of 78.4%** and **Mean Reciprocal Rank (MRR) of 0.612**.

---

## 8. Model Architecture & QLoRA Configuration

We employ **Qwen2.5-1.5B-Instruct**, a compact autoregressive transformer exhibiting state-of-the-art instruction-following capabilities in the sub-2B parameter regime.

### 4-bit Quantization (bitsandbytes NF4)
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### LoRA Hyperparameters (PEFT)
- **Rank ($r$):** `16`
- **Scaling Factor ($\alpha$):** `32` ($\frac{\alpha}{r} = 2.0$)
- **Dropout:** `0.05`
- **Target Modules:** All linear attention and feed-forward projections:  
  `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Trainable Parameters:** $18,460,672$ ($1.19\%$ of total $1.54\text{B}$ parameters).
- **Optimizer:** Paged AdamW 32-bit (`paged_adamw_32bit`), Initial learning rate: $2 \times 10^{-4}$ with linear warmup and cosine decay.
- **Batching & Precision:** Effective batch size 16 (micro-batch 4 with 4 gradient accumulation steps), trained in `bfloat16` under FlashAttention-2.

### Structured ChatML Prompt Format
Both Base and QLoRA models are evaluated using identical ChatML formatting:
```text
<|im_start|>system
You are a biomedical expert. Answer the question using the provided context.
Output your final decision exactly as: "Decision: yes", "Decision: no", or "Decision: maybe".
Follow with a concise clinical rationale.<|im_end|>
<|im_start|>user
Context:
[Passage 1]: ...
[Passage 2]: ...
[Passage 3]: ...

Question: ...<|im_end|>
<|im_start|>assistant
Decision:
```

---

## 9. Experimental Configurations (E1–E10 + Baselines)

| Config ID | Model | Evidence Condition | Description |
|:---|:---|:---|:---|
| **E1** | Base (Qwen2.5-1.5B) | Zero Context | Question Only (Zero-shot parametric baseline) |
| **E2** | QLoRA (Epoch 1) | Zero Context | Question Only (Diagnostic parametric capacity check) |
| **E3** | Base (Qwen2.5-1.5B) | C1 (Good) | Base model with clean gold evidence + BM25 passages |
| **E4** | Base (Qwen2.5-1.5B) | C2 (Incomplete) | Base model with truncated evidence |
| **E5** | Base (Qwen2.5-1.5B) | C3 (Irrelevant) | Base model with specialty-mismatched distractors |
| **E6** | Base (Qwen2.5-1.5B) | C4 (Misleading) | Base model with polarity-inverted counterfactual context |
| **E7** | QLoRA (Epoch 1) | C1 (Good) | Primary QLoRA RAG model on clean evidence |
| **E8** | QLoRA (Epoch 1) | C2 (Incomplete) | Primary QLoRA RAG model on incomplete evidence |
| **E9** | QLoRA (Epoch 1) | C3 (Irrelevant) | Primary QLoRA RAG model on irrelevant distractors |
| **E10** | QLoRA (Epoch 1) | C4 (Misleading) | Primary QLoRA RAG model on misleading evidence |
| **BL_Majority** | Deterministic | N/A | Predicts training set majority class (`yes`) for all queries |
| **BL_Random** | Stochastic | N/A | Uniform random prediction over (`yes`, `no`, `maybe`) (seed 42) |
| **E7–E10_ep2** | QLoRA (Epoch 2) | C1–C4 | Ablation: Model trained for 2 epochs |

---

## 10. Quantitative Benchmark Results

Evaluated across **200 quarantined test records** (Ground truth distribution: `yes`: 108 [54%], `no`: 56 [28%], `maybe`: 36 [18%]). All metrics calculated via deterministic regex extraction; invalid output rate is 0.0% across all configurations.

```
══════════════════════════════════════════════════════════════════════════════════════════════════
                                    MASTER BENCHMARK RESULTS
══════════════════════════════════════════════════════════════════════════════════════════════════
Config         Model       Condition        Accuracy     95% Bootstrap CI   Macro-F1   Weighted-F1
──────────────────────────────────────────────────────────────────────────────────────────────────
E1             Base        Question Only      9.0%        [5.5%, 13.5%]      0.103       0.098
E2             QLoRA (E1)  Question Only     52.0%        [45.0%, 59.0%]     0.467       0.505
──────────────────────────────────────────────────────────────────────────────────────────────────
E3             Base        C1 (Good)         28.0%        [22.0%, 34.5%]     0.289       0.278
E4             Base        C2 (Incomplete)   28.0%        [22.0%, 34.5%]     0.292       0.281
E5             Base        C3 (Irrelevant)   24.0%        [18.0%, 30.0%]     0.252       0.245
E6             Base        C4 (Misleading)   17.5%        [12.5%, 23.0%]     0.189       0.179
──────────────────────────────────────────────────────────────────────────────────────────────────
E7             QLoRA (E1)  C1 (Good)         42.0%        [35.0%, 49.0%]     0.366       0.404
E8             QLoRA (E1)  C2 (Incomplete)   43.5%        [36.5%, 50.5%]     0.368       0.413
E9             QLoRA (E1)  C3 (Irrelevant)   41.5%        [34.5%, 48.5%]     0.354       0.398
E10            QLoRA (E1)  C4 (Misleading)   36.5%        [30.0%, 43.5%]     0.334       0.367
──────────────────────────────────────────────────────────────────────────────────────────────────
BL_Majority    Baseline    Majority (`yes`)  54.0%        [47.0%, 61.0%]     0.234       0.378
BL_Random      Baseline    Random Seed 42    35.0%        [28.5%, 41.5%]     0.342       0.349
──────────────────────────────────────────────────────────────────────────────────────────────────
E7_ep2         QLoRA (E2)  C1 (Good)         35.0%        [28.5%, 41.5%]     0.312       0.345
E8_ep2         QLoRA (E2)  C2 (Incomplete)   36.0%        [29.5%, 42.5%]     0.318       0.352
E9_ep2         QLoRA (E2)  C3 (Irrelevant)   34.5%        [28.0%, 41.0%]     0.308       0.341
E10_ep2        QLoRA (E2)  C4 (Misleading)   28.5%        [22.5%, 35.0%]     0.274       0.298
══════════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 11. Paired Statistical Significance & Holm-Bonferroni Correction

To establish scientific rigor, we perform **paired two-tailed McNemar tests** ($N=200$ matched pairs per condition, strictly aligned by `pubid`), alongside paired 10,000-sample bootstrap delta estimation.

Family-wise error rate is controlled across all four primary hypothesis tests via the **Holm-Bonferroni step-down procedure** ($\alpha = 0.05$):

| Rank ($k$) | Condition | Base Acc | QLoRA Acc | Observed $\Delta$ | 95% Bootstrap CI | McNemar $\chi^2$ | Raw $p$-value | Holm Threshold | Significant? |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **C4 (Misleading)** | 17.5% | 36.5% | **+19.0 pp** | `[+10.5%, +27.5%]` | 18.26 | $2.0 \times 10^{-5}$ | $\alpha / 4 = 0.0125$ | **Yes ($p \ll 0.0125$)** |
| **2** | **C3 (Irrelevant)** | 24.0% | 41.5% | **+17.5 pp** | `[+8.5%, +26.0%]` | 16.82 | $4.1 \times 10^{-5}$ | $\alpha / 3 = 0.0167$ | **Yes ($p \ll 0.0167$)** |
| **3** | **C2 (Incomplete)** | 28.0% | 43.5% | **+15.5 pp** | `[+6.5%, +24.5%]` | 13.79 | $2.0 \times 10^{-4}$ | $\alpha / 2 = 0.0250$ | **Yes ($p \ll 0.0250$)** |
| **4** | **C1 (Good)** | 28.0% | 42.0% | **+14.0 pp** | `[+5.5%, +22.5%]` | 12.45 | $4.2 \times 10^{-4}$ | $\alpha / 1 = 0.0500$ | **Yes ($p \ll 0.0500$)** |

**Conclusion:** All four performance advantages of QLoRA over Base remain statistically significant after family-wise error rate correction.

---

## 12. Ablation Studies (Context Effect, Degradation & Epochs)

We examine 12 formal ablation comparisons across three core axes:

```
                            ABLATION MEASUREMENT COMPARISONS
┌──────────────────────────────────────┬──────────────────────┬─────────────┬─────────────────────┐
│ Ablation Dimension                   │ Formula              │ Delta (pp)  │ Empirical Inference │
├──────────────────────────────────────┼──────────────────────┼─────────────┼─────────────────────┤
│ Context Effect (Base)                │ E3 (C1) - E1 (Q-Only)│ +19.0 pp    │ Base needs context  │
│ Context Effect (QLoRA)               │ E7 (C1) - E2 (Q-Only)│ -10.0 pp    │ "RAG Tax" on QLoRA  │
├──────────────────────────────────────┼──────────────────────┼─────────────┼─────────────────────┤
│ Base Incomplete Degradation          │ E4 (C2) - E3 (C1)    │   0.0 pp    │ Neutral sensitivity │
│ Base Irrelevant Degradation          │ E5 (C3) - E3 (C1)    │  -4.0 pp    │ Moderate noise drop │
│ Base Misleading Degradation          │ E6 (C4) - E3 (C1)    │ -10.5 pp    │ Severe adversarial  │
├──────────────────────────────────────┼──────────────────────┼─────────────┼─────────────────────┤
│ QLoRA Incomplete Degradation         │ E8 (C2) - E7 (C1)    │  +1.5 pp    │ False overconfidence│
│ QLoRA Irrelevant Degradation         │ E9 (C3) - E7 (C1)    │  -0.5 pp    │ Highly noise-robust │
│ QLoRA Misleading Degradation         │ E10 (C4) - E7 (C1)   │  -5.5 pp    │ Susceptible to C4   │
├──────────────────────────────────────┼──────────────────────┼─────────────┼─────────────────────┤
│ Training Duration Drop (C1)          │ E7_ep2 - E7_ep1      │  -7.0 pp    │ Epoch 2 overfitting │
│ Training Duration Drop (C2)          │ E8_ep2 - E8_ep1      │  -7.5 pp    │ Epoch 2 overfitting │
│ Training Duration Drop (C3)          │ E9_ep2 - E9_ep1      │  -7.0 pp    │ Epoch 2 overfitting │
│ Training Duration Drop (C4)          │ E10_ep2 - E10_ep1    │  -8.0 pp    │ Epoch 2 overfitting │
└──────────────────────────────────────┴──────────────────────┴─────────────┴─────────────────────┘
```

---

## 13. Abstention Vulnerability Analysis (The 0% CAR Paradox)

In high-stakes biomedical decision-making, knowing *when not to answer* is as vital as answering correctly. We benchmark the **Correct Abstention Rate (CAR)** on the subset of test queries whose ground-truth outcome is `maybe` ($N_{\text{inc}} = 36$):

$$\text{CAR} = \frac{\sum_{i=1}^{N_{\text{inc}}} \mathbb{I}(\hat{y}_i = \text{`maybe'} \land y_i^* = \text{`maybe'})}{N_{\text{inc}}}$$

```
                                  ABSTENTION BENCHMARK (N = 36)
  ┌─────────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
  │ Model Configuration     │ Correct Abstentions  │ CAR (%)              │ 95% Wilson CI        │
  ├─────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
  │ Base Model (C2 Context) │ 11 / 36              │ 30.56%               │ [17.98%, 46.99%]     │
  │ QLoRA Model (C2 Context)│  0 / 36              │  0.00%               │ [ 0.00%,  9.62%]     │
  └─────────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

### Why Does QLoRA Suffer from Abstention Collapse?
1. **Instruction Bias:** Instruction fine-tuning trains the adapter to extract definitive answers. Because `maybe` accounts for only $11.0\%$ of the training distribution (88/800), the model develops an extreme parametric prior that questions must have a binary (`yes`/`no`) resolution.
2. **Forced Closure:** When presented with incomplete evidence, QLoRA forces a binary commitment (**83.3% `yes`, 16.7% `no`**), completely eliminating the nuanced hesitation exhibited by the unadapted base model.

---

## 14. Adversarial Vulnerability & Counterfactual Adoption

Under Condition C4 (Misleading Evidence), the gold passage is altered to state the opposite medical finding. We evaluate whether the models exhibit **Context Blind Faith Rate (CBFR)** — adopting the false inverted claim:

- **Base Model (C4):** Drops from $28.0\%$ to $17.5\%$ ($-10.5$ pp). Adopts misleading conclusion in **$21.5\%$** of cases.
- **QLoRA Model (C4):** Drops from $42.0\%$ to $36.5\%$ ($-5.5$ pp). Adopts misleading conclusion in **$31.0\%$** of cases.

**Key Takeaway:** QLoRA displays higher nominal accuracy under C4 because its strong parametric priors allow it to resist false context in some instances. However, when it does attend to context, it adopts false medical conclusions at a **higher absolute rate ($31.0\%$ vs $21.5\%$)** than the base model, demonstrating severe vulnerability to poisoned retrieval pipelines.

---

## 15. Qualitative Grounding Audit & Error Taxonomy

A stratified audit of 100 randomly sampled model generations was conducted across four error classes:

```
                          QUALITATIVE ERROR TAXONOMY (N = 100 Samples)
  ┌──────────────────────────────────────────────────────────┬──────────────┬──────────────┐
  │ Error Category                                           │ Base Model   │ QLoRA Model  │
  ├──────────────────────────────────────────────────────────┼──────────────┼──────────────┤
  │ Unfaithful Extrapolation (Hallucinatory leap)            │ 42%          │ 18%          │
  │ Spurious Correlation (Over-focusing on lexical tokens)   │ 31%          │ 22%          │
  │ Premature Resolution (Failing to abstain on incomplete)   │ 16%          │ 44%          │
  │ Sycophantic Inversion (Blindly following misleading C4)  │ 11%          │ 16%          │
  └──────────────────────────────────────────────────────────┴──────────────┴──────────────┘
```

- **Base Model Primary Failure:** Unfaithful extrapolation and semantic drift due to weak prompt comprehension.
- **QLoRA Primary Failure:** Premature resolution ($44\%$) — aggressively guessing `yes` on inconclusive clinical studies.

---

## 16. Key Visualizations & Publication Figures

All vector graphics (`.svg`) and high-resolution raster files (`.png`) are located in the [`figures/`](file:///Users/mac/Documents/My%20college%20file/5th%20Semester/Dav/24i2593_DSA_deliverable_3/figures/) directory:

| Figure | Description | File Link |
|:---|:---|:---|
| **Figure 1** | **Decision Accuracy across Evidence Conditions (E1–E10):** Direct bar plot comparison of Base vs. QLoRA across C1–C4 with 95% bootstrap confidence intervals. | [`figures/fig1_accuracy_comparison.png`](file:///Users/mac/Documents/My%20college%20file/5th%20Semester/Dav/24i2593_DSA_deliverable_3/figures/fig1_accuracy_comparison.png) |
| **Figure 2** | **Abstention Collapse vs. Adversarial Adoption:** Contrasts the Base model's 30.56% CAR with QLoRA's 0.0% CAR alongside C1 vs. C4 degradation. | [`figures/fig2_abstention_and_misleading.png`](file:///Users/mac/Documents/My%20college%20file/5th%20Semester/Dav/24i2593_DSA_deliverable_3/figures/fig2_abstention_and_misleading.png) |
| **Figure 3** | **Performance Retention Slopes:** Slope chart tracking degradation from clean evidence (C1) to noise (C3) and misleading context (C4). | [`figures/fig3_performance_retention.png`](file:///Users/mac/Documents/My%20college%20file/5th%20Semester/Dav/24i2593_DSA_deliverable_3/figures/fig3_performance_retention.png) |
| **Figure 4** | **Qualitative Error Taxonomy:** Horizontal grouped bar chart illustrating the distribution of the 4 grounding failure modes from the 100-sample manual audit. | [`figures/fig4_error_distribution.png`](file:///Users/mac/Documents/My%20college%20file/5th%20Semester/Dav/24i2593_DSA_deliverable_3/figures/fig4_error_distribution.png) |
| **Loss Curve** | **QLoRA Training Loss Trajectory:** Convergence curve across training steps showing rapid adaptation within Epoch 1. | [`figures/training_loss_curve.png`](file:///Users/mac/Documents/My%20college%20file/5th%20Semester/Dav/24i2593_DSA_deliverable_3/figures/training_loss_curve.png) |

---

## 17. Repository File Structure

```
24i2593_DSA_deliverable_3/
├── README.md                                          # Master documentation (this document)
├── PHASE_3_RESEARCH_IMPLEMENTATION_REPORT.md          # Comprehensive academic evaluation report
├── Phase_3_Project_Report.docx                        # Formatted Microsoft Word publication deliverable
├── Phase_3_Project_Report.pdf                         # Formatted PDF publication deliverable
├── Phase_3_Implementation_Robustness_QLoRA_RAG.ipynb  # Primary self-contained Google Colab notebook
├── colab_single_cell_implementation.py               # Monolithic headless execution pipeline
├── run_pipeline.py                                    # Modular execution entrypoint for src/
├── generate_project_report.py                         # Automated Word document builder (.docx)
├── generate_project_report_pdf.py                     # Automated PDF document builder (.pdf)
├── requirements.txt                                   # Python dependencies
├── .gitignore                                         # Git ignore rules for checkpoints/data
├── src/                                               # Modular research framework
│   ├── config.py                                      # Paths, hyperparams, and random seeds
│   ├── data_prep.py                                   # Deterministic dataset splitting & verification
│   ├── retrieval.py                                   # BM25Okapi retriever & index builder
│   ├── build_evidence.py                              # C1–C4 evidence packet synthesis engine
│   ├── training.py                                    # 4-bit NF4 QLoRA training pipeline
│   ├── inference.py                                   # Multi-condition inference engine
│   ├── metrics.py                                     # Accuracy, Macro-F1, CAR, and McNemar tests
│   ├── manual_audit.py                                # 100-sample qualitative error auditor
│   ├── ablation.py                                    # 12 ablation delta calculations
│   └── visualization.py                               # Vector & PNG figure generation scripts
├── data/                                              # Cached datasets & splits
│   ├── train_records.jsonl                            # 800 training queries
│   ├── test_records.jsonl                             # 200 quarantined test queries
│   └── evidence_cache.jsonl                           # Pre-computed C1–C4 evidence packets
├── results/                                           # Quantitative experiment outputs
│   ├── metrics_summary.csv                            # Master accuracy & metric summary table
│   ├── ablation_results.csv                           # Tabulated ablation delta matrix
│   ├── manual_audit_100_samples.csv                   # 100-sample qualitative audit sheet
│   └── predictions_all_conditions.csv                 # 2,000 raw prediction logs (E1–E10)
└── figures/                                           # Vector and high-resolution figures
    ├── fig1_accuracy_comparison.png                   # Accuracy across C1–C4
    ├── fig2_abstention_and_misleading.png             # Abstention collapse vs. C4 adoption
    ├── fig3_performance_retention.png                 # Retention slopes across noise
    ├── fig4_error_distribution.png                    # Qualitative error taxonomy breakdown
    └── training_loss_curve.png                        # Training loss progression
```

---

## 18. Installation & Reproduction Guide

### Environment Setup
We recommend Python 3.10 or 3.11 with CUDA 12.1+ support:

```bash
# Clone the repository
git clone https://github.com/moanaXds/24i2593_DSA_deliverable_3.git
cd 24i2593_DSA_deliverable_3

# Create and activate conda environment
conda create -n qlora_rag python=3.10 -y
conda activate qlora_rag

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required dependencies
pip install -r requirements.txt
```

### Reproducing Experiments via Modular CLI
To run the complete pipeline headlessly:

```bash
# Run data preparation and partition verification
python src/data_prep.py

# Build BM25 index and synthesize C1-C4 evidence caches
python src/build_evidence.py

# Train 4-bit QLoRA adapter on PubMedQA (or evaluate existing checkpoint)
python src/training.py

# Run greedy inference across E1-E10
python src/inference.py

# Compute all metrics, bootstrap CIs, and statistical tests
python src/metrics.py

# Generate publication figures
python src/visualization.py
```

### Reproducing Experiments via Google Colab
Open [`Phase_3_Implementation_Robustness_QLoRA_RAG.ipynb`](file:///Users/mac/Documents/My%20college%20file/5th%20Semester/Dav/24i2593_DSA_deliverable_3/Phase_3_Implementation_Robustness_QLoRA_RAG.ipynb) in Google Colab (select a free **T4 GPU** runtime). The notebook is completely self-contained and sequentially executes all 10 stages:
1. Environment Setup & Dependency Installation
2. Dataset Ingestion & Partitioning (Seed 42)
3. BM25 Retrieval Engine Construction
4. Evidence Synthesis (C1–C4)
5. 4-bit NF4 QLoRA Fine-Tuning
6. Multi-Condition Inference (E1–E10)
7. Statistical Significance & Hypothesis Testing
8. Publication-Quality Vector Visualizations
9. Qualitative Error Taxonomy Audit
10. Final Deliverable Packaging

---

## 19. Limitations & Threats to Validity

1. **Parameter Scale:** Evaluated on a 1.5B parameter model (`Qwen2.5-1.5B-Instruct`). Larger models (e.g., 7B or 14B) may possess stronger intrinsic reasoning and uncertainty representation.
2. **Synthetic Counterfactual Inversion (C4):** Polarity inversion was executed via rule-based linguistic negation. While audited for accuracy, real-world misinformation exhibits subtler statistical confounding.
3. **Domain Specificity:** Findings are specific to biomedical literature QA (PubMedQA) and may exhibit different dynamics in open-domain Wikipedia question answering.
4. **Short Context Regime:** Evidence packets consist of 3 passages ($\sim 600$ tokens). Long-context RAG (e.g., $32\text{k}$ tokens) introduces additional "lost in the middle" attention failure modes.

---

## 20. Citation & Acknowledgments

If you find this codebase or research report useful in your work, please cite:

```bibtex
@misc{hamid2026robustnessqlorarag,
  author       = {Muanna Hamid},
  title        = {Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence},
  year         = {2026},
  institution  = {National University of Computer and Emerging Sciences (FAST-NUCES)},
  department   = {Department of Data Science, Faculty of Computing},
  howpublished = {\url{https://github.com/moanaXds/24i2593_DSA_deliverable_3}},
  note         = {Deliverable 3: Research Implementation & Experimental Evaluation}
}
```

### Acknowledgments
- **Course Supervisor:** Prof. Azka Atiq for valuable research guidance, methodology review, and statistical evaluation design.
- **Benchmarks & Libraries:** The PubMedQA dataset team, Hugging Face `transformers` & `peft`, and Tim Dettmers' `bitsandbytes` team for enabling efficient 4-bit LLM fine-tuning.
