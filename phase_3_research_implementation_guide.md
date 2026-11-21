I have read both documents:

1.  **Phase 1:** Literature Review and Research Gap Identification.
2.  **Phase 2:** Research Methodology Design.

I understand your research now. Let me explain it in the simplest
possible way, without unnecessary research terminology.

# **1. What is our research about?**

**Our research title**

# **Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence**

**In simple words:**

We are investigating whether fine-tuning an AI model makes it better at
handling information retrieved from documents, especially when that
information is incomplete, irrelevant, or misleading.

For example:

Imagine you ask an AI:

*Does a particular medicine improve a patient's condition?*

The AI retrieves a research paper. But what if the retrieved information
is incomplete or contains a misleading conclusion?

Will the AI:

-   Give the correct answer?
-   Follow the misleading information?
-   Recognize that the information is insufficient?
-   Ignore irrelevant information?

We want to investigate whether fine-tuning the AI using QLoRA changes
its behaviour.

Our experiment compares two versions of the same AI model:

-   **Base Model:** Original Qwen2.5-1.5B-Instruct.
-   **QLoRA Model:** The same model after fine-tuning.

We want to find out which one handles bad evidence better.

# **2. What did we do in Phase 1?**

Completed

**Phase 1 was about understanding the research problem and identifying
what existing research has not adequately answered.**

We did four major things.

**1. Identified the problem**

We identified that RAG systems can produce incorrect answers when
retrieved documents contain:

-   Incomplete information.
-   Irrelevant information.
-   Misleading information.
-   Contradictory information.

**2. Studied existing research**

We reviewed 10 research papers from 2025--2026.

We studied approaches such as InstructRAG, RbFT and RAGuard to
understand how researchers have already addressed retrieval problems.

**3. Identified the research gap**

This was the most important part.

We discovered that researchers have already developed specialized
techniques to handle bad evidence.

However, we still wanted to investigate something more specific:

Does ordinary QA-based QLoRA fine-tuning make a small RAG model more
resistant to bad evidence, or does it make the model more likely to
follow misleading information?

**4. Finalized our research questions**

We established three main questions:

-   Does QLoRA improve answer accuracy?
-   Does QLoRA make the model more likely to follow misleading evidence?
-   Does QLoRA improve the model's ability to reject irrelevant
    information and abstain when evidence is incomplete?

**In one sentence:** Phase 1 established what problem we were
investigating, what previous researchers had done, and what we wanted to
find out.

# **3. What did we do in Phase 2?**

Completed

Phase 2 converted our research idea into an actual experimental plan.

We decided exactly how we would conduct the research.

Here is what we finalized:

  Component            Our decision
  -------------------- -----------------------------
  Dataset              PubMedQA
  Dataset size         1,000 questions
  Training data        800 questions
  Testing data         200 questions
  Base model           Qwen2.5-1.5B-Instruct
  Fine-tuning          4-bit QLoRA
  Retriever            BM25
  Retrieved passages   Top 3
  Epochs               1
  Random seed          42
  Environment          Google Colab with NVIDIA T4

We also designed four experimental conditions.

## **Our four evidence conditions**

C1

**Good Evidence**

The AI receives the correct research information needed to answer the
question.

Expected behaviour: Give a correct, evidence-supported answer.

C2

**Incomplete Evidence**

The AI receives a research passage, but the important result or finding
is missing.

Expected behaviour: Recognize insufficient evidence and abstain.

C3

**Irrelevant Evidence**

The AI receives unrelated biomedical research passages.

Expected behaviour: Ignore irrelevant information.

C4

**Misleading Evidence**

The AI receives a passage containing an intentionally inverted research
finding.

Expected behaviour: Avoid blindly following the misleading claim.

We also designed:

-   How to preprocess the dataset.
-   How to divide training and testing data.
-   How to fine-tune the model.
-   How to retrieve documents using BM25.
-   How to compare the Base and QLoRA models.
-   Which evaluation metrics to calculate.
-   Which graphs to generate.
-   How to conduct statistical analysis.
-   How to manually inspect model responses.

**In one sentence:** Phase 2 established the complete blueprint for
conducting our experiments.

Important distinction: We designed the experiment. That does not mean we
have already trained the model or generated experimental results.

# **4. What do we have to do in Phase 3?**

Current phase

Now comes the actual implementation.

Your professor wants evidence that we have executed the methodology we
designed in Phase 2.

There are seven requirements. Let me translate each into practical
tasks.

  ------------------------------------------------------------------------
  Requirement      What we actually need to do
  ---------------- -------------------------------------------------------
  1\.              Build and execute our Base and QLoRA models.
  Implementation   

  2\. Dataset and  Load PubMedQA, clean it, split it into 800 training and
  preprocessing    200 testing examples, and prepare evidence passages.

  3\. Baseline     Run the original model and compare it with the
  evaluation       fine-tuned model.

  4\.              Generate graphs comparing accuracy, robustness and
  Visualization    other metrics.

  5\. Existing     Compare our findings with the approaches and results
  research         reported in the papers we reviewed.
  comparison       

  6\. Ablation     Remove or modify experimental components and observe
  studies          how performance changes.

  7\. Results and  Explain what happened, why it happened, and whether our
  discussion       research questions were answered.
  ------------------------------------------------------------------------

## **What exactly should we implement?**

Our Phase 2 experimental matrix contains 10 configurations.

EXPERIMENTAL MATRIX

  ID    Model   Experiment
  ----- ------- ---------------------
  E1    Base    Question only
  E2    QLoRA   Question only
  E3    Base    Good evidence
  E4    Base    Incomplete evidence
  E5    Base    Irrelevant evidence
  E6    Base    Misleading evidence
  E7    QLoRA   Good evidence
  E8    QLoRA   Incomplete evidence
  E9    QLoRA   Irrelevant evidence
  E10   QLoRA   Misleading evidence

The purpose is to compare the two models under identical experimental
conditions.

For example:

-   Does Base perform better on good evidence?
-   Does QLoRA perform better on incomplete evidence?
-   Which model follows misleading evidence more frequently?
-   Which model ignores irrelevant information more effectively?

These results will form the core of our research.

# **5. Which graphs and metrics do we need?**

Our Phase 2 already specifies the evaluation metrics.

We don't need to invent new ones.

  -----------------------------------------------------------------------
  Metric       What it measures
  ------------ ----------------------------------------------------------
  Accuracy     How many answers are correct?

  Macro-F1     How balanced is performance across yes, no and maybe?

  Recall@3     Does retrieval find the correct research abstract?

  UCAR         How often does the model adopt the misleading claim?

  CAR          How often does it correctly abstain when evidence is
               incomplete?

  IRR          How effectively does it reject irrelevant information?

  Grounding    Are the model's factual claims supported by the supplied
  Rate         evidence?

  Latency      How much time does the model take to generate an answer?
  -----------------------------------------------------------------------

We should generate these main visualizations:

[Frontiers \| Ensemble learning approach with explainable AI for
improved heart disease
prediction](https://images.openai.com/static-rsc-4/Wf9jhzUKJZaAd3oJAG355soPKnaWihSJISrK9RyQ5hq3yiNBwlllVu7QHF3AI7m2HLxX7qUcNz-a4FKI08AdFSQLnBFc6OasyAc8azddo8khgRCHaEa5FQsJTe1TcoYsMpny0akPt-H6_CIS0mGv4op6sjTWmZP7IvOrQNrDr3E?purpose=inline)

**1. Accuracy comparison**

Base vs QLoRA across four conditions.

[Detecting Coordinated Inauthentic Behavior Under Symmetry Breaking: An
Adaptive Memory-Guided Causal Framework\[v1\] \|
Preprints.org](https://images.openai.com/static-rsc-4/1UxwXkEctnaPznJeVgeyIHZUyz5Au2VAO3jlmeyvQwSRTJIKmenwa33cjlqe3ZhvY7ORMfYrS4yvrBqvHCzSI5NaaUTK1VX9s2GTOmKqsCmbM4Ef22CEyF9iYbm5PjqO1niZTSz9IeMUd1JiIw1VUj_Sq3G21jsspqREvJ6TmAI?purpose=inline)

**2. Abstention and misleading evidence**

CAR and UCAR/CBFR.

[\[ICLR 2021\] Dataset Condensation with Gradient Matching \| Awesome
Reviews](https://images.openai.com/static-rsc-4/Q7qo8XM3jjpKv89YkaglPpPGguEEtE-h5ZwyYJW1zmheW15_rdD6SOvF5YkbHiVod1xZ76UbYDx-HXW4KayzBpYb3aSBTtRIsTYy-VU0HBUXNalUpdCxOCyAE3WnQEeNARQqOYKBuseJPdtRNW4PrnzWos9yX4A2Xqjg7rYvPXo?purpose=inline)

**3. Performance retention**

How performance changes across evidence conditions.

[Accelerating Deep Learning Inference: A Comparative Analysis of Modern
Acceleration
Frameworks](https://images.openai.com/static-rsc-4/dNkvzwF5YoFZjjPTunvbVopJm1NsIomtqoNY1w1nT7YseG8EcXZmobJ9-eS-rduUvpbgE4nA98vnxpZ15EHyO5rC1AS5SJXctbEPuWoaVB0ekmB4vezY8V-ozL7qwDWm8drsNRAudghupyQOA3QquMwSyD-LI9Bw6BMFvsW1e_8?purpose=inline)

**4. Computational performance**

Inference time and computational requirements.

These are illustrative chart types, not our actual experimental results.

# **6. Is Phase 3 doable in 4--5 hours?**

Full implementation: Not realistically

I will be direct: **Completing the entire Phase 3 methodology properly
in 4--5 hours is not realistic.**

Look at what we planned in Phase 2:

-   Fine-tune a 1.5B parameter model using QLoRA.
-   Process 800 training examples.
-   Prepare four different evidence conditions.
-   Generate up to 1,600 + 2N_inc model outputs.
-   Evaluate all 10 configurations.
-   Perform a manual audit of 100 outputs.
-   Calculate confidence intervals using 10,000 bootstrap samples.
-   Perform statistical tests.
-   Generate visualizations.
-   Compare findings with existing research.

This is a substantial experimental project, not a five-hour coding
exercise.

However, there is a difference between completing the full research
experiment and producing a credible, limited Phase 3 implementation.

**A smaller, honest implementation is doable in 4--5 hours**, provided
we have access to Google Colab GPU and the required packages and models
load successfully.

## **My recommended 4--5-hour execution plan**

## **Five-hour implementation sprint**

A realistic reduced-scope plan

1.  **First 30 minutes: Dataset preparation**
    -   Download PubMedQA.
    -   Load the 1,000 labeled examples.
    -   Create the deterministic 800/200 split.
    -   Inspect and preprocess the dataset.
2.  **Next 60--90 minutes: Model setup and QLoRA**
    -   Load Qwen2.5-1.5B-Instruct.
    -   Configure 4-bit quantization.
    -   Prepare the QLoRA adapter.
    -   Attempt a one-epoch training run. Actual duration depends
        heavily on GPU availability and setup.
3.  **Next 60 minutes: Evidence construction and evaluation**
    -   Prepare the four evidence conditions.
    -   Run a smaller, consistent test subset.
    -   Compare Base and QLoRA predictions.
4.  **Next 45 minutes: Metrics and visualizations**
    -   Calculate accuracy and Macro-F1.
    -   Calculate applicable robustness metrics.
    -   Generate comparison charts.
5.  **Final 45--60 minutes: Analysis and documentation**
    -   Compare with existing papers.
    -   Conduct a small ablation experiment if time permits.
    -   Document limitations.
    -   Write results and discussion.

This is a proposed schedule, not a guarantee. If model setup or training
fails, the full schedule will not fit.

# **7. What can we reduce without destroying our research?**

Here is where I would make some practical decisions.

  -----------------------------------------------------------------------
  Component                    Recommendation
  ---------------------------- ------------------------------------------
  Two model variants           Must keep

  PubMedQA dataset             Must keep

  Good evidence                Must keep

  Incomplete evidence          Must keep

  Irrelevant evidence          Must keep

  Misleading evidence          Keep if possible

  Accuracy and Macro-F1        Must keep

  Grounding analysis           Keep a small manual sample

  Latency                      Measure if possible

  Bootstrap and statistical    Reduce or defer
  tests                        

  Dense retrieval              Remove; we already selected BM25

  Full 100-output manual audit Reduce and disclose

  Ablation experiments         Perform at least one small, meaningful
                               ablation
  -----------------------------------------------------------------------

One important point: Do not fabricate experimental results, statistical
significance, training time, or model performance to make the report
look complete.

If we only test 40 questions, report 40 questions. If the misleading
condition is not ready, disclose it. If QLoRA training fails, do not
pretend it succeeded.

# **8. My final assessment**

**What is achievable in 4--5 hours?**

Dataset preparation

Achievable

Base model evaluation

Achievable

QLoRA fine-tuning

Uncertain

Four-condition evaluation

Reduced scope

Graphs and metrics

Achievable

Full statistical evaluation

Unrealistic

Full manual audit and ablations

Reduced scope

## **Before we start, I need one important answer.**

**Have you already created any Google Colab notebook or written any code
for this research?**

**1. Current implementation status**

Nothing yet. We only have Phase 1 and Phase 2 documents.

I have created a Colab notebook but haven't implemented anything.

I have some code or dataset preparation already done.

I have a working model or partial experimental results.

**2. Google Colab GPU availability**

I can access a T4 GPU.

I have access to another GPU.

I haven't checked yet.

I only have my local MacBook.

Submit

Once I know this, we can choose the fastest implementation route and
start building the actual experiments instead of spending more time
discussing the methodology.
