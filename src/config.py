"""
Configuration file for Phase 3 Research Implementation:
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ExperimentConfig:
    # Random Seed
    seed: int = 42

    # Dataset Parameters
    dataset_name: str = "pubmed_qa"
    dataset_subset: str = "pqa_labeled"
    total_records: int = 1000
    train_size: int = 800
    test_size: int = 200
    valid_decisions: tuple = ("yes", "no", "maybe")

    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    results_dir: str = os.path.join(base_dir, "results")
    figures_dir: str = os.path.join(base_dir, "figures")
    adapter_dir: str = os.path.join(base_dir, "saved_qlora_adapter")
    
    split_manifest_path: str = os.path.join(data_dir, "split_manifest.json")
    train_data_path: str = os.path.join(data_dir, "train_records.jsonl")
    test_data_path: str = os.path.join(data_dir, "test_records.jsonl")
    evidence_cache_path: str = os.path.join(data_dir, "evidence_cache.jsonl")
    c4_audit_path: str = os.path.join(data_dir, "c4_counterfactual_audit.jsonl")
    predictions_path: str = os.path.join(results_dir, "predictions_all_conditions.csv")
    metrics_path: str = os.path.join(results_dir, "metrics_summary.csv")
    manual_audit_path: str = os.path.join(results_dir, "manual_audit_100_samples.csv")

    # BM25 Retrieval Parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    bm25_top_k: int = 3
    token_pattern: str = r"\b[a-zA-Z]{2,}\b"

    # Model Parameters
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    quant_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "float16"

    # PEFT / LoRA Parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Training Parameters
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    num_train_epochs: float = 1.0        # 100 steps for 800 examples
    max_seq_length: int = 512
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    save_strategy: str = "no"

    # Inference Protocol
    max_new_tokens: int = 128
    context_budget: int = 1024
    do_sample: bool = False  # Greedy decoding

    # 10 Experimental Matrix Configurations
    experimental_matrix: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "E1": {"model": "Base", "condition": "None", "name": "Question Only"},
        "E2": {"model": "QLoRA", "condition": "None", "name": "Question Only (Diagnostic)"},
        "E3": {"model": "Base", "condition": "C1", "name": "Good Evidence"},
        "E4": {"model": "Base", "condition": "C2", "name": "Incomplete Evidence"},
        "E5": {"model": "Base", "condition": "C3", "name": "Irrelevant Evidence"},
        "E6": {"model": "Base", "condition": "C4", "name": "Misleading Evidence"},
        "E7": {"model": "QLoRA", "condition": "C1", "name": "Good Evidence"},
        "E8": {"model": "QLoRA", "condition": "C2", "name": "Incomplete Evidence"},
        "E9": {"model": "QLoRA", "condition": "C3", "name": "Irrelevant Evidence"},
        "E10": {"model": "QLoRA", "condition": "C4", "name": "Misleading Evidence"},
    })

    # Abstention Keyword Screening Proxy for CAR (Condition C2)
    abstention_proxies: List[str] = field(default_factory=lambda: [
        "insufficient evidence",
        "cannot be determined",
        "does not report",
        "unclear from the provided",
        "not enough information",
        "cannot conclude",
        "insufficient information",
        "not reported",
        "data are lacking",
        "not provided in the context",
        "no conclusive evidence",
        "unable to determine",
        "evidence does not specify"
    ])

    # System Prompt for ChatML
    system_prompt: str = (
        "You are a biomedical expert reader. Answer the given clinical or biomedical research question "
        "faithfully based on the provided evidence passages. If the evidence is insufficient or contradictory, "
        "state that clearly. Conclude your answer strictly with 'Final Decision: yes', 'Final Decision: no', "
        "or 'Final Decision: maybe'."
    )

