"""
Evidence Construction and Caching Engine
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Constructs the cached 3-passage evidence context for each test query across:
- C1: Good supporting evidence
- C2: Incomplete evidence (outcome removed, validation flag)
- C3: Irrelevant evidence (off-target distractors)
- C4: Misleading evidence (counterfactual inversion)

Stores exact passage text, passage IDs, condition labels, and permutation orders in JSONL cache.
"""

import json
import os
import random
from typing import List, Dict, Any
from src.config import ExperimentConfig
from src.data_prep import PubMedQAPreparer
from src.retrieval import BM25Retriever


def build_evidence_cache(train_records: List[Dict[str, Any]], 
                         test_records: List[Dict[str, Any]], 
                         config: ExperimentConfig = None) -> List[Dict[str, Any]]:
    cfg = config or ExperimentConfig()
    os.makedirs(cfg.data_dir, exist_ok=True)
    
    print(f"[Evidence] Initializing controlled BM25 index over {len(train_records)} training abstracts...")
    # Controlled index contains ONLY the 800 training abstracts (Leakage Guardrail)
    train_retriever = BM25Retriever(train_records, cfg)
    
    preparer = PubMedQAPreparer(cfg)
    evidence_cache = []
    c4_audit_records = []
    
    print(f"[Evidence] Generating C1-C4 evidence packages for {len(test_records)} test queries...")
    for idx, test_row in enumerate(test_records):
        pubid = test_row["pubid"]
        question = test_row["question"]
        gold_decision = test_row["final_decision"]
        
        # --- C1: Good Evidence ---
        intact_target = preparer.construct_intact_context(test_row["context"])
        # Top 2 distractors from training corpus
        top_distractors = train_retriever.retrieve(question, top_k=2)
        distractor_passages = [(pid, text) for pid, text, _ in top_distractors]
        
        c1_slots = [("target_intact", pubid, intact_target)] + [
            (f"distractor_{i+1}", pid, txt) for i, (pid, txt) in enumerate(distractor_passages)
        ]
        # Shuffle with fixed seed
        rng_c1 = random.Random(cfg.seed + idx * 17)
        c1_permutation = list(range(len(c1_slots)))
        rng_c1.shuffle(c1_permutation)
        c1_ordered_passages = [c1_slots[i] for i in c1_permutation]
        
        # --- C2: Incomplete Evidence ---
        incomplete_target, is_valid_inc = preparer.construct_incomplete_context(test_row["context"])
        disjoint_passages = train_retriever.retrieve_disjoint(question, top_k=2)
        c2_slots = [("target_incomplete", pubid, incomplete_target)] + [
            (f"disjoint_{i+1}", pid, txt) for i, (pid, txt, _) in enumerate(disjoint_passages)
        ]
        rng_c2 = random.Random(cfg.seed + idx * 19)
        c2_permutation = list(range(len(c2_slots)))
        rng_c2.shuffle(c2_permutation)
        c2_ordered_passages = [c2_slots[i] for i in c2_permutation]
        
        # --- C3: Irrelevant Evidence ---
        irrelevant_passages = train_retriever.retrieve_disjoint(question, top_k=3)
        c3_slots = [
            (f"irrelevant_{i+1}", pid, txt) for i, (pid, txt, _) in enumerate(irrelevant_passages)
        ]
        rng_c3 = random.Random(cfg.seed + idx * 23)
        c3_permutation = list(range(len(c3_slots)))
        rng_c3.shuffle(c3_permutation)
        c3_ordered_passages = [c3_slots[i] for i in c3_permutation]
        
        # --- C4: Misleading Counterfactual Evidence ---
        cf_target, inverted_label = preparer.construct_counterfactual_context(test_row["context"], gold_decision)
        # Use exact same 2 distractors and same slot assignment as C1
        c4_slots = [("target_counterfactual", pubid, cf_target)] + [
            (f"distractor_{i+1}", pid, txt) for i, (pid, txt) in enumerate(distractor_passages)
        ]
        c4_ordered_passages = [c4_slots[i] for i in c1_permutation]  # identical ordering to C1
        
        # Store in audit list for C4
        c4_audit_records.append({
            "pubid": pubid,
            "question": question,
            "gold_decision": gold_decision,
            "inverted_label": inverted_label,
            "original_abstract": intact_target,
            "counterfactual_abstract": cf_target
        })
        
        def format_context_string(passages):
            formatted_blocks = []
            for slot_idx, (role, pid, text) in enumerate(passages, 1):
                formatted_blocks.append(f"[Passage {slot_idx} (ID: {pid})]\n{text}")
            return "\n\n".join(formatted_blocks)

        entry = {
            "pubid": pubid,
            "question": question,
            "gold_decision": gold_decision,
            "C1": {
                "passages": c1_ordered_passages,
                "context_text": format_context_string(c1_ordered_passages),
                "permutation": c1_permutation
            },
            "C2": {
                "passages": c2_ordered_passages,
                "context_text": format_context_string(c2_ordered_passages),
                "permutation": c2_permutation,
                "valid_incomplete": is_valid_inc
            },
            "C3": {
                "passages": c3_ordered_passages,
                "context_text": format_context_string(c3_ordered_passages),
                "permutation": c3_permutation
            },
            "C4": {
                "passages": c4_ordered_passages,
                "context_text": format_context_string(c4_ordered_passages),
                "permutation": c1_permutation,
                "inverted_label": inverted_label
            }
        }
        evidence_cache.append(entry)
        
    # Write to files
    with open(cfg.evidence_cache_path, "w", encoding="utf-8") as f:
        for item in evidence_cache:
            f.write(json.dumps(item) + "\n")
            
    with open(cfg.c4_audit_path, "w", encoding="utf-8") as f:
        for item in c4_audit_records:
            f.write(json.dumps(item) + "\n")
            
    valid_c2_count = sum(1 for e in evidence_cache if e["C2"]["valid_incomplete"])
    print(f"[Evidence] Evidence cache saved to {cfg.evidence_cache_path}")
    print(f"[Evidence] Valid incomplete cases (N_inc for C2): {valid_c2_count} / {len(test_records)}")
    print(f"[Evidence] C4 Counterfactual audit sheet saved to {cfg.c4_audit_path}")
    return evidence_cache


if __name__ == "__main__":
    from src.data_prep import prepare_dataset_pipeline
    train, test = prepare_dataset_pipeline()
    build_evidence_cache(train, test)

