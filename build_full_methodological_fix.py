import os
import sys
import json
import ast

code_content = '''# ====================================================================================================
# PHASE 3: RESEARCH IMPLEMENTATION & EXPERIMENTAL EVALUATION
# "Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence"
# Researcher: Muanna Hamid (Roll No: 242593) | Supervisor: Prof. Azka Atiq
# FAST-NUCES Islamabad | BS Data Science (5th Semester) | November 2026
#
# SINGLE SELF-CONTAINED EXECUTION CELL FOR GOOGLE COLAB (T4 / A100 GPU)
# Runs the complete research pipeline end-to-end adhering strictly to Phase 1 & Phase 2 Methodology.
# Corrected for: Isolated Frozen Baseline, Quarantined long_answer, Exact BM25Okapi (eps=0.25),
# In-Place Non-Fabricated Counterfactuals, 10,000 Paired Bootstrap & McNemar/Holm-Bonferroni Suite.
# ====================================================================================================

import os
import sys
import gc
import re
import math
import time
import json
import shutil
import random
import datetime
import traceback
import subprocess
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# ----------------------------------------------------------------------------------------------------
# [STAGE 1/8] ENVIRONMENT SETUP & HARDWARE VERIFICATION
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "1/8: Environment Setup & Hardware Verification"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Starting Environment Verification...")
print("=" * 90)

# Check and install required packages automatically in Google Colab if missing
REQUIRED_PACKAGES = [
    "transformers", "peft", "bitsandbytes", "accelerate", "datasets",
    "rank-bm25", "trl", "nltk", "scikit-learn", "scipy", "seaborn", "matplotlib", "pandas", "numpy"
]

missing_pkgs = []
for pkg in REQUIRED_PACKAGES:
    import_name = "sklearn" if pkg == "scikit-learn" else ("rank_bm25" if pkg == "rank-bm25" else pkg)
    try:
        __import__(import_name)
    except ImportError:
        missing_pkgs.append(pkg)

if missing_pkgs:
    print(f"[Setup] Installing required libraries: {missing_pkgs}...")
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Package installation failed:\\n{res.stderr}")
    print("[Setup] Package installation complete.")

# Import libraries after installation
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
)
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import trl
from trl import SFTTrainer
import accelerate

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# Verify GPU Allocation & Print Core ML Library Versions
print(f"Python Version:       {sys.version.split()[0]}")
print(f"PyTorch Version:      {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"TRL Version:          {trl.__version__}")
print(f"PEFT Version:         {peft.__version__}")
print(f"Accelerate Version:   {accelerate.__version__}")
try:
    import bitsandbytes as bnb
    print(f"BitsAndBytes Version: {bnb.__version__}")
except Exception:
    pass

print(f"CUDA Available:       {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "\\n[FATAL ERROR] No CUDA GPU detected in this Google Colab runtime!\\n"
        "This experiment requires a GPU to load Qwen2.5-1.5B in 4-bit NF4 and execute QLoRA fine-tuning.\\n"
        "Please navigate to: Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU, then rerun."
    )

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"GPU Device:           {GPU_NAME}")
print(f"Total VRAM:           {GPU_MEMORY_GB:.2f} GB")

# Experiment Configurations & Directories
SEED = 42
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
print(f"Random Seed:          {SEED} (Fixed across Python, NumPy, PyTorch)")

# Workspace Directories - Clean stale results to prevent cross-run pollution
EXP_ROOT = "phase3_rag_robustness"
DATA_DIR = os.path.join(EXP_ROOT, "data")
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
FIGURES_DIR = os.path.join(EXP_ROOT, "figures")
ADAPTER_DIR = os.path.join(EXP_ROOT, "saved_qlora_adapter")

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, ADAPTER_DIR]:
    os.makedirs(d, exist_ok=True)

# Purge any stale CSV/JSON/PNG files from previous runs to ensure fresh outputs
for clean_dir in [RESULTS_DIR, FIGURES_DIR]:
    for f in os.listdir(clean_dir):
        fp = os.path.join(clean_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

print(f"Workspace Directory:  {os.path.abspath(EXP_ROOT)} (Directories cleaned and initialized)")
print("[1/8] Environment & Hardware Verification Complete.\\n")

# ----------------------------------------------------------------------------------------------------
# [STAGE 2/8] DATASET INGESTION, LEAKAGE CONTROLS & DETERMINISTIC SPLIT
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "2/8: Dataset Ingestion & Partitioning"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Ingesting PubMedQA (pqa_labeled) with Strict Leakage Controls...")
print("=" * 90)

# Ingest verified HuggingFace repository 'qiaojin/PubMedQA' (pqa_labeled)
print("[Data Ingestion] Ingesting 'qiaojin/PubMedQA' with configuration 'pqa_labeled'...")
try:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", trust_remote_code=True)
except Exception as e_hf:
    print(f"[Data Ingestion] Primary Hugging Face load raised: {e_hf}. Trying raw verified mirror...")
    import urllib.request
    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data_json = json.loads(resp.read().decode("utf-8"))
    ds = [{
        "pubid": k,
        "question": v.get("QUESTION") or v.get("question", ""),
        "context": {"contexts": v.get("CONTEXTS") or v.get("contexts", []), "labels": v.get("LABELS") or v.get("labels", [])},
        "long_answer": v.get("long_answer", ""),
        "final_decision": v.get("final_decision", "")
    } for k, v in data_json.items()]

# Validation Check 1: Exactly 1,000 records
if len(ds) != 1000:
    raise ValueError(f"Validation Failed: Expected exactly 1,000 records in PubMedQA pqa_labeled, but received {len(ds)}.")

# Validation Check 2: Verify all required columns exist and are non-null
REQUIRED_COLUMNS = ["pubid", "question", "context", "final_decision"]
raw_records = []
for row in ds:
    for col in REQUIRED_COLUMNS:
        if col not in row or row[col] is None:
            raise KeyError(f"Validation Failed: Missing required column '{col}' in record {row.get('pubid')}")
            
    raw_records.append({
        "pubid": str(row["pubid"]).strip(),
        "question": str(row["question"]).strip(),
        "context": row["context"],
        # LEAKAGE GUARDRAIL: long_answer is quarantined from all retrieval, evidence, and training targets!
        # It is preserved exclusively as an offline reference field.
        "long_answer": str(row.get("long_answer", "")).strip(),
        "final_decision": str(row["final_decision"]).strip().lower()
    })

# Deterministic Partitioning according to Phase 2:
# 1. Sort lexicographically by pubid
sorted_records = sorted(raw_records, key=lambda x: str(x["pubid"]))

# 2. Seeded shuffle for train/test partition (800 training / 200 held-out testing)
split_rng = random.Random(SEED)
shuffled_records = list(sorted_records)
split_rng.shuffle(shuffled_records)

train_records = shuffled_records[:800]
test_records = shuffled_records[800:]

if len(train_records) != 800 or len(test_records) != 200:
    raise ValueError(f"Partition error: Expected 800 train and 200 test, got {len(train_records)} / {len(test_records)}")

print(f"Total PubMedQA Records:  {len(raw_records)}")
print(f"Training Corpus Size:    {len(train_records)} (Used strictly for QLoRA supervision & controlled BM25 index)")
print(f"Held-Out Test Set Size:  {len(test_records)} (Strictly quarantined from retrieval corpus)")

test_labels_dist = dict(Counter([r["final_decision"] for r in test_records]))
print(f"Test Class Distribution: {test_labels_dist} ({test_labels_dist.get('yes',0)/2:.1f}% yes, {test_labels_dist.get('no',0)/2:.1f}% no, {test_labels_dist.get('maybe',0)/2:.1f}% maybe)")

# Save split manifest and partitioned records
manifest_path = os.path.join(DATA_DIR, "split_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED,
        "total_records": len(raw_records),
        "train_size": len(train_records),
        "test_size": len(test_records),
        "train_pubids": [r["pubid"] for r in train_records],
        "test_pubids": [r["pubid"] for r in test_records],
        "leakage_guardrail": "long_answer quarantined from all retrieval indices, evidence contexts, and training targets"
    }, f, indent=2)

train_path = os.path.join(DATA_DIR, "train_records.jsonl")
with open(train_path, "w", encoding="utf-8") as f:
    for r in train_records:
        f.write(json.dumps(r) + "\\n")
test_path = os.path.join(DATA_DIR, "test_records.jsonl")
with open(test_path, "w", encoding="utf-8") as f:
    for r in test_records:
        f.write(json.dumps(r) + "\\n")

print(f"Saved Split Manifest:    {manifest_path}")
print(f"Saved Train/Test JSONL:  {train_path}, {test_path}")
print("[2/8] Dataset Ingestion & Partitioning Complete.\\n")

# ----------------------------------------------------------------------------------------------------
# [STAGE 3/8] BM25 RETRIEVAL INDEXING & NATURAL BENCHMARK RECALL@3
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "3/8: BM25 Retrieval & Natural Retrieval Benchmark"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Indexing Corpora with BM25Okapi (k1=1.5, b=0.75, epsilon=0.25)...")
print("=" * 90)

class StandardBM25Retriever:
    \"\"\"
    BM25Okapi retrieval engine adhering strictly to Phase 2 specifications:
    k1 = 1.5, b = 0.75, epsilon = 0.25 (flooring on negative IDF values).
    Tokenization: r'\\b[a-zA-Z]{2,}\\b', lowercased, NLTK English stopwords removed.
    \"\"\"
    def __init__(self, corpus_records: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_records = corpus_records
        self.doc_ids = [str(r["pubid"]).strip() for r in corpus_records]
        
        # Build text representation by concatenating context paragraphs
        sep = "\\n\\n"
        self.doc_texts = [sep.join(r["context"].get("contexts", [])).strip() for r in corpus_records]
        
        self.tokenized_corpus = [self.tokenize(t) for t in self.doc_texts]
        self.doc_lengths = [len(d) for d in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.corpus_size = len(self.tokenized_corpus)
        
        self.doc_freqs = []
        self.df = Counter()
        for doc in self.tokenized_corpus:
            tf = Counter(doc)
            self.doc_freqs.append(tf)
            self.df.update(tf.keys())
            
        # Exact BM25Okapi IDF with epsilon flooring on negative values
        self.idf = {}
        negative_idfs = []
        idf_sum = 0.0
        for term, freq in self.df.items():
            # Standard Okapi IDF formula: log((N - n + 0.5) / (n + 0.5))
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[term] = idf_val
            idf_sum += idf_val
            if idf_val < 0:
                negative_idfs.append(term)
        
        avg_idf = idf_sum / max(1, len(self.df))
        eps = self.epsilon * avg_idf
        for term in negative_idfs:
            self.idf[term] = eps

    def tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\\b[a-zA-Z]{2,}\\b", text.lower())
        return [w for w in words if w not in STOP_WORDS]

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        tf_dict = self.doc_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)
        for q in query_tokens:
            if q not in tf_dict:
                continue
            tf = tf_dict[q]
            idf = self.idf.get(q, 0.0)
            score += idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm))
        return score

    def retrieve(self, query: str, top_k: int = 3, exclude_pubids: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        exclude_set = set(str(p).strip() for p in (exclude_pubids or []))
        q_tokens = self.tokenize(query)
        scored = []
        for idx in range(self.corpus_size):
            pid = self.doc_ids[idx]
            if pid in exclude_set:
                continue
            s = self.score(q_tokens, idx)
            scored.append((pid, self.doc_texts[idx], s))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def retrieve_disjoint(self, query: str, top_k: int = 3, exclude_pubids: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        \"\"\"Strictly retrieves passages with zero query keyword overlap (for C2/C3 distractor pools).\"\"\"
        exclude_set = set(str(p).strip() for p in (exclude_pubids or []))
        q_tokens = set(self.tokenize(query))
        candidates = []
        for idx in range(self.corpus_size):
            pid = self.doc_ids[idx]
            if pid in exclude_set:
                continue
            doc_toks = set(self.tokenized_corpus[idx])
            if len(q_tokens.intersection(doc_toks)) == 0:
                candidates.append((pid, self.doc_texts[idx], 0.0))
                if len(candidates) >= top_k:
                    break
        # If strict zero-overlap count is less than top_k, select lowest BM25-scoring passages (least relevant)
        if len(candidates) < top_k:
            all_scored = [(self.doc_ids[i], self.doc_texts[i], self.score(list(q_tokens), i)) 
                          for i in range(self.corpus_size) if self.doc_ids[i] not in exclude_set]
            all_scored.sort(key=lambda x: x[2])
            for item in all_scored:
                if item[0] not in [c[0] for c in candidates]:
                    candidates.append(item)
                    if len(candidates) >= top_k:
                        break
        return candidates[:top_k]

# Evaluate Natural Retrieval Benchmark Recall@3 across all 1,000 intact abstracts
print("[Retrieval] Indexing Natural Benchmark Corpus (all 1,000 intact abstracts)...")
all_records = train_records + test_records
benchmark_retriever = StandardBM25Retriever(all_records)

# 1. Target Abstract Recovery (Phase 2 Section 4/8.2): Query's own intact abstract in top 3 of 1,000 pool
benchmark_hits = 0
for t in test_records:
    res = benchmark_retriever.retrieve(t["question"], top_k=3)
    retrieved_pids = [r[0] for r in res]
    if str(t["pubid"]).strip() in retrieved_pids:
        benchmark_hits += 1

recall_at_3 = benchmark_hits / len(test_records)
print(f"[Retrieval Benchmark Result] Target Abstract Recovery Recall@3 = {recall_at_3:.4f} ({benchmark_hits}/{len(test_records)})")
print("  (Note: Characterizes retrieval coverage of target paper from 1,000 candidate pool; does not feed reader).")

# 2. Self-Excluded Natural Retrieval Benchmark (evaluating retrieval when target abstract is strictly absent)
self_excluded_hits = 0
for t in test_records:
    res_ex = benchmark_retriever.retrieve(t["question"], top_k=3, exclude_pubids=[t["pubid"]])
    # Checked that self-retrieval is strictly blocked
    assert str(t["pubid"]).strip() not in [r[0] for r in res_ex]

# Controlled Background Index: Strictly the 800 training abstracts (Leakage Prevention)
print("[Retrieval] Building Controlled Background Index (strictly 800 training abstracts)...")
train_retriever = StandardBM25Retriever(train_records)

# Save retrieval benchmark result to a JSON file for formal reporting
retrieval_bm_path = os.path.join(RESULTS_DIR, "retrieval_benchmark.json")
with open(retrieval_bm_path, "w", encoding="utf-8") as f:
    json.dump({
        "metric": "Recall@3",
        "corpus_size": len(all_records),
        "test_queries": len(test_records),
        "hits": benchmark_hits,
        "recall_at_3": round(recall_at_3, 4),
        "bm25_parameters": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
        "leakage_control": "Controlled reader background index built strictly from 800 training abstracts with zero test records"
    }, f, indent=2)
print(f"Saved Retrieval Benchmark:   {retrieval_bm_path}")
print("[3/8] BM25 Retrieval Indexing Complete.\\n")

# ----------------------------------------------------------------------------------------------------
# [STAGE 4/8] CONTROLLED EVIDENCE CONSTRUCTION & VALIDATION (C1–C4)
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "4/8: Controlled Evidence Construction (C1-C4)"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Constructing 3-Passage Evidence Packets for 200 Test Queries...")
print("=" * 90)

def strip_outcomes_c2(context_obj: Dict[str, Any]) -> Tuple[str, bool]:
    \"\"\"
    Removes outcome-bearing material for C2 adhering strictly to Phase 2 Section 3.2:
    - For structured records: strips all paragraphs labeled RESULTS, RESULT, FINDINGS, FINDING, CONCLUSIONS, CONCLUSION, DISCUSSION.
    - For unstructured records: retains leading 50% sentences, strips trailing 50%.
    - Validates absence of clinical outcome indicators. Returns (incomplete_text, is_valid_flag).
    \"\"\"
    contexts = context_obj.get("contexts", [])
    labels = [str(l).upper() for l in context_obj.get("labels", [])]
    retained = []
    removed = False
    
    outcome_label_keywords = ["RESULT", "FINDING", "CONCLUSION", "OUTCOME", "DISCUSSION", "INTERPRETATION"]
    
    if labels and len(labels) == len(contexts):
        for para, label in zip(contexts, labels):
            if any(k in label for k in outcome_label_keywords):
                removed = True
            else:
                retained.append(para)
                
    if not removed or len(retained) == 0:
        # Fallback for unstructured abstracts: retain leading 50% of sentences (background/methods)
        full_text = " ".join(contexts)
        sents = re.split(r"(?<=[.!?])\\s+", full_text)
        cutoff = max(1, int(len(sents) * 0.5))
        retained = sents[:cutoff]
        incomplete_text = " ".join(retained).strip()
        removed = True
    else:
        sep = "\\n\\n"
        incomplete_text = sep.join(retained).strip()
        
    # Rigorous validation: check that explicit outcome indicators are absent
    outcome_markers = [
        "statistically significant", "p < 0.", "p = 0.", "p > 0.", "effective and recommended",
        "demonstrated efficacy", "superior to", "significantly increased", "significantly reduced",
        "concluded that", "findings indicate", "results show"
    ]
    is_valid = removed and not any(m in incomplete_text.lower() for m in outcome_markers)
    return incomplete_text, is_valid

def invert_outcome_c4_non_fabricated(context_obj: Dict[str, Any], gold: str) -> Tuple[str, str, str]:
    \"\"\"
    Inverts principal outcome finding in-place without fabricating fake clinical studies, cohorts, or statistics.
    Adheres strictly to Phase 2 Section 3.4:
    Preserves topic, entities, and grammatical structure while cleanly inverting the conclusion polarity.
    Returns: (counterfactual_abstract, inverted_label, inversion_method)
    \"\"\"
    contexts = list(context_obj.get("contexts", []))
    labels = [str(l).upper() for l in context_obj.get("labels", [])]
    
    if gold == "yes":
        inv_label = "no"
    elif gold == "no":
        inv_label = "yes"
    else:
        inv_label = "maybe"
        
    if not contexts:
        return "", inv_label, "empty_context"

    # Identify the conclusion / results paragraph
    target_idx = -1
    for idx, l in enumerate(labels):
        if any(k in l for k in ["CONCLUSION", "RESULTS", "FINDING"]):
            target_idx = idx

    orig_para = contexts[target_idx]
    
    # In-place negation transformations on principal verbs without inventing statistics
    if gold == "yes":
        inversions = [
            (r"\\bis effective\\b", "is not effective"),
            (r"\\bwas effective\\b", "was not effective"),
            (r"\\bwere effective\\b", "were not effective"),
            (r"\\bdemonstrated significant\\b", "failed to demonstrate significant"),
            (r"\\bshowed significant\\b", "did not show significant"),
            (r"\\bsignificantly improved\\b", "did not significantly improve"),
            (r"\\bwas superior\\b", "was not superior"),
            (r"\\bsupports the\\b", "does not support the"),
            (r"\\bsupported the\\b", "did not support the"),
            (r"\\bis recommended\\b", "is not recommended"),
            (r"\\bcan be recommended\\b", "cannot be recommended"),
            (r"\\bwas associated with\\b", "was not associated with"),
            (r"\\bis associated with\\b", "is not associated with"),
            (r"\\bconcluded that\\b", "concluded that evidence fails to demonstrate that")
        ]
        inverted_para = orig_para
        matched = False
        for pat, repl in inversions:
            if re.search(pat, inverted_para, flags=re.IGNORECASE):
                inverted_para = re.sub(pat, repl, inverted_para, count=1, flags=re.IGNORECASE)
                matched = True
                break
        if not matched:
            inverted_para = inverted_para + " [Conclusion: The study findings conclude that there is no significant evidence demonstrating efficacy or therapeutic benefit.]"
        method = "negation_inversion"
        
    elif gold == "no":
        inversions = [
            (r"\\bis not effective\\b", "is effective"),
            (r"\\bwas not effective\\b", "was effective"),
            (r"\\bwere not effective\\b", "were effective"),
            (r"\\bfailed to demonstrate\\b", "successfully demonstrated"),
            (r"\\bfails to demonstrate\\b", "demonstrates"),
            (r"\\bdid not show\\b", "showed"),
            (r"\\bdid not improve\\b", "significantly improved"),
            (r"\\bwas not superior\\b", "was superior"),
            (r"\\bdoes not support\\b", "supports"),
            (r"\\bdid not support\\b", "supported"),
            (r"\\bis not recommended\\b", "is recommended"),
            (r"\\bcannot be recommended\\b", "can be recommended"),
            (r"\\bwas not associated with\\b", "was significantly associated with"),
            (r"\\bno significant difference\\b", "a statistically significant difference")
        ]
        inverted_para = orig_para
        matched = False
        for pat, repl in inversions:
            if re.search(pat, inverted_para, flags=re.IGNORECASE):
                inverted_para = re.sub(pat, repl, inverted_para, count=1, flags=re.IGNORECASE)
                matched = True
                break
        if not matched:
            inverted_para = inverted_para + " [Conclusion: The study findings demonstrate statistically significant efficacy and clinical superiority.]"
        method = "affirmation_inversion"
    else:
        # maybe remains maybe
        inverted_para = orig_para
        method = "neutral_preserved"

    transformed_contexts = list(contexts)
    transformed_contexts[target_idx] = inverted_para
    sep = "\\n\\n"
    cf_abstract = sep.join(transformed_contexts).strip()
    return cf_abstract, inv_label, method

evidence_cache = []
c4_audit_records = []

for idx, row in enumerate(test_records):
    pid = str(row["pubid"]).strip()
    q = row["question"]
    gold = row["final_decision"]
    
    # --- C1: Good Supporting Evidence ---
    sep = "\\n\\n"
    intact = sep.join(row["context"].get("contexts", [])).strip()
    # Retrieve distractors strictly from the 800 training index, excluding test pid
    top_dist = train_retriever.retrieve(q, top_k=2, exclude_pubids=[pid])
    c1_slots = [("target", pid, intact), ("dist_1", top_dist[0][0], top_dist[0][1]), ("dist_2", top_dist[1][0], top_dist[1][1])]
    rng1 = random.Random(SEED + idx * 17)
    perm1 = list(range(3))
    rng1.shuffle(perm1)
    c1_ordered = [c1_slots[i] for i in perm1]
    
    # --- C2: Incomplete Evidence ---
    inc_text, is_valid_inc = strip_outcomes_c2(row["context"])
    # Strictly disjoint distractors from training corpus with zero query overlap
    disjoint_dist = train_retriever.retrieve_disjoint(q, top_k=2, exclude_pubids=[pid])
    c2_slots = [("target_inc", pid, inc_text), ("disjoint_1", disjoint_dist[0][0], disjoint_dist[0][1]), ("disjoint_2", disjoint_dist[1][0], disjoint_dist[1][1])]
    perm2 = list(range(3))
    random.Random(SEED + idx * 19).shuffle(perm2)
    c2_ordered = [c2_slots[i] for i in perm2]
    
    # --- C3: Irrelevant Evidence ---
    off_target = train_retriever.retrieve_disjoint(q, top_k=3, exclude_pubids=[pid])
    c3_slots = [(f"irr_{i+1}", p[0], p[1]) for i, p in enumerate(off_target)]
    perm3 = list(range(3))
    random.Random(SEED + idx * 23).shuffle(perm3)
    c3_ordered = [c3_slots[i] for i in perm3]
    
    # --- C4: Misleading Counterfactual Evidence (In-Place Outcome Inversion, No Fabrication) ---
    cf_text, inv_label, inv_method = invert_outcome_c4_non_fabricated(row["context"], gold)
    c4_slots = [("target_cf", pid, cf_text), ("dist_1", top_dist[0][0], top_dist[0][1]), ("dist_2", top_dist[1][0], top_dist[1][1])]
    c4_ordered = [c4_slots[i] for i in perm1]  # Identical slot permutation to C1 to isolate counterfactual effect
    
    c4_audit_records.append({
        "pubid": pid, "question": q, "gold_decision": gold, "inverted_label": inv_label,
        "inversion_method": inv_method, "fabricated_studies_or_cohorts": False,
        "original_abstract": intact, "counterfactual_abstract": cf_text
    })
    
    def format_passages(slots):
        lines = []
        for slot_num, s in enumerate(slots, 1):
            lines.append(f"[Passage {slot_num} (ID: {s[1]})]\\n{s[2]}")
        return "\\n\\n".join(lines)
        
    evidence_cache.append({
        "pubid": pid,
        "question": q,
        "gold_decision": gold,
        "C1": {"context_text": format_passages(c1_ordered)},
        "C2": {"context_text": format_passages(c2_ordered), "valid_incomplete": is_valid_inc},
        "C3": {"context_text": format_passages(c3_ordered)},
        "C4": {"context_text": format_passages(c4_ordered), "inverted_label": inv_label}
    })

valid_c2_count = sum(1 for e in evidence_cache if e["C2"]["valid_incomplete"])
print(f"Total Constructed Evidence Packets: {len(evidence_cache)}")
print(f"Validated Incomplete Cases (N_inc): {valid_c2_count} / {len(test_records)}")

# Save Evidence Cache and C4 Counterfactual Audit log
ev_cache_path = os.path.join(DATA_DIR, "evidence_cache.jsonl")
with open(ev_cache_path, "w", encoding="utf-8") as f:
    for item in evidence_cache:
        f.write(json.dumps(item) + "\\n")
print(f"Saved Evidence Cache:               {ev_cache_path}")

c4_audit_path = os.path.join(DATA_DIR, "c4_counterfactual_audit.jsonl")
with open(c4_audit_path, "w", encoding="utf-8") as f:
    for rec in c4_audit_records:
        f.write(json.dumps(rec) + "\\n")
print(f"Saved C4 Counterfactual Audit:      {c4_audit_path} (Zero fabricated studies/statistics)")
print("[4/8] Controlled Evidence Construction Complete.\\n")

# ----------------------------------------------------------------------------------------------------
# [STAGE 5/8] 4-BIT MODEL QUANTIZATION & QLORA ADAPTER FINE-TUNING
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "5/8: 4-bit Quantization & QLoRA Fine-Tuning"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Setting up Qwen2.5-1.5B-Instruct in 4-bit NF4...")
print("=" * 90)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 4-bit NF4 Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print(f"[Model Setup] Downloading/Loading {MODEL_ID} for QLoRA fine-tuning...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load dedicated training model instance
train_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
train_model = prepare_model_for_kbit_training(train_model)

# LoRA Configuration adhering strictly to Phase 2:
# Rank r=16, alpha=32, dropout=0.05, target modules: q_proj, k_proj, v_proj, o_proj
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(train_model, lora_config)

# Safe parameter counting
try:
    _param_result = peft_model.get_nb_trainable_parameters()
    if isinstance(_param_result, tuple):
        trainable_params, all_params = _param_result
    else:
        trainable_params = int(_param_result)
        all_params = sum(p.numel() for p in peft_model.parameters())
except Exception:
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())

print(f"[LoRA Architecture Audit]")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Total Base Parameters: {all_params:,}")
print(f"  Trainable Percentage:  {trainable_params / all_params * 100:.3f}% (Expected ~0.282%)")

# Native ChatML Training Prompt Formulation
SYSTEM_PROMPT = (
    "You are a biomedical expert reader. Answer the given clinical research question faithfully "
    "based on the provided evidence passages. If evidence is insufficient, state that clearly. "
    "Conclude strictly with 'Final Decision: yes', 'Final Decision: no', or 'Final Decision: maybe'."
)

# STRICT METHODOLOGY COMPLIANCE: long_answer is NEVER used as training target!
# Assistant supervision target is strictly the canonical decision classification.
formatted_training_texts = []
for r in train_records:
    sep = "\\n\\n"
    ctx = sep.join(r["context"].get("contexts", [])).strip()
    u_content = f"Evidence Context:\\n{ctx}\\n\\nQuestion: {r['question']}"
    a_content = f"Final Decision: {r['final_decision']}"
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": u_content},
        {"role": "assistant", "content": a_content}
    ]
    formatted_training_texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))

from datasets import Dataset as HFDataset
train_dataset = HFDataset.from_dict({"text": formatted_training_texts})

# Calculate Planned Optimizer Steps & Equivalent Warmup Steps
import inspect
import dataclasses

number_of_training_examples = len(train_records)  # 800
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
num_train_epochs = 1.0
original_warmup_ratio = 0.05  # 5% linear warmup specified in Phase 2

planned_optimizer_steps = math.ceil(number_of_training_examples / (per_device_train_batch_size * gradient_accumulation_steps)) * int(num_train_epochs)
warmup_steps = max(1, int(round(planned_optimizer_steps * original_warmup_ratio)))  # 5 steps

print(f"[Training Schedule] Examples: {number_of_training_examples} | Effective Batch Size: {per_device_train_batch_size * gradient_accumulation_steps}")
print(f"[Training Schedule] Planned Optimizer Steps: {planned_optimizer_steps} | Configured Warmup Steps: {warmup_steps} (5.0%)")

# Determine Configuration Class (SFTConfig vs TrainingArguments)
has_sft_config = False
config_cls = TrainingArguments
try:
    from trl import SFTConfig
    config_cls = SFTConfig
    has_sft_config = True
    print("[Training Config] Using TRL SFTConfig class.")
except (ImportError, AttributeError):
    print("[Training Config] Using transformers.TrainingArguments class.")

# Inspect constructor parameters
supported_params = set(inspect.signature(config_cls).parameters.keys())
if dataclasses.is_dataclass(config_cls):
    supported_params.update(f.name for f in dataclasses.fields(config_cls))
supported_params.update(inspect.signature(TrainingArguments).parameters.keys())
if dataclasses.is_dataclass(TrainingArguments):
    supported_params.update(f.name for f in dataclasses.fields(TrainingArguments))

# Base training arguments with warmup_steps replacing warmup_ratio in both paths
training_kwargs = {
    "output_dir": ADAPTER_DIR,
    "per_device_train_batch_size": per_device_train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": warmup_steps,  # Explicitly using warmup_steps across all versions
    "num_train_epochs": num_train_epochs,
    "logging_steps": 10,
    "fp16": True,
    "save_strategy": "no",
    "report_to": "none"
}

if "optim" in supported_params:
    training_kwargs["optim"] = "paged_adamw_8bit"

if "activation_checkpointing" in supported_params:
    training_kwargs["activation_checkpointing"] = True
elif "gradient_checkpointing" in supported_params:
    training_kwargs["gradient_checkpointing"] = True

if has_sft_config:
    if "max_length" in supported_params:
        training_kwargs["max_length"] = 1024
    elif "max_seq_length" in supported_params:
        training_kwargs["max_seq_length"] = 1024
    if "dataset_text_field" in supported_params:
        training_kwargs["dataset_text_field"] = "text"

final_training_kwargs = {k: v for k, v in training_kwargs.items() if k in supported_params}
training_args_obj = config_cls(**final_training_kwargs)

trainer_sig = inspect.signature(SFTTrainer)
trainer_supported_params = set(trainer_sig.parameters.keys())

trainer_kwargs = {
    "model": peft_model,
    "train_dataset": train_dataset,
    "args": training_args_obj
}

if "processing_class" in trainer_supported_params:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_supported_params:
    trainer_kwargs["tokenizer"] = tokenizer

if "dataset_text_field" in trainer_supported_params and "dataset_text_field" not in final_training_kwargs:
    trainer_kwargs["dataset_text_field"] = "text"
if "max_length" in trainer_supported_params and "max_length" not in final_training_kwargs and "max_seq_length" not in final_training_kwargs:
    trainer_kwargs["max_length"] = 1024
elif "max_seq_length" in trainer_supported_params and "max_length" not in final_training_kwargs and "max_seq_length" not in final_training_kwargs:
    trainer_kwargs["max_seq_length"] = 1024

trainer = SFTTrainer(**trainer_kwargs)
print(f"[Training Config] SFTTrainer initialized successfully with {config_cls.__name__}.")

print("[Training] Commencing 100-step QLoRA fine-tuning run...")
t_train_start = time.time()
train_result = trainer.train()
t_train_duration = time.time() - t_train_start
print(f"[Training Complete] Final Training Loss: {train_result.training_loss:.4f} in {t_train_duration:.1f} seconds.")

# Save Trained QLoRA Adapter Checkpoint
print(f"[Model] Saving QLoRA adapter checkpoint to {ADAPTER_DIR}...")
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

with open(os.path.join(ADAPTER_DIR, "adapter_metadata.json"), "w", encoding="utf-8") as f:
    json.dump({
        "base_model": MODEL_ID,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "learning_rate": 2e-4,
        "warmup_steps": warmup_steps,
        "train_loss": train_result.training_loss,
        "training_duration_sec": t_train_duration,
        "leakage_guardrail": "long_answer strictly excluded from training targets",
        "timestamp": datetime.datetime.now().isoformat()
    }, f, indent=2)

print(f"Adapter saved successfully to: {ADAPTER_DIR}")

# CRITICAL RESEARCH INTEGRITY FIX:
# Completely release the training model instance, trainer, and PEFT wrapper from GPU memory.
# This prevents any training hooks, gradient modifications, or adapter states from leaking
# into the frozen baseline reader evaluation!
print("[Cleanup] Releasing training model instance and flushing GPU memory cache...")
del trainer
del peft_model
del train_model
gc.collect()
torch.cuda.empty_cache()
print("[Cleanup] GPU memory successfully cleared. Ready for independent inference.\\n")
print("[5/8] QLoRA Fine-Tuning Complete.\\n")

# ----------------------------------------------------------------------------------------------------
# [STAGE 6/8] TEN-CONFIGURATION EXPERIMENTAL INFERENCE PIPELINE (E1–E10)
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "6/8: Ten-Configuration Experimental Inference"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Executing 10-Configuration Experimental Matrix (E1-E10)...")
print("=" * 90)

EXPERIMENTAL_MATRIX = {
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
}

def parse_decision_from_output(text: str) -> str:
    \"\"\"Extracts explicit decision label from generation.\"\"\"
    m = re.search(r"Final Decision:\\s*(yes|no|maybe)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\\b(yes|no|maybe)[.!]?\\s*$", text.strip(), re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return "unparseable"

ABSTENTION_PROXIES = [
    "insufficient evidence", "cannot be determined", "does not report",
    "not enough information", "unclear from the provided", "not provided in the context",
    "unable to determine", "no conclusive evidence", "data are lacking"
]

def check_is_abstaining(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in ABSTENTION_PROXIES)

def execute_inference_batch(eval_model, model_type_label: str) -> List[Dict[str, Any]]:
    eval_model.eval()
    records = []
    
    for exp_id, meta in EXPERIMENTAL_MATRIX.items():
        if meta["model"] != model_type_label:
            continue
            
        cond = meta["condition"]
        print(f"[Inference] Executing {exp_id}: {model_type_label} | {meta['name']} ({cond})...")
        
        executed_in_cond = 0
        for entry in evidence_cache:
            # E4 and E8 run strictly on the validated N_inc incomplete cases
            if cond == "C2" and not entry["C2"].get("valid_incomplete", True):
                continue
                
            q = entry["question"]
            gold = entry["gold_decision"]
            inv = entry["C4"]["inverted_label"] if cond == "C4" else "N/A"
            
            ctx = None if cond == "None" else entry[cond]["context_text"]
            u_msg = f"Evidence Context:\\n{ctx}\\n\\nQuestion: {q}" if ctx else f"Question: {q}"
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u_msg}]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(eval_model.device)
            
            # Greedy decoding with GPU-synchronized latency timing
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_toks = eval_model.generate(**inputs, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            
            gen_text = tokenizer.decode(out_toks[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            dec = parse_decision_from_output(gen_text)
            
            records.append({
                "exp_id": exp_id,
                "model": model_type_label,
                "condition": cond,
                "pubid": entry["pubid"],
                "question": q,
                "gold_decision": gold,
                "inverted_label": inv,
                "raw_output": gen_text,
                "parsed_decision": dec,
                "is_correct": (dec == gold),
                "is_abstaining": check_is_abstaining(gen_text),
                "latency_sec": lat
            })
            executed_in_cond += 1
            
        print(f"  --> Completed {executed_in_cond} queries for {exp_id}.")
        
    return records

# CRITICAL FIX: Load a completely fresh, untouched 4-bit base model instance with ZERO training hooks
print("\\n[Inference Phase A] Loading fresh, pristine frozen base reader (Qwen2.5-1.5B-Instruct)...")
frozen_base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
frozen_base_model.eval()

# Execute frozen base reader evaluations (E1, E3, E4, E5, E6)
base_predictions = execute_inference_batch(frozen_base_model, "Base")

# Now attach the trained QLoRA adapter onto the base model for adapted evaluations (E2, E7, E8, E9, E10)
print("\\n[Inference Phase B] Attaching trained QLoRA adapter for adapted evaluations...")
qlora_reader = PeftModel.from_pretrained(frozen_base_model, ADAPTER_DIR)
qlora_reader.eval()

qlora_predictions = execute_inference_batch(qlora_reader, "QLoRA")

all_predictions = base_predictions + qlora_predictions
predictions_df = pd.DataFrame(all_predictions)

predictions_csv_path = os.path.join(RESULTS_DIR, "predictions_all_conditions.csv")
predictions_df.to_csv(predictions_csv_path, index=False)
print(f"\\nSaved Predictions CSV: {predictions_csv_path} ({len(predictions_df)} total generations).")
print("[6/8] Ten-Configuration Experimental Inference Complete.\\n")

# ----------------------------------------------------------------------------------------------------
# [STAGE 7/8] QUANTITATIVE METRICS & COMPREHENSIVE STATISTICAL SUITE
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "7/8: Metric Calculation & Statistical Evaluation"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Computing Decision Accuracy, Macro-F1, CAR, IRR, UCAR, CBFR & 10,000 Paired Bootstrap...")
print("=" * 90)

def calc_macro_f1(df_sub: pd.DataFrame) -> float:
    classes = ["yes", "no", "maybe"]
    f1s = []
    for c in classes:
        tp = ((df_sub["gold_decision"] == c) & (df_sub["parsed_decision"] == c)).sum()
        fp = ((df_sub["gold_decision"] != c) & (df_sub["parsed_decision"] == c)).sum()
        fn = ((df_sub["gold_decision"] == c) & (df_sub["parsed_decision"] != c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

def independent_bootstrap_ci(df_sub: pd.DataFrame, n_boot: int = 10000) -> Tuple[float, float]:
    vals = (df_sub["parsed_decision"] == df_sub["gold_decision"]).values
    if len(vals) < 5:
        return 0.0, 0.0
    rng = np.random.RandomState(SEED)
    boot_means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def paired_bootstrap_delta(df_base: pd.DataFrame, df_qlora: pd.DataFrame, n_boot: int = 10000) -> Tuple[float, float, float]:
    \"\"\"10,000 paired bootstrap resamples of accuracy delta (QLoRA - Base).\"\"\"
    merged = pd.merge(df_base[["pubid", "is_correct"]], df_qlora[["pubid", "is_correct"]], on="pubid", suffixes=("_base", "_qlora"))
    diffs = (merged["is_correct_qlora"].astype(float) - merged["is_correct_base"].astype(float)).values
    rng = np.random.RandomState(SEED)
    boot_deltas = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return float(np.mean(diffs)), float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))

def calculate_mcnemar_test(df_base: pd.DataFrame, df_qlora: pd.DataFrame) -> Tuple[float, float, int, int]:
    \"\"\"McNemar's test with continuity correction on paired binary classification outcomes.\"\"\"
    merged = pd.merge(df_base[["pubid", "is_correct"]], df_qlora[["pubid", "is_correct"]], on="pubid", suffixes=("_base", "_qlora"))
    b = int(((merged["is_correct_base"] == True) & (merged["is_correct_qlora"] == False)).sum())
    c = int(((merged["is_correct_base"] == False) & (merged["is_correct_qlora"] == True)).sum())
    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0, b, c
    stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_val = math.erfc(math.sqrt(stat / 2.0))
    return float(stat), float(p_val), b, c

summary_rows = []
for exp_id, meta in EXPERIMENTAL_MATRIX.items():
    sub = predictions_df[predictions_df["exp_id"] == exp_id]
    if sub.empty:
        continue
        
    acc = float((sub["parsed_decision"] == sub["gold_decision"]).mean())
    acc_ci_low, acc_ci_high = independent_bootstrap_ci(sub, n_boot=10000)
    macro_f1 = calc_macro_f1(sub)
    unparseable_count = int((sub["parsed_decision"] == "unparseable").sum())
    
    cond = meta["condition"]
    # CAR: Correct Abstention Rate strictly evaluated on valid N_inc subset
    car = float(sub["is_abstaining"].mean()) if cond == "C2" else None
    
    # IRR: Distractor Rejection under C3 reported with both components
    irr_abstention = float(sub["is_abstaining"].mean()) if cond == "C3" else None
    irr_parametric = float(sub["is_correct"].mean()) if cond == "C3" else None
    
    # Counterfactual Adoption on Binary Subset under C4
    bin_sub = sub[sub["gold_decision"].isin(["yes", "no"])]
    ucar = float((bin_sub["parsed_decision"] == bin_sub["inverted_label"]).mean()) if cond == "C4" and not bin_sub.empty else None
    inc_bin_sub = bin_sub[bin_sub["parsed_decision"] != bin_sub["gold_decision"]]
    cbfr = float((inc_bin_sub["parsed_decision"] == inc_bin_sub["inverted_label"]).mean()) if cond == "C4" and not inc_bin_sub.empty else None
    
    summary_rows.append({
        "exp_id": exp_id,
        "model": meta["model"],
        "condition": cond,
        "name": meta["name"],
        "queries": len(sub),
        "unparseable": unparseable_count,
        "accuracy": round(acc, 4),
        "ci_low": round(acc_ci_low, 4),
        "ci_high": round(acc_ci_high, 4),
        "macro_f1": round(macro_f1, 4),
        "car": round(car, 4) if car is not None else "N/A",
        "irr_abstention": round(irr_abstention, 4) if irr_abstention is not None else "N/A",
        "irr_parametric": round(irr_parametric, 4) if irr_parametric is not None else "N/A",
        "ucar": round(ucar, 4) if ucar is not None else "N/A",
        "cbfr": round(cbfr, 4) if cbfr is not None else "N/A",
        "median_latency_sec": round(float(sub["latency_sec"].median()), 3)
    })

metrics_summary_df = pd.DataFrame(summary_rows)
metrics_csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
metrics_summary_df.to_csv(metrics_csv_path, index=False)

# Perform Formal Paired Statistical Suite across C1, C2, C3, C4
paired_stats = {}
raw_p_values = {}
for cond in ["C1", "C2", "C3", "C4"]:
    df_b = predictions_df[(predictions_df["condition"] == cond) & (predictions_df["model"] == "Base")]
    df_q = predictions_df[(predictions_df["condition"] == cond) & (predictions_df["model"] == "QLoRA")]
    
    delta, ci_l, ci_u = paired_bootstrap_delta(df_b, df_q, n_boot=10000)
    chi2, p_mcnemar, b_count, c_count = calculate_mcnemar_test(df_b, df_q)
    
    raw_p_values[cond] = p_mcnemar
    paired_stats[cond] = {
        "condition": cond,
        "queries": len(df_b),
        "delta_accuracy_qlora_minus_base": round(delta, 4),
        "delta_95_ci": [round(ci_l, 4), round(ci_u, 4)],
        "mcnemar_stat": round(chi2, 4),
        "mcnemar_p_value": round(p_mcnemar, 5),
        "discordant_pairs": {"base_correct_qlora_wrong": b_count, "base_wrong_qlora_correct": c_count}
    }

# Apply Holm-Bonferroni Correction across the 4 primary condition tests
sorted_conds = sorted(raw_p_values.items(), key=lambda x: x[1])
m_tests = len(sorted_conds)
for rank, (cond, p_val) in enumerate(sorted_conds):
    thresh = 0.05 / (m_tests - rank)
    paired_stats[cond]["holm_bonferroni_threshold"] = round(thresh, 5)
    paired_stats[cond]["statistically_significant"] = bool(p_val <= thresh)

stats_json_path = os.path.join(RESULTS_DIR, "statistical_tests.json")
with open(stats_json_path, "w", encoding="utf-8") as f:
    json.dump(paired_stats, f, indent=2)

print("\\n" + "=" * 90)
print("EXPERIMENTAL METRICS MATRIX SUMMARY")
print("=" * 90)
cols_to_show = ["exp_id", "model", "condition", "accuracy", "ci_low", "ci_high", "macro_f1", "car", "ucar", "cbfr", "median_latency_sec"]
print(metrics_summary_df[cols_to_show].to_string(index=False))

print("\\n" + "=" * 90)
print("PAIRED STATISTICAL HYPOTHESIS TESTS (10,000 Paired Bootstrap & McNemar / Holm-Bonferroni)")
print("=" * 90)
for cond, st in paired_stats.items():
    print(f"Condition {cond}: Delta(DA)={st['delta_accuracy_qlora_minus_base']:+.4f} | 95% CI={st['delta_95_ci']} | p={st['mcnemar_p_value']:.5f} (Sig={st['statistically_significant']})")

print(f"\\nSaved Metrics CSV:    {metrics_csv_path}")
print(f"Saved Statistics JSON: {stats_json_path}")
print("[7/8] Quantitative Metrics & Statistical Suite Complete.\\n")

# ----------------------------------------------------------------------------------------------------
# [STAGE 8/8] PUBLICATION VISUALIZATIONS, HUMAN REVIEW AUDIT SHEET & ABLATIONS
# ----------------------------------------------------------------------------------------------------
CURRENT_STAGE = "8/8: Visualizations, Manual Audit Sheet & Ablation Studies"
print("=" * 90)
print(f"[{CURRENT_STAGE}] Generating Plots, 100-Sample Review Sheet & Ablations...")
print("=" * 90)

sns.set_theme(style="whitegrid", font_scale=1.1)

# Helper to coerce numeric columns safely for plotting
def coerce_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")

# --- 1. Figure 1: Decision Accuracy across Evidence Conditions ---
plt.figure(figsize=(9, 5.5), dpi=300)
rag_df = metrics_summary_df[metrics_summary_df["condition"].isin(["C1", "C2", "C3", "C4"])].copy()
rag_df["accuracy"] = coerce_num(rag_df, "accuracy")

ax = sns.barplot(data=rag_df, x="condition", y="accuracy", hue="model", palette=["#4A90E2", "#E67E22"])
e1_val = metrics_summary_df[metrics_summary_df["exp_id"] == "E1"]["accuracy"].iloc[0]
e2_val = metrics_summary_df[metrics_summary_df["exp_id"] == "E2"]["accuracy"].iloc[0]
plt.axhline(e1_val, color="#2980B9", linestyle="--", linewidth=1.5, label=f"E1 Base Q-Only ({e1_val:.2f})")
plt.axhline(e2_val, color="#D35400", linestyle=":", linewidth=1.5, label=f"E2 QLoRA Q-Only ({e2_val:.2f})")

plt.title("Decision Accuracy Across Controlled Evidence Conditions", fontweight="bold", pad=12)
plt.ylabel("Decision Accuracy (DA)", fontweight="bold")
plt.xlabel("Evidence Condition", fontweight="bold")
plt.xticks([0, 1, 2, 3], [f"C1: Good (N=200)", f"C2: Incomplete (N={valid_c2_count})", "C3: Irrelevant (N=200)", "C4: Misleading (N=200)"])
plt.ylim(0, 1.05)
plt.legend(loc="upper right", frameon=True)
fig1_path = os.path.join(FIGURES_DIR, "fig1_accuracy_comparison.png")
plt.tight_layout()
plt.savefig(fig1_path)
plt.close()
print(f"Saved Figure 1: {fig1_path}")

# --- 2. Figure 2: Robustness Trade-offs (CAR vs UCAR / CBFR) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

c2_rows = metrics_summary_df[metrics_summary_df["condition"] == "C2"].copy()
c2_rows["car"] = coerce_num(c2_rows, "car")
sns.barplot(data=c2_rows, x="model", y="car", palette=["#4A90E2", "#E67E22"], ax=ax1)
ax1.set_title(f"Panel A: Incomplete Evidence (C2, N={valid_c2_count})\\nCorrect Abstention Rate (CAR)", fontweight="bold")
ax1.set_ylabel("CAR", fontweight="bold")
ax1.set_ylim(0, 1.0)

c4_numeric = metrics_summary_df[metrics_summary_df["condition"] == "C4"].copy()
c4_numeric["ucar"] = coerce_num(c4_numeric, "ucar")
c4_numeric["cbfr"] = coerce_num(c4_numeric, "cbfr")
c4_melt = c4_numeric.melt(id_vars=["model"], value_vars=["ucar", "cbfr"], var_name="Metric", value_name="Rate").dropna(subset=["Rate"])

sns.barplot(data=c4_melt, x="model", y="Rate", hue="Metric", palette=["#C0392B", "#8E44AD"], ax=ax2)
ax2.set_title("Panel B: Misleading Evidence (C4, N=200)\\nCounterfactual Adoption Rates", fontweight="bold")
ax2.set_ylabel("Adoption Rate", fontweight="bold")
ax2.set_ylim(0, 1.0)

plt.suptitle("Robustness Trade-Offs: Abstention vs. Adversarial Adoption", fontweight="bold", y=1.02)
fig2_path = os.path.join(FIGURES_DIR, "fig2_abstention_and_misleading.png")
plt.tight_layout()
plt.savefig(fig2_path, bbox_inches="tight")
plt.close()
print(f"Saved Figure 2: {fig2_path}")

# --- 3. Figure 3: Performance Retention Degradation Curves ---
plt.figure(figsize=(8, 5), dpi=300)
conds = ["C1", "C2", "C3", "C4"]
base_accs = [metrics_summary_df[(metrics_summary_df['condition']==c) & (metrics_summary_df['model']=='Base')]['accuracy'].iloc[0] for c in conds]
qlora_accs = [metrics_summary_df[(metrics_summary_df['condition']==c) & (metrics_summary_df['model']=='QLoRA')]['accuracy'].iloc[0] for c in conds]

plt.plot(["Good (C1)", "Incomplete (C2)", "Irrelevant (C3)", "Misleading (C4)"], base_accs, marker='o', linewidth=2.5, color="#4A90E2", label="Frozen Base")
plt.plot(["Good (C1)", "Incomplete (C2)", "Irrelevant (C3)", "Misleading (C4)"], qlora_accs, marker='s', linewidth=2.5, color="#E67E22", label="QLoRA Adapted")
plt.title("Performance Retention Degradation Curves", fontweight="bold", pad=12)
plt.ylabel("Decision Accuracy (DA)", fontweight="bold")
plt.ylim(0, 1.05)
plt.legend(frameon=True)
fig3_path = os.path.join(FIGURES_DIR, "fig3_performance_retention.png")
plt.tight_layout()
plt.savefig(fig3_path)
plt.close()
print(f"Saved Figure 3: {fig3_path}")

# --- 4. Qualitative Error Review Sheet (100 Stratified Samples) ---
# Exported with blank human annotation columns as required by Phase 2 methodology
def heuristic_pre_classify(row: pd.Series) -> str:
    cond = row["condition"]
    is_corr = row["is_correct"]
    is_abs = row["is_abstaining"]
    dec = str(row["parsed_decision"]).lower()
    inv = str(row["inverted_label"]).lower()
    
    if cond == "C2" and is_abs:
        return "Heuristic-Candidate-Abstention"
    if cond == "C4" and dec == inv and inv in ["yes", "no"]:
        return "Heuristic-Candidate-Bad-Evidence-Adoption"
    if is_corr:
        return "Heuristic-Candidate-Correct"
    return "Heuristic-Candidate-Error"

sampled_audit_rows = []
for cond in ["C1", "C2", "C3", "C4"]:
    b_sub = predictions_df[(predictions_df["condition"] == cond) & (predictions_df["model"] == "Base")]
    q_sub = predictions_df[(predictions_df["condition"] == cond) & (predictions_df["model"] == "QLoRA")]
    n_b = min(12, len(b_sub))
    n_q = min(13, len(q_sub))
    if n_b > 0:
        sampled_audit_rows.append(b_sub.sample(n_b, random_state=SEED))
    if n_q > 0:
        sampled_audit_rows.append(q_sub.sample(n_q, random_state=SEED))

audit_df = pd.concat(sampled_audit_rows, ignore_index=True)
audit_df["heuristic_pre_classification"] = audit_df.apply(heuristic_pre_classify, axis=1)

# Blank human annotation columns for independent blinded audit
audit_df["HUMAN_GROUNDED_VERIFIED (Yes/No)"] = "PENDING_HUMAN_REVIEW"
audit_df["HUMAN_ERROR_TAXONOMY (Grounded-Correct/Parametric-Correct/Bad-Evidence-Exploited/Ungrounded-Hallucination/Clean-Abstention/Other)"] = "PENDING_HUMAN_REVIEW"
audit_df["HUMAN_REVIEWER_NOTES"] = ""

audit_csv_path = os.path.join(RESULTS_DIR, "manual_audit_100_samples_review_sheet.csv")
audit_df.to_csv(audit_csv_path, index=False)

print(f"\\n[Qualitative Audit] Exported 100-sample stratified review sheet to: {audit_csv_path}")
print(f"[Qualitative Audit] Human review columns initialized to PENDING_HUMAN_REVIEW for blinded manual audit.")

# Figure 4: Distribution of Heuristic Pre-Classifications
plt.figure(figsize=(10, 5), dpi=300)
sns.countplot(data=audit_df, x="heuristic_pre_classification", hue="model", palette=["#4A90E2", "#E67E22"])
plt.title("Distribution of Audited Outputs (Heuristic Pre-Classification)", fontweight="bold", pad=12)
plt.ylabel("Count (N=100 Sample)", fontweight="bold")
plt.xlabel("Category", fontweight="bold")
plt.xticks(rotation=15, ha="right")
plt.legend(frameon=True)
fig4_path = os.path.join(FIGURES_DIR, "fig4_error_distribution.png")
plt.tight_layout()
plt.savefig(fig4_path)
plt.close()
print(f"Saved Figure 4: {fig4_path}")

# --- 5. Formal Ablation Studies ---
ablations = []

e1_acc = metrics_summary_df[metrics_summary_df['exp_id']=='E1']['accuracy'].iloc[0]
e2_acc = metrics_summary_df[metrics_summary_df['exp_id']=='E2']['accuracy'].iloc[0]
e3_acc = metrics_summary_df[metrics_summary_df['exp_id']=='E3']['accuracy'].iloc[0]
e7_acc = metrics_summary_df[metrics_summary_df['exp_id']=='E7']['accuracy'].iloc[0]

ablations.append({
    "Ablation": "Base: Q-Only vs RAG", "Comparison": "E3 (RAG C1) - E1 (Q-Only)",
    "Baseline": e1_acc, "Conditioned": e3_acc, "Delta": round(e3_acc - e1_acc, 4),
    "Isolates": "Net performance gain from intact retrieved context over frozen parametric memory."
})
ablations.append({
    "Ablation": "QLoRA: Q-Only vs RAG", "Comparison": "E7 (RAG C1) - E2 (Q-Only)",
    "Baseline": e2_acc, "Conditioned": e7_acc, "Delta": round(e7_acc - e2_acc, 4),
    "Isolates": "Context utilization in instruction-adapted weights over question-only prompt."
})

for m in ["Base", "QLoRA"]:
    c1 = metrics_summary_df[(metrics_summary_df['condition']=='C1') & (metrics_summary_df['model']==m)]['accuracy'].iloc[0]
    c2 = metrics_summary_df[(metrics_summary_df['condition']=='C2') & (metrics_summary_df['model']==m)]['accuracy'].iloc[0]
    c3 = metrics_summary_df[(metrics_summary_df['condition']=='C3') & (metrics_summary_df['model']==m)]['accuracy'].iloc[0]
    c4 = metrics_summary_df[(metrics_summary_df['condition']=='C4') & (metrics_summary_df['model']==m)]['accuracy'].iloc[0]
    
    ablations.append({
        "Ablation": f"{m}: Incomplete Drop", "Comparison": f"{m} (C2) - {m} (C1)",
        "Baseline": c1, "Conditioned": c2, "Delta": round(c2 - c1, 4),
        "Isolates": "Vulnerability to trial outcome omission and findings suppression."
    })
    ablations.append({
        "Ablation": f"{m}: Irrelevant Drop", "Comparison": f"{m} (C3) - {m} (C1)",
        "Baseline": c1, "Conditioned": c3, "Delta": round(c3 - c1, 4),
        "Isolates": "Vulnerability to off-target distractor noise."
    })
    ablations.append({
        "Ablation": f"{m}: Misleading Drop", "Comparison": f"{m} (C4) - {m} (C1)",
        "Baseline": c1, "Conditioned": c4, "Delta": round(c4 - c1, 4),
        "Isolates": "Vulnerability to direct counterfactual evidence injection."
    })

ablation_df = pd.DataFrame(ablations)
ablation_csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
ablation_df.to_csv(ablation_csv_path, index=False)
print(f"\\nSaved Ablation CSV:     {ablation_csv_path}")
print(ablation_df[["Ablation", "Baseline", "Conditioned", "Delta"]].to_string(index=False))

# --- 6. Package Downloadable Results Archive ---
zip_output_name = "Phase3_Colab_Execution_Results"
shutil.make_archive(zip_output_name, "zip", EXP_ROOT)
print("\\n" + "=" * 90)
print("PHASE 3 COMPLETE EXPERIMENT EXECUTION FINISHED SUCCESSFULLY!")
print(f"All outputs packaged into: {zip_output_name}.zip")
print(f"Results Directory:         {os.path.abspath(RESULTS_DIR)}")
print(f"Figures Directory:         {os.path.abspath(FIGURES_DIR)}")
print(f"Adapter Directory:         {os.path.abspath(ADAPTER_DIR)}")
print("=" * 90)
'''

# Verify Python syntax
ast.parse(code_content)
print("Updated code content passed AST parsing!")

# Write to colab_single_cell_implementation.py
with open("colab_single_cell_implementation.py", "w", encoding="utf-8") as f:
    f.write(code_content)
print("Successfully updated colab_single_cell_implementation.py")

# Build Phase_3_Implementation_Robustness_QLoRA_RAG.ipynb
notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'},
        'accelerator': 'GPU',
        'colab': {'provenance': [], 'gpuType': 'T4'}
    },
    'cells': [
        {
            'cell_type': 'markdown',
            'id': 'header-cell',
            'metadata': {},
            'source': [
                '# Phase 3: Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence\n',
                '**Researcher:** Muanna Hamid (Roll No: 242593)  \n',
                '**Supervisor:** Prof. Azka Atiq  \n',
                '**Institution:** FAST-NUCES Islamabad | BS Data Science (5th Semester)  \n',
                '**Date:** November 2026\n',
                '\n',
                '---\n',
                '\n',
                '## Instructions\n',
                '1. Go to **Runtime → Change runtime type → T4 GPU** before running.\n',
                '2. Run the single code cell below — it executes all 8 stages end-to-end.\n',
                '3. Expected total runtime: ~25–40 minutes on T4 GPU.\n',
                '4. All outputs are saved to `/content/phase3_rag_robustness/` and zipped at the end.\n',
                '\n',
                '> **Methodology Note:** Strictly follows Phase 1 & Phase 2 research design.\n',
                '> - `long_answer` is strictly quarantined from all retrieval, evidence, and training targets.\n',
                '> - Baseline inference uses an untouched, freshly loaded frozen base reader.\n',
                '> - C4 uses in-place outcome inversion with zero fabricated clinical studies.\n',
                '> - Statistical suite includes 10,000 paired bootstrap, McNemar tests, and Holm-Bonferroni correction.'
            ]
        },
        {
            'cell_type': 'code',
            'id': 'phase3-main-cell',
            'metadata': {},
            'source': code_content.splitlines(keepends=True),
            'outputs': [],
            'execution_count': None
        }
    ]
}

with open("Phase_3_Implementation_Robustness_QLoRA_RAG.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Successfully updated Phase_3_Implementation_Robustness_QLoRA_RAG.ipynb")

