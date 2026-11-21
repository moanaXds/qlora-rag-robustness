"""
Inference Pipeline and Evaluation Engine
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Adheres strictly to Phase 2 Methodology:
- Evaluates the 10 Experimental Matrix configurations (E1-E10).
- Identical system instructions, questions, evidence passages, and passage orders.
- Greedy decoding: do_sample=False, max_new_tokens=128, context budget=1024 tokens.
- Robust parsing of explicit 'Final Decision:' field.
- Resumable CSV/JSONL checkpointing to prevent duplicate inference.
- GPU-synchronized latency tracking.
"""

import os
import re
import csv
import json
import time
from typing import List, Dict, Any, Optional
from src.config import ExperimentConfig

try:
    import pandas as pd
except ImportError:
    pd = None


class InferenceEngine:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.tokenizer = None
        self.model = None
        self.current_model_type = None  # "Base" or "QLoRA"
        
    def parse_decision(self, generated_text: str) -> str:
        """
        Parses 'Final Decision: yes/no/maybe' from model output.
        Returns 'yes', 'no', 'maybe', or 'unparseable'.
        """
        # Primary regex: explicit Final Decision label
        match = re.search(r"Final Decision:\s*(yes|no|maybe)", generated_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
            
        # Secondary fallback regex if prefix slightly altered
        match_secondary = re.search(r"\b(decision|conclusion|answer):\s*(yes|no|maybe)\b", generated_text, re.IGNORECASE)
        if match_secondary:
            return match_secondary.group(2).lower()
            
        # Standalone check at end of text
        end_match = re.search(r"\b(yes|no|maybe)[.!]?\s*$", generated_text.strip(), re.IGNORECASE)
        if end_match:
            return end_match.group(1).lower()
            
        return "unparseable"

    def check_abstention(self, text: str) -> bool:
        """Checks if text contains an explicit evidence-deficiency abstention phrase."""
        lowered = text.lower()
        return any(phrase in lowered for phrase in self.config.abstention_proxies)

    def format_prompt(self, question: str, context_text: Optional[str] = None) -> str:
        """Formats the input into ChatML template."""
        if context_text and context_text.strip():
            user_msg = f"Evidence Context:\n{context_text}\n\nQuestion: {question}"
        else:
            user_msg = f"Question: {question}"
            
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return (
                f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def load_model(self, model_type: str = "Base", adapter_path: str = None):
        """Loads Base or QLoRA model with 4-bit quantization."""
        if self.current_model_type == model_type and self.model is not None:
            return  # Already loaded
            
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[Model] Initializing {model_type} model (Device: {'CUDA' if torch.cuda.is_available() else 'CPU'})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_use_double_quant=self.config.use_double_quant,
                bnb_4bit_compute_dtype=torch.float16
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float32
            )

        if model_type == "QLoRA":
            ad_path = adapter_path or self.config.adapter_dir
            print(f"[Model] Attaching QLoRA adapter from {ad_path}...")
            self.model = PeftModel.from_pretrained(base, ad_path)
        else:
            self.model = base

        self.model.eval()
        self.current_model_type = model_type
        print(f"[Model] Successfully loaded {model_type}!")

    def generate_single(self, prompt: str) -> Tuple[str, float]:
        """Runs greedy decoding with latency measurement."""
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.context_budget)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start_time

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return generated_text, latency

    def run_experiment_matrix(self, evidence_cache: List[Dict[str, Any]], 
                              adapter_path: str = None, 
                              max_queries: int = None) -> pd.DataFrame:
        """
        Executes all 10 configurations across test queries with checkpointing.
        """
        os.makedirs(self.config.results_dir, exist_ok=True)
        results = []
        
        # Load existing checkpoint if available
        if os.path.exists(self.config.predictions_path):
            existing_df = pd.read_csv(self.config.predictions_path)
            print(f"[Inference] Found existing checkpoint with {len(existing_df)} predictions.")
            existing_keys = set(zip(existing_df["exp_id"], existing_df["pubid"].astype(str)))
        else:
            existing_df = pd.DataFrame()
            existing_keys = set()

        test_subset = evidence_cache[:max_queries] if max_queries else evidence_cache
        total_configs = len(self.config.experimental_matrix)
        print(f"[Inference] Running {total_configs} configurations over {len(test_subset)} queries...")

        # Group by model type to minimize model reloads
        for model_type in ["Base", "QLoRA"]:
            model_configs = {
                eid: cfg for eid, cfg in self.config.experimental_matrix.items() 
                if cfg["model"] == model_type
            }
            
            # Check if model has pending inferences
            needs_run = any(
                (eid, str(entry["pubid"])) not in existing_keys
                for eid in model_configs
                for entry in test_subset
            )
            if not needs_run:
                continue

            try:
                self.load_model(model_type, adapter_path)
            except Exception as e:
                print(f"[Inference Error] Could not load {model_type} model: {e}")
                continue

            for exp_id, exp_spec in model_configs.items():
                cond = exp_spec["condition"]
                print(f"\n--- Executing {exp_id}: {model_type} | {exp_spec['name']} ({cond}) ---")

                for idx, entry in enumerate(test_subset, 1):
                    pubid = str(entry["pubid"])
                    if (exp_id, pubid) in existing_keys:
                        continue

                    question = entry["question"]
                    gold_decision = entry["gold_decision"]

                    # Skip C2 if query failed incomplete validation
                    if cond == "C2" and not entry["C2"].get("valid_incomplete", True):
                        continue

                    # Context preparation
                    if cond == "None":
                        context_text = None
                        inverted_target = None
                    elif cond == "C4":
                        context_text = entry["C4"]["context_text"]
                        inverted_target = entry["C4"]["inverted_label"]
                    else:
                        context_text = entry[cond]["context_text"]
                        inverted_target = None

                    prompt = self.format_prompt(question, context_text)
                    raw_output, latency = self.generate_single(prompt)
                    parsed_decision = self.parse_decision(raw_output)
                    is_abstaining = self.check_abstention(raw_output)
                    is_correct = (parsed_decision == gold_decision)

                    record = {
                        "exp_id": exp_id,
                        "model": model_type,
                        "condition": cond,
                        "pubid": pubid,
                        "question": question,
                        "gold_decision": gold_decision,
                        "inverted_label": inverted_target or "N/A",
                        "raw_output": raw_output,
                        "parsed_decision": parsed_decision,
                        "is_correct": is_correct,
                        "is_abstaining": is_abstaining,
                        "latency_sec": latency
                    }
                    results.append(record)
                    existing_keys.add((exp_id, pubid))

                    # Checkpoint periodically
                    if len(results) % 25 == 0:
                        self._save_checkpoint(results, existing_df)

        final_df = self._save_checkpoint(results, existing_df)
        print(f"\n[Inference Complete] Total predictions recorded: {len(final_df)}")
        return final_df

    def _save_checkpoint(self, new_results: List[Dict], existing_df: pd.DataFrame) -> pd.DataFrame:
        """Saves combined results to CSV."""
        if new_results:
            new_df = pd.DataFrame(new_results)
            combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
        else:
            combined = existing_df
            
        if not combined.empty:
            combined.drop_duplicates(subset=["exp_id", "pubid"], keep="last", inplace=True)
            combined.to_csv(self.config.predictions_path, index=False)
        return combined
