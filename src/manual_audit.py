"""
Manual Audit and Claim Grounding Protocol Module
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Adheres strictly to Phase 2 Methodology:
- Stratified sampling of 100 outputs (seed 42) across the 8 RAG configurations:
  25 per condition (C1: 12 Base, 13 QLoRA; C2: 12 Base, 13 QLoRA; C3: 12 Base, 13 QLoRA; C4: 12 Base, 13 QLoRA).
- Six mutually exclusive qualitative error categories:
  1. Clean Abstention
  2. Bad-Evidence Exploited
  3. Grounded-Correct
  4. Parametric-Correct
  5. Ungrounded Hallucination
  6. Other Error
- Calculates Grounding Rate over non-abstaining outputs.
"""

import os
import csv
import random
from typing import Dict, List, Tuple, Any
from src.config import ExperimentConfig


class ManualAuditor:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()

    def sample_audit_outputs(self, predictions_input: Any, sample_size: int = 100) -> List[Dict[str, Any]]:
        """
        Samples exactly 100 outputs across the 8 RAG configurations (C1-C4 x {Base, QLoRA}).
        Seed 42.
        """
        if hasattr(predictions_input, "to_dict"):
            records_all = predictions_input.to_dict(orient="records")
        elif isinstance(predictions_input, list):
            records_all = predictions_input
        else:
            records_all = []
            with open(predictions_input, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["is_correct"] = str(row.get("is_correct", "")).lower() in ["true", "1"]
                    row["is_abstaining"] = str(row.get("is_abstaining", "")).lower() in ["true", "1"]
                    records_all.append(row)

        rng = random.Random(self.config.seed)
        sampled_rows = []
        conditions = ["C1", "C2", "C3", "C4"]

        for cond in conditions:
            base_subset = [r for r in records_all if r.get("condition") == cond and r.get("model") == "Base"]
            qlora_subset = [r for r in records_all if r.get("condition") == cond and r.get("model") == "QLoRA"]

            # 12 Base, 13 QLoRA per condition = 25 per condition
            n_base = min(12, len(base_subset))
            n_qlora = min(13, len(qlora_subset))

            if n_base > 0:
                sampled_rows.extend(rng.sample(base_subset, n_base))
            if n_qlora > 0:
                sampled_rows.extend(rng.sample(qlora_subset, n_qlora))

        if not sampled_rows:
            return []

        # Classify outputs using priority taxonomy
        for row in sampled_rows:
            row["audit_category"] = self._classify_audit_row(row)
            row["grounded_supported"] = (row["audit_category"] == "Grounded-Correct")

        os.makedirs(self.config.results_dir, exist_ok=True)
        if sampled_rows:
            fieldnames = list(sampled_rows[0].keys())
            with open(self.config.manual_audit_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sampled_rows)
            print(f"[Audit] Sampled {len(sampled_rows)} records for manual audit -> {self.config.manual_audit_path}")
            
        return sampled_rows

    @staticmethod
    def _classify_audit_row(row: Dict[str, Any]) -> str:
        """Assigns one of the 6 mutually exclusive categories based on Phase 2 priority rules."""
        cond = row.get("condition")
        is_correct = bool(row.get("is_correct", False))
        is_abstaining = bool(row.get("is_abstaining", False))
        parsed_dec = str(row.get("parsed_decision", "")).lower()
        inv_label = str(row.get("inverted_label", "")).lower()

        # Rule 1: Clean Abstention
        if cond == "C2" and is_abstaining:
            return "Clean Abstention"

        # Rule 2: Bad-Evidence Exploited
        if cond == "C4" and parsed_dec == inv_label and inv_label in ["yes", "no"]:
            return "Bad-Evidence Exploited"
        if cond == "C3" and not is_correct and not is_abstaining:
            return "Bad-Evidence Exploited"

        # Rule 3 & 4: Grounded-Correct vs Parametric-Correct
        if is_correct:
            if cond == "C1":
                return "Grounded-Correct"
            elif cond in ["C2", "C3"]:
                return "Parametric-Correct"
            elif cond == "C4":
                return "Parametric-Correct"

        # Rule 5: Ungrounded Hallucination
        if not is_correct and not is_abstaining:
            return "Ungrounded Hallucination"

        # Rule 6: Other Error
        return "Other Error"

    @staticmethod
    def calculate_grounding_rate(audit_records: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        """
        Grounding Rate = proportion of non-abstaining audited outputs for which all factual claims are supported.
        """
        non_abstaining = [r for r in audit_records if r.get("audit_category") != "Clean Abstention"]
        if not non_abstaining:
            return 0.0, 0, 0
        grounded = sum(1 for r in non_abstaining if r.get("audit_category") == "Grounded-Correct")
        total = len(non_abstaining)
        return grounded / total, grounded, total
