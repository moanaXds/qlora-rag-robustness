"""
Ablation Study Module
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Implements the required ablation analyses:
1. Question-Only vs. Evidence-Conditioned RAG (E1/E2 vs. E3/E7).
2. Evidence Quality Sensitivity & Degradation Deltas (C1 -> C2 -> C3 -> C4).
Documents what each ablation isolates and what it cannot establish.
"""

import os
import csv
from typing import Dict, List, Any
from src.config import ExperimentConfig


class AblationAnalyzer:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()

    def run_ablations(self, metrics_input: Any) -> List[Dict[str, Any]]:
        """
        Computes formal ablation deltas between question-only and evidence-conditioned configurations.
        """
        if hasattr(metrics_input, "to_dict"):
            metrics_records = metrics_input.to_dict(orient="records")
        elif isinstance(metrics_input, list):
            metrics_records = metrics_input
        else:
            metrics_records = []
            with open(metrics_input, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_records.append(row)

        by_exp = {r["exp_id"]: r for r in metrics_records}
        records = []

        # 1. Ablation 1: Parametric Question-Only vs RAG Grounding
        e1_acc = float(by_exp.get("E1", {}).get("accuracy", 0.0))
        e2_acc = float(by_exp.get("E2", {}).get("accuracy", 0.0))
        e3_acc = float(by_exp.get("E3", {}).get("accuracy", 0.0))
        e7_acc = float(by_exp.get("E7", {}).get("accuracy", 0.0))

        records.append({
            "ablation_id": "ABL-1A",
            "name": "Base: Question-Only vs RAG (Good Evidence)",
            "comparison": "E3 (RAG C1) - E1 (Q-Only)",
            "baseline_accuracy": e1_acc,
            "conditioned_accuracy": e3_acc,
            "delta_accuracy": round(e3_acc - e1_acc, 4),
            "isolates": "Net performance gain from intact retrieved context over frozen parametric memory.",
            "limitation": "Cannot isolate whether gain is genuine reasoning or lexical pattern matching."
        })

        records.append({
            "ablation_id": "ABL-1B",
            "name": "QLoRA: Question-Only vs RAG (Good Evidence)",
            "comparison": "E7 (RAG C1) - E2 (Q-Only)",
            "baseline_accuracy": e2_acc,
            "conditioned_accuracy": e7_acc,
            "delta_accuracy": round(e7_acc - e2_acc, 4),
            "isolates": "Impact of retrieved evidence on adapter fine-tuned with context block.",
            "limitation": "E2 suffers input-distribution shift since training included context blocks."
        })

        # 2. Ablation 2: Robustness Degradation under Corrupted Evidence
        for model in ["Base", "QLoRA"]:
            exp_map = {
                "Base": {"C1": "E3", "C2": "E4", "C3": "E5", "C4": "E6"},
                "QLoRA": {"C1": "E7", "C2": "E8", "C3": "E9", "C4": "E10"}
            }
            c1_exp = exp_map[model]["C1"]
            c2_exp = exp_map[model]["C2"]
            c3_exp = exp_map[model]["C3"]
            c4_exp = exp_map[model]["C4"]

            acc_c1 = float(by_exp.get(c1_exp, {}).get("accuracy", 0.0))
            acc_c2 = float(by_exp.get(c2_exp, {}).get("accuracy", 0.0))
            acc_c3 = float(by_exp.get(c3_exp, {}).get("accuracy", 0.0))
            acc_c4 = float(by_exp.get(c4_exp, {}).get("accuracy", 0.0))

            records.append({
                "ablation_id": f"ABL-2_{model}_C2",
                "name": f"{model}: Good vs Incomplete Evidence Degradation",
                "comparison": f"{model} (C1) - {model} (C2)",
                "baseline_accuracy": acc_c1,
                "conditioned_accuracy": acc_c2,
                "delta_accuracy": round(acc_c2 - acc_c1, 4),
                "isolates": "Vulnerability to missing findings and outcome suppression.",
                "limitation": "Does not prove whether model knows what is missing without manual audit."
            })

            records.append({
                "ablation_id": f"ABL-2_{model}_C3",
                "name": f"{model}: Good vs Irrelevant Evidence Degradation",
                "comparison": f"{model} (C1) - {model} (C3)",
                "baseline_accuracy": acc_c1,
                "conditioned_accuracy": acc_c3,
                "delta_accuracy": round(acc_c3 - acc_c1, 4),
                "isolates": "Sensitivity to off-target distractor interference.",
                "limitation": "Heuristic lexical filtering cannot guarantee zero semantic overlap."
            })

            records.append({
                "ablation_id": f"ABL-2_{model}_C4",
                "name": f"{model}: Good vs Misleading Evidence Degradation",
                "comparison": f"{model} (C1) - {model} (C4)",
                "baseline_accuracy": acc_c1,
                "conditioned_accuracy": acc_c4,
                "delta_accuracy": round(acc_c4 - acc_c1, 4),
                "isolates": "Vulnerability to direct counterfactual evidence injection.",
                "limitation": "Restricted to synthetic outcome-inversion templates."
            })

        out_path = os.path.join(self.config.results_dir, "ablation_results.csv")
        os.makedirs(self.config.results_dir, exist_ok=True)
        if records:
            fieldnames = list(records[0].keys())
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            print(f"[Ablation] Analysis saved to {out_path}")
            
        return records
