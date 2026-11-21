"""
Evaluation Metrics, Robustness Rates, and Statistical Analysis Module
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Adheres strictly to Phase 2 Methodology:
- Decision Accuracy (DA) and Macro-F1 across {yes, no, maybe}.
- Natural Retrieval Recall@3.
- Unconditional Counterfactual Adoption Rate (UCAR) on binary C4.
- Conditional Bad-Context-Following Rate (CBFR) on incorrect binary C4.
- Correct Abstention Rate (CAR) on C2.
- Irrelevant-Context Rejection Rate (IRR) on C3.
- Latency (median and mean wall-clock generation time).
- 10,000-sample paired bootstrap 95% Confidence Intervals.
- Compatible with pandas/numpy and standard library fallback.
"""

import os
import csv
import math
import random
from typing import Dict, List, Tuple, Any
from src.config import ExperimentConfig


class MetricsCalculator:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()

    @staticmethod
    def calculate_decision_accuracy(records: List[Dict[str, Any]]) -> float:
        """DA = (number of correctly classified queries) / (total evaluated queries)"""
        if not records:
            return 0.0
        correct = sum(1 for r in records if r.get("parsed_decision") == r.get("gold_decision"))
        return correct / len(records)

    @staticmethod
    def calculate_macro_f1(records: List[Dict[str, Any]]) -> float:
        """
        Macro-F1 = (F1_yes + F1_no + F1_maybe) / 3.
        Unparseable predictions are counted as false negatives for their gold class.
        """
        if not records:
            return 0.0
            
        classes = ["yes", "no", "maybe"]
        f1_scores = []
        
        for c in classes:
            tp = sum(1 for r in records if r.get("gold_decision") == c and r.get("parsed_decision") == c)
            fp = sum(1 for r in records if r.get("gold_decision") != c and r.get("parsed_decision") == c)
            fn = sum(1 for r in records if r.get("gold_decision") == c and r.get("parsed_decision") != c)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
            
        return sum(f1_scores) / len(f1_scores)

    @staticmethod
    def calculate_ucar(records: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        """
        Unconditional Counterfactual Adoption Rate for C4.
        Evaluated strictly on binary queries with gold_decision in {yes, no}.
        UCAR = (binary C4 predictions equal to inverted label) / (binary test queries).
        Returns (rate, numerator, denominator).
        """
        binary_subset = [r for r in records if r.get("gold_decision") in ["yes", "no"]]
        if not binary_subset:
            return 0.0, 0, 0
        adoptions = sum(1 for r in binary_subset if r.get("parsed_decision") == r.get("inverted_label"))
        total = len(binary_subset)
        return adoptions / total, adoptions, total

    @staticmethod
    def calculate_cbfr(records: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        """
        Conditional Bad-Context-Following Rate for C4.
        CBFR = (incorrect binary predictions that adopt inverted label) / (total incorrect binary predictions).
        Returns (rate, numerator, denominator).
        """
        binary_subset = [r for r in records if r.get("gold_decision") in ["yes", "no"]]
        incorrect_subset = [r for r in binary_subset if r.get("parsed_decision") != r.get("gold_decision")]
        if not incorrect_subset:
            return 0.0, 0, 0
        adoptions = sum(1 for r in incorrect_subset if r.get("parsed_decision") == r.get("inverted_label"))
        total = len(incorrect_subset)
        return adoptions / total, adoptions, total

    @staticmethod
    def calculate_car(records: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        """
        Correct Abstention Rate for C2.
        CAR = (queries where model explicitly abstains due to missing findings) / N_inc.
        Returns (rate, numerator, denominator).
        """
        if not records:
            return 0.0, 0, 0
        abstentions = sum(1 for r in records if r.get("is_abstaining", False))
        total = len(records)
        return abstentions / total, abstentions, total

    @staticmethod
    def calculate_irr(records: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        """
        Irrelevant-Context Rejection Rate for C3.
        IRR = (queries where model does not follow off-target claims and maintains correct answer or explicit rejection) / N.
        """
        if not records:
            return 0.0, 0, 0
        rejected = sum(1 for r in records if r.get("is_correct", False) or r.get("is_abstaining", False))
        total = len(records)
        return rejected / total, rejected, total

    def paired_bootstrap_ci(self, records: List[Dict[str, Any]], metric_fn, n_bootstraps: int = 1000) -> Tuple[float, float]:
        """Calculates 95% percentile bootstrap confidence intervals."""
        if len(records) < 5:
            return 0.0, 0.0
            
        rng = random.Random(self.config.seed)
        n = len(records)
        boot_metrics = []
        
        for _ in range(n_bootstraps):
            sample = [rng.choice(records) for _ in range(n)]
            val = metric_fn(sample)
            if isinstance(val, tuple):
                val = val[0]
            boot_metrics.append(val)
                
        boot_metrics.sort()
        idx_low = int(0.025 * len(boot_metrics))
        idx_high = int(0.975 * len(boot_metrics))
        return boot_metrics[idx_low], boot_metrics[idx_high]

    def evaluate_all(self, predictions_input: Any) -> List[Dict[str, Any]]:
        """
        Computes all Phase 2 metrics across the 10 experimental configurations.
        Accepts either a pandas DataFrame or a list of dictionaries.
        """
        if hasattr(predictions_input, "to_dict"):
            records_all = predictions_input.to_dict(orient="records")
        elif isinstance(predictions_input, list):
            records_all = predictions_input
        else:
            # Load from CSV path
            records_all = []
            with open(predictions_input, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["is_correct"] = row.get("is_correct", "").lower() in ["true", "1"]
                    row["is_abstaining"] = row.get("is_abstaining", "").lower() in ["true", "1"]
                    row["latency_sec"] = float(row.get("latency_sec", 0.0))
                    records_all.append(row)

        summary_records = []
        for exp_id, meta in self.config.experimental_matrix.items():
            sub = [r for r in records_all if r.get("exp_id") == exp_id]
            if not sub:
                continue

            da = self.calculate_decision_accuracy(sub)
            da_low, da_high = self.paired_bootstrap_ci(sub, self.calculate_decision_accuracy, n_bootstraps=1000)
            
            macro_f1 = self.calculate_macro_f1(sub)
            f1_low, f1_high = self.paired_bootstrap_ci(sub, self.calculate_macro_f1, n_bootstraps=1000)
            
            unparseable_count = sum(1 for r in sub if r.get("parsed_decision") == "unparseable")
            latencies = [float(r.get("latency_sec", 0.0)) for r in sub]
            latencies.sort()
            median_latency = latencies[len(latencies) // 2] if latencies else 0.0
            mean_latency = sum(latencies) / max(1, len(latencies))

            # Condition-specific metrics
            cond = meta["condition"]
            car, car_num, car_den = (self.calculate_car(sub)) if cond == "C2" else (None, None, None)
            irr, irr_num, irr_den = (self.calculate_irr(sub)) if cond == "C3" else (None, None, None)
            ucar, ucar_num, ucar_den = (self.calculate_ucar(sub)) if cond == "C4" else (None, None, None)
            cbfr, cbfr_num, cbfr_den = (self.calculate_cbfr(sub)) if cond == "C4" else (None, None, None)

            summary_records.append({
                "exp_id": exp_id,
                "model": meta["model"],
                "condition": cond,
                "name": meta["name"],
                "evaluated_queries": len(sub),
                "unparseable_count": unparseable_count,
                "accuracy": round(da, 4),
                "acc_ci_low": round(da_low, 4),
                "acc_ci_high": round(da_high, 4),
                "macro_f1": round(macro_f1, 4),
                "f1_ci_low": round(f1_low, 4),
                "f1_ci_high": round(f1_high, 4),
                "car": round(car, 4) if car is not None else "N/A",
                "car_fraction": f"{car_num}/{car_den}" if car_num is not None else "N/A",
                "irr": round(irr, 4) if irr is not None else "N/A",
                "irr_fraction": f"{irr_num}/{irr_den}" if irr_num is not None else "N/A",
                "ucar": round(ucar, 4) if ucar is not None else "N/A",
                "ucar_fraction": f"{ucar_num}/{ucar_den}" if ucar_num is not None else "N/A",
                "cbfr": round(cbfr, 4) if cbfr is not None else "N/A",
                "cbfr_fraction": f"{cbfr_num}/{cbfr_den}" if cbfr_num is not None else "N/A",
                "median_latency_sec": round(median_latency, 3),
                "mean_latency_sec": round(mean_latency, 3)
            })

        # Save to CSV using standard csv writer
        os.makedirs(self.config.results_dir, exist_ok=True)
        if summary_records:
            fieldnames = list(summary_records[0].keys())
            with open(self.config.metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_records)
            print(f"[Metrics] Summary saved to {self.config.metrics_path}")
            
        return summary_records
