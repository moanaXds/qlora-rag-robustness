"""
Publication-Quality Visualization Engine
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Generates all required Phase 2 Visualizations:
1. Figure 1: Decision Accuracy & Macro-F1 across C1-C4 with E1/E2 reference lines.
2. Figure 2: Robustness Trade-offs: CAR under C2 vs UCAR and CBFR under C4.
3. Figure 3: Performance Retention degradation curve across C1 -> C2 -> C3 -> C4.
4. Figure 4: Qualitative Error Taxonomy distribution from audited outputs.
5. Figure 5: Per-query inference latency profile across experimental conditions.
"""

import os
from typing import Dict, Any, List
from src.config import ExperimentConfig


class Visualizer:
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        os.makedirs(self.config.figures_dir, exist_ok=True)

    def plot_accuracy_comparison(self, metrics_input: Any, output_path: str = None) -> str:
        """Figure 1: Grouped bar chart of Decision Accuracy across C1-C4 with E1/E2 reference lines."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        metrics_df = metrics_input if isinstance(metrics_input, pd.DataFrame) else pd.DataFrame(metrics_input)

        path = output_path or os.path.join(self.config.figures_dir, "fig1_accuracy_comparison.png")
        
        # Filter for C1-C4
        rag_df = metrics_df[metrics_df["condition"].isin(["C1", "C2", "C3", "C4"])].copy()
        base_df = metrics_df[metrics_df["exp_id"] == "E1"]
        qlora_df = metrics_df[metrics_df["exp_id"] == "E2"]
        
        e1_acc = float(base_df["accuracy"].iloc[0]) if not base_df.empty else None
        e2_acc = float(qlora_df["accuracy"].iloc[0]) if not qlora_df.empty else None

        conditions = ["C1", "C2", "C3", "C4"]
        cond_labels = ["C1: Good", "C2: Incomplete", "C3: Irrelevant", "C4: Misleading"]
        
        base_accs = []
        base_errs_low = []
        base_errs_high = []
        qlora_accs = []
        qlora_errs_low = []
        qlora_errs_high = []

        for c in conditions:
            b_row = rag_df[(rag_df["condition"] == c) & (rag_df["model"] == "Base")]
            q_row = rag_df[(rag_df["condition"] == c) & (rag_df["model"] == "QLoRA")]
            
            b_val = float(b_row["accuracy"].iloc[0]) if not b_row.empty else 0.0
            b_low = float(b_row["acc_ci_low"].iloc[0]) if not b_row.empty else b_val
            b_high = float(b_row["acc_ci_high"].iloc[0]) if not b_row.empty else b_val
            base_accs.append(b_val)
            base_errs_low.append(max(0, b_val - b_low))
            base_errs_high.append(max(0, b_high - b_val))

            q_val = float(q_row["accuracy"].iloc[0]) if not q_row.empty else 0.0
            q_low = float(q_row["acc_ci_low"].iloc[0]) if not q_row.empty else q_val
            q_high = float(q_row["acc_ci_high"].iloc[0]) if not q_row.empty else q_val
            qlora_accs.append(q_val)
            qlora_errs_low.append(max(0, q_val - q_low))
            qlora_errs_high.append(max(0, q_high - q_val))

        x = np.arange(len(conditions))
        width = 0.35

        plt.figure(figsize=(9, 5.5), dpi=300)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        rects1 = plt.bar(x - width/2, base_accs, width, yerr=[base_errs_low, base_errs_high], 
                         capsize=4, label="Frozen Base (Qwen2.5-1.5B)", color="#4A90E2", edgecolor="#2C3E50", alpha=0.9)
        rects2 = plt.bar(x + width/2, qlora_accs, width, yerr=[qlora_errs_low, qlora_errs_high], 
                         capsize=4, label="QLoRA Adapted", color="#E67E22", edgecolor="#2C3E50", alpha=0.9)

        if e1_acc is not None:
            plt.axhline(e1_acc, color="#2980B9", linestyle="--", linewidth=1.5, label=f"E1 Base Q-Only ({e1_acc:.2f})")
        if e2_acc is not None:
            plt.axhline(e2_acc, color="#D35400", linestyle=":", linewidth=1.5, label=f"E2 QLoRA Q-Only ({e2_acc:.2f})")

        plt.ylabel("Decision Accuracy (DA)", fontsize=12, fontweight="bold")
        plt.xlabel("Evidence Condition", fontsize=12, fontweight="bold")
        plt.title("Decision Accuracy Across Controlled Evidence Conditions (95% Bootstrap CI)", fontsize=13, fontweight="bold", pad=12)
        plt.xticks(x, cond_labels, fontsize=11)
        plt.ylim(0.0, 1.05)
        plt.legend(frameon=True, loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[Visualization] Figure 1 saved to {path}")
        return path

    def plot_abstention_and_misleading(self, metrics_input: Any, output_path: str = None) -> str:
        """Figure 2: Two-panel plot: CAR under C2 and UCAR/CBFR under C4."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        metrics_df = metrics_input if isinstance(metrics_input, pd.DataFrame) else pd.DataFrame(metrics_input)

        path = output_path or os.path.join(self.config.figures_dir, "fig2_abstention_and_misleading.png")
        
        c2_base = metrics_df[(metrics_df["condition"] == "C2") & (metrics_df["model"] == "Base")]
        c2_qlora = metrics_df[(metrics_df["condition"] == "C2") & (metrics_df["model"] == "QLoRA")]
        c4_base = metrics_df[(metrics_df["condition"] == "C4") & (metrics_df["model"] == "Base")]
        c4_qlora = metrics_df[(metrics_df["condition"] == "C4") & (metrics_df["model"] == "QLoRA")]

        base_car = float(c2_base["car"].iloc[0]) if not c2_base.empty and c2_base["car"].iloc[0] != "N/A" else 0.0
        qlora_car = float(c2_qlora["car"].iloc[0]) if not c2_qlora.empty and c2_qlora["car"].iloc[0] != "N/A" else 0.0

        base_ucar = float(c4_base["ucar"].iloc[0]) if not c4_base.empty and c4_base["ucar"].iloc[0] != "N/A" else 0.0
        qlora_ucar = float(c4_qlora["ucar"].iloc[0]) if not c4_qlora.empty and c4_qlora["ucar"].iloc[0] != "N/A" else 0.0

        base_cbfr = float(c4_base["cbfr"].iloc[0]) if not c4_base.empty and c4_base["cbfr"].iloc[0] != "N/A" else 0.0
        qlora_cbfr = float(c4_qlora["cbfr"].iloc[0]) if not c4_qlora.empty and c4_qlora["cbfr"].iloc[0] != "N/A" else 0.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

        # Panel 1: Correct Abstention Rate (CAR) under C2
        models = ["Base", "QLoRA"]
        cars = [base_car, qlora_car]
        colors = ["#4A90E2", "#E67E22"]
        bars1 = ax1.bar(models, cars, color=colors, width=0.45, edgecolor="#2C3E50")
        ax1.set_ylabel("Correct Abstention Rate (CAR)", fontsize=11, fontweight="bold")
        ax1.set_title("Panel A: Incomplete Evidence (C2)\nAbstention Capability", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 1.0)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.1%}", ha="center", va="bottom", fontweight="bold")

        # Panel 2: UCAR and CBFR under C4
        x = np.arange(2)
        width = 0.35
        ucars = [base_ucar, qlora_ucar]
        cbfrs = [base_cbfr, qlora_cbfr]

        ax2.bar(x - width/2, ucars, width, label="UCAR (Unconditional)", color="#C0392B", edgecolor="#2C3E50", alpha=0.85)
        ax2.bar(x + width/2, cbfrs, width, label="CBFR (Conditional)", color="#8E44AD", edgecolor="#2C3E50", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=11)
        ax2.set_ylabel("Adoption Rate", fontsize=11, fontweight="bold")
        ax2.set_title("Panel B: Misleading Evidence (C4)\nCounterfactual Adoption Rates", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 1.0)
        ax2.legend(frameon=True, loc="upper right")

        plt.suptitle("Robustness Metrics: Abstention vs. Adversarial Adoption", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[Visualization] Figure 2 saved to {path}")
        return path

    def plot_performance_retention(self, metrics_input: Any, output_path: str = None) -> str:
        """Figure 3: Categorical dot/line plot of performance retention across conditions."""
        import matplotlib.pyplot as plt
        import pandas as pd

        metrics_df = metrics_input if isinstance(metrics_input, pd.DataFrame) else pd.DataFrame(metrics_input)

        path = output_path or os.path.join(self.config.figures_dir, "fig3_performance_retention.png")
        
        conditions = ["C1", "C2", "C3", "C4"]
        cond_labels = ["Good (C1)", "Incomplete (C2)", "Irrelevant (C3)", "Misleading (C4)"]

        base_accs = []
        qlora_accs = []
        for c in conditions:
            b_row = metrics_df[(metrics_df["condition"] == c) & (metrics_df["model"] == "Base")]
            q_row = metrics_df[(metrics_df["condition"] == c) & (metrics_df["model"] == "QLoRA")]
            base_accs.append(float(b_row["accuracy"].iloc[0]) if not b_row.empty else 0.0)
            qlora_accs.append(float(q_row["accuracy"].iloc[0]) if not q_row.empty else 0.0)

        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(cond_labels, base_accs, marker="o", markersize=8, linewidth=2.5, color="#4A90E2", label="Frozen Base")
        plt.plot(cond_labels, qlora_accs, marker="s", markersize=8, linewidth=2.5, color="#E67E22", label="QLoRA Adapted")

        plt.ylabel("Decision Accuracy (DA)", fontsize=12, fontweight="bold")
        plt.xlabel("Evidence Reliability Condition", fontsize=12, fontweight="bold")
        plt.title("Performance Retention Curve Across Evidence Quality Degradation", fontsize=13, fontweight="bold", pad=12)
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True, fontsize=11)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[Visualization] Figure 3 saved to {path}")
        return path

    def plot_error_distribution(self, audit_input: Any, output_path: str = None) -> str:
        """Figure 4: Qualitative error taxonomy distribution by model and condition."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        audit_df = audit_input if isinstance(audit_input, pd.DataFrame) else pd.DataFrame(audit_input)

        path = output_path or os.path.join(self.config.figures_dir, "fig4_error_distribution.png")
        if audit_df.empty or "audit_category" not in audit_df:
            return ""

        categories = [
            "Grounded-Correct", "Parametric-Correct", "Clean Abstention",
            "Bad-Evidence Exploited", "Ungrounded Hallucination", "Other Error"
        ]
        
        base_audit = audit_df[audit_df["model"] == "Base"]
        qlora_audit = audit_df[audit_df["model"] == "QLoRA"]

        base_counts = [int((base_audit["audit_category"] == cat).sum()) for cat in categories]
        qlora_counts = [int((qlora_audit["audit_category"] == cat).sum()) for cat in categories]

        x = np.arange(len(categories))
        width = 0.38

        plt.figure(figsize=(10, 5.5), dpi=300)
        plt.bar(x - width/2, base_counts, width, label="Base Reader", color="#4A90E2", edgecolor="#2C3E50")
        plt.bar(x + width/2, qlora_counts, width, label="QLoRA Reader", color="#E67E22", edgecolor="#2C3E50")

        plt.ylabel("Output Count (N = 100 Sample)", fontsize=12, fontweight="bold")
        plt.title("Distribution of Audited Error and Grounding Categories", fontsize=13, fontweight="bold", pad=12)
        plt.xticks(x, categories, rotation=25, ha="right", fontsize=10, fontweight="bold")
        plt.legend(frameon=True, fontsize=11)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[Visualization] Figure 4 saved to {path}")
        return path
