#!/usr/bin/env python3
"""
generate_project_report_pdf.py

Builds a clean, plain black-and-white PDF report (Phase_3_Project_Report.pdf)
using ReportLab.
Strictly adheres to:
1. Standard Times-Roman font, strictly black text, plain black table borders with white cells.
2. Direct, humanized student tone with no AI buzzwords.
3. ZERO em dashes (\\u2014) or en dashes (\\u2013) anywhere in the document.
4. Clean running headers and page numbers ("Page X of Y").
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8.5)
        self.setFillColor(colors.black)
        
        # Omit header and footer on cover / title page if first page
        if self._pageNumber > 1:
            # Running header
            header_text = "Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence"
            self.drawString(1.0 * 72, 10.35 * 72, header_text)
            self.setStrokeColor(colors.black)
            self.setLineWidth(0.5)
            self.line(1.0 * 72, 10.28 * 72, 7.5 * 72, 10.28 * 72)
            
            # Running footer
            footer_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(7.5 * 72, 0.55 * 72, footer_text)
            self.drawString(1.0 * 72, 0.55 * 72, "Phase 3 Implementation and Results Report")
            self.line(1.0 * 72, 0.68 * 72, 7.5 * 72, 0.68 * 72)
        else:
            footer_text = f"Page 1 of {page_count}"
            self.drawRightString(7.5 * 72, 0.55 * 72, footer_text)
            
        self.restoreState()

def clean_text(text: str) -> str:
    """Strict sanitizer guaranteeing zero em dashes or en dashes."""
    return str(text).replace("\u2014", ", ").replace("\u2013", ", ")

def build_pdf_report():
    output_path = os.path.join(os.getcwd(), "Phase_3_Project_Report.pdf")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=1.0 * 72,
        rightMargin=1.0 * 72,
        topMargin=0.85 * 72,
        bottomMargin=0.85 * 72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Plain Black and White Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12.5,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=10
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=13.5,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=2
    )
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10.5,
        leading=14.5,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10.5,
        leading=14,
        textColor=colors.black,
        leftIndent=15,
        spaceAfter=3
    )
    caption_style = ParagraphStyle(
        'CustomCaption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceBefore=3,
        spaceAfter=8
    )
    tbl_cell = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    tbl_cell_left = ParagraphStyle(
        'TblCellLeft',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.black
    )
    tbl_header = ParagraphStyle(
        'TblHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    tbl_note_style = ParagraphStyle(
        'TblNote',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceBefore=3,
        spaceAfter=6
    )

    story = []
    
    # ------------------------------------------------------------------------------------------------
    # 1. Title Block and Project Overview
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence"), title_style))
    story.append(Paragraph(clean_text("Phase 3 Implementation and Results Report"), subtitle_style))
    story.append(Spacer(1, 4))
    
    meta_lines = [
        "Student: Muanna Hamid (Roll No: 242593)",
        "Course Instructor and Supervisor: Prof. Azka Atiq",
        "Department of Data Science, Faculty of Computing",
        "National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad",
        "Degree Program: BS Data Science (5th Semester, Fall 2026)",
        "Course: Data Analysis and Visualization (Deliverable 3)"
    ]
    for m in meta_lines:
        story.append(Paragraph(clean_text(m), meta_style))
    story.append(Spacer(1, 8))
    
    story.append(Paragraph(clean_text("1. Project Overview and Context"), h1_style))
    story.append(Paragraph(clean_text(
        "In Retrieval-Augmented Generation (RAG) systems, small language models are frequently adapted using "
        "parameter-efficient fine-tuning methods such as Quantized Low-Rank Adaptation (QLoRA). While fine-tuning "
        "is widely used to teach models domain-specific vocabulary and output formatting, its impact on model "
        "robustness during noisy or adversarial retrieval is not well understood."
    ), body_style))
    
    story.append(Paragraph(clean_text(
        "Phase 1 reviewed ten relevant research papers published in 2025 and 2026, including InstructRAG, RbFT, "
        "and RAGuard. These papers proposed complex multi-step architectures, noise-injected training datasets, "
        "and defensive filtering networks to protect models from corrupted retrieval context. However, they left "
        "a clear gap: how does plain, standard question-answering instruction tuning with QLoRA affect a model "
        "when retrieval fails? Does ordinary fine-tuning help the model rely on evidence, or does it make the "
        "model excessively dependent on whatever context is placed in front of it, even when that context is defective?"
    ), body_style))
    
    story.append(Paragraph(clean_text(
        "Phase 2 designed a controlled, deterministic experiment to answer this question. The benchmark evaluates a "
        "compact open-source model (Qwen2.5-1.5B-Instruct) across ten experimental configurations (E1 to E10) under "
        "four carefully engineered evidence conditions (C1 to C4). Phase 3 completes this research pipeline by running "
        "the entire benchmark on Google Colab using genuine MEDLINE records, computing paired statistical hypothesis "
        "tests, running ablation studies, and performing a manual grounding audit."
    ), body_style))
    
    story.append(Paragraph(clean_text("This project directly addresses three research questions:"), body_style))
    story.append(Paragraph(clean_text("1. Research Question 1 (RQ1): Does ordinary QLoRA fine-tuning improve answer accuracy when retrieved evidence is reliable?"), bullet_style))
    story.append(Paragraph(clean_text("2. Research Question 2 (RQ2): Does QLoRA fine-tuning make the model more likely to follow misleading or counterfactual evidence?"), bullet_style))
    story.append(Paragraph(clean_text("3. Research Question 3 (RQ3): Does QLoRA improve the model's ability to reject irrelevant background information and abstain when evidence is incomplete?"), bullet_style))
    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------------------------------------
    # 2. Dataset, Preprocessing, and Retrieval Setup
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("2. Dataset, Preprocessing, and Retrieval Setup"), h1_style))
    story.append(Paragraph(clean_text(
        "The experiment uses the genuine PubMedQA dataset (the pqa_labeled subset), downloaded directly from Hugging "
        "Face (qiaojin/PubMedQA). This subset contains exactly 1,000 real biomedical question-answering records collected "
        "from MEDLINE abstracts. Every record includes a PubMed identifier (pubid), a clinical research question, an abstract "
        "partitioned into labeled context sections, a long expert answer, and a final consensus decision (yes, no, or maybe). "
        "All synthetic data generation code, placeholder biomarker strings, and fake numerical identifiers were removed from the project."
    ), body_style))
    
    story.append(Paragraph(clean_text(
        "The 1,000 records were sorted lexicographically by pubid to guarantee deterministic ordering, and then partitioned "
        "into 800 training records and 200 held-out test records using a fixed pseudo-random seed of 42. Strict data leakage "
        "guardrails were maintained throughout: the expert long_answer field was quarantined from all BM25 retrieval indices "
        "and evaluation contexts, and the BM25 background search corpus was built exclusively from the 800 training abstracts."
    ), body_style))
    
    story.append(Paragraph(clean_text("2.1 Class Label Distributions"), h2_style))
    story.append(Paragraph(clean_text(
        "Table 1 shows the exact distribution of gold decision labels across the training and test splits. In both partitions, "
        "positive outcomes (yes) represent the majority class, followed by negative outcomes (no) and unresolved or neutral outcomes (maybe)."
    ), body_style))

    # Table 1: Dataset Splits
    t1_data = [
        [Paragraph(clean_text(h), tbl_header) for h in ["Dataset Split", "Total Records", "Yes Count", "No Count", "Maybe Count", "Majority Baseline"]],
        [Paragraph(clean_text("Training Split"), tbl_cell_left), Paragraph("800", tbl_cell), Paragraph("444 (55.5%)", tbl_cell), Paragraph("263 (32.9%)", tbl_cell), Paragraph("93 (11.6%)", tbl_cell), Paragraph("55.50%", tbl_cell)],
        [Paragraph(clean_text("Held-Out Test Split"), tbl_cell_left), Paragraph("200", tbl_cell), Paragraph("108 (54.0%)", tbl_cell), Paragraph("75 (37.5%)", tbl_cell), Paragraph("17 (8.5%)", tbl_cell), Paragraph("54.00%", tbl_cell)]
    ]
    t1 = Table(t1_data, colWidths=[105, 65, 80, 80, 75, 85])
    t1.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1)
    story.append(Paragraph(clean_text("Table 1. Distribution of consensus decision labels in genuine PubMedQA (seed 42)."), tbl_note_style))

    story.append(Paragraph(clean_text("2.2 Retrieval Engine and Recall@3 Benchmark"), h2_style))
    story.append(Paragraph(clean_text(
        "Retrieval was implemented using Okapi BM25 (rank-bm25) with standard parameters (k1 = 1.5, b = 0.75, and epsilon = 0.25). "
        "Tokenization extracted alphanumeric words of two or more characters, converted all tokens to lower case, and filtered out "
        "standard English stopwords from NLTK. Before constructing evidence packets, pure retrieval coverage was benchmarked over "
        "the entire 1,000-abstract collection. For 192 out of the 200 test queries, the target abstract was successfully retrieved "
        "within the top 3 BM25 search results, establishing a pure Recall@3 of 0.9600 (96.0%)."
    ), body_style))

    story.append(Paragraph(clean_text("2.3 Construction of Controlled Evidence Packets (C1 to C4)"), h2_style))
    story.append(Paragraph(clean_text("For each of the 200 test queries, four distinct 3-passage evidence packets were built and stored in data/evidence_cache.jsonl:"), body_style))
    story.append(Paragraph(clean_text(
        "1. Condition C1 (Good Evidence): Contains the genuine intact target abstract with its conclusions fully preserved, "
        "combined with the top two BM25 distractor abstracts retrieved from the 800-record training corpus (excluding the test query pubid). "
        "The three passage slots were shuffled using a deterministic seed (42 + query_index * 17)."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "2. Condition C2 (Incomplete Evidence): Contains the target abstract with all outcome-bearing and conclusion text stripped away. "
        "In structured abstracts, sections labeled CONCLUSIONS, RESULTS, DISCUSSION, or OUTCOMES were removed. For unstructured abstracts, "
        "explicit conclusion sentences were removed. Out of 200 test records, exactly N_inc = 36 records had explicit conclusion sections "
        "identified and removed. These 36 cases were flagged with valid_incomplete = True for calculating the Correct Abstention Rate (CAR). "
        "The remaining two passage slots were filled with distractor passages."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "3. Condition C3 (Irrelevant Evidence): Contains three low-relevance biomedical abstracts sampled deterministically from ranks 50 "
        "to 150 of the BM25 retrieval ranking for that query (excluding the gold abstract). This guarantees genuine off-topic biomedical "
        "context without sharing high keyword overlap with the test question."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "4. Condition C4 (Misleading Evidence): Contains the target abstract where the main clinical conclusion was inverted in place "
        "without fabricating external cohorts or fake numbers. Out of 200 test records, 186 were transformed via explicit polarity "
        "inversions, 10 through lexical negation swaps (such as changing 'is effective' to 'is not effective'), and 4 through antonym "
        "replacements (such as changing 'demonstrated significant improvement' to 'failed to demonstrate improvement'). The remaining two "
        "slots used the exact same BM25 distractors and permutation order as C1."
    ), bullet_style))
    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------------------------------------
    # 3. Model Implementation and QLoRA Training
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("3. Model Implementation and QLoRA Training"), h1_style))
    story.append(Paragraph(clean_text(
        "The reader model is Qwen/Qwen2.5-1.5B-Instruct, an open-source transformer architecture with 28 causal self-attention blocks "
        "and 1,548,072,448 total parameters. The model was loaded in 4-bit NormalFloat (NF4) quantization using BitsAndBytesConfig with "
        "double quantization enabled and compute dtype set to float16 on a single Google Colab Tesla T4 GPU (14.56 GB VRAM)."
    ), body_style))
    
    story.append(Paragraph(clean_text(
        "Low-Rank Adaptation (LoRA) adapters were injected into all attention projection weights (q_proj, k_proj, v_proj, and o_proj) "
        "across all 28 layers. The LoRA hyperparameters followed the Phase 2 specification: rank r = 16, alpha = 32 (alpha / r = 2.0), "
        "and dropout = 0.05. This configuration introduced 4,358,144 trainable parameters, representing exactly 0.282% of the total base "
        "model architecture. All adapter weights and gradients were maintained in float32 for numerical stability."
    ), body_style))

    story.append(Paragraph(clean_text("3.1 Training Protocol and Rationale Supervision"), h2_style))
    story.append(Paragraph(clean_text(
        "Fine-tuning was conducted on the 800 training examples using Hugging Face Trainer. Unlike earlier setups that trained exclusively "
        "on bare labels, the training target was formulated with concise expert reasoning extracted from PubMedQA:"
    ), body_style))
    story.append(Paragraph(clean_text("Rationale: {concise_long_answer}<br/>Final Decision: {final_decision}"), ParagraphStyle('CenterCode', parent=body_style, alignment=TA_CENTER, fontName='Times-Italic')))
    story.append(Paragraph(clean_text(
        "Long answers were truncated cleanly at two sentences or roughly 350 characters so the compact 1.5B model could learn concise "
        "evidence synthesis. Training examples were structured using Qwen2.5 ChatML formatting. Prompt-token loss masking was applied "
        "by setting labels to -100 for all system and user prompt tokens. As a result, cross-entropy loss was calculated strictly on the assistant response."
    ), body_style))
    story.append(Paragraph(clean_text(
        "Training ran for 2.0 epochs (200 optimizer update steps total) with per-device batch size 1 and 8 gradient accumulation steps "
        "(effective batch size 8). The optimizer was 8-bit Paged AdamW with an initial learning rate of 2e-4, linear learning rate decay, "
        "and 10 warmup steps (5% of planned steps). Total training runtime was 1,027.9 seconds (approximately 17.1 minutes)."
    ), body_style))

    story.append(Paragraph(clean_text("3.2 Training Loss and Checkpoint Selection"), h2_style))
    story.append(Paragraph(clean_text(
        "Training loss was recorded every 10 steps and saved to results/training_loss_history.csv. The loss started at 2.0426 at step 10, "
        "dropped steadily to 1.5592 at step 100 (completion of Epoch 1), reached an empirical minimum of 1.4108 at step 190, and finished "
        "at 1.5455 at step 200 (completion of Epoch 2). In accordance with the Phase 2 plan, the Epoch 1 checkpoint (step 100) was saved "
        "and used as the primary QLoRA reader for all main evaluations (E2, E7, E8, E9, and E10). The Epoch 2 checkpoint was preserved "
        "separately to conduct an ablation study comparing 100-step versus 200-step adaptation."
    ), body_style))
    story.append(Paragraph(clean_text(
        "Output diversity was verified after inference. Generated responses were checked across all conditions. The base model produced "
        "between 142 and 197 unique rationales per condition. The QLoRA model generated 200 out of 200 unique rationales in every evaluated "
        "condition, confirming that the model generated free-form explanations rather than repeating fixed boilerplate text."
    ), body_style))
    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------------------------------------
    # 4. Baseline vs. QLoRA Evaluation and Statistical Testing
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("4. Baseline vs. QLoRA Evaluation and Statistical Testing"), h1_style))
    story.append(Paragraph(clean_text(
        "Evaluation was carried out across 14 configurations: the 10 primary conditions specified in Phase 2 (E1 through E10) plus 4 additional "
        "conditions evaluating the Epoch 2 adapter under C1 through C4 (E7_ep2 through E10_ep2). Inference used greedy decoding (do_sample = False) "
        "with a generation budget of max_new_tokens = 160 and left-padded batching (batch size 16). GPU-synchronized wall-clock latencies were tracked per query."
    ), body_style))
    story.append(Paragraph(clean_text(
        "In addition to overall Decision Accuracy (DA) and Macro-F1 across the three decision classes (yes, no, maybe), evaluation tracked two abstention-aware metrics:"
    ), body_style))
    story.append(Paragraph(clean_text("1. Coverage: The fraction of queries where the model committed to a definitive answer rather than abstaining (parsed_decision != 'maybe')."), bullet_style))
    story.append(Paragraph(clean_text("2. Committed Accuracy: The accuracy calculated strictly on the subset of queries where the model made a definitive commitment."), bullet_style))
    story.append(Paragraph(clean_text("3. Correct Abstention Rate (CAR): The proportion of valid incomplete cases (N_inc = 36) where the model explicitly abstained by predicting 'maybe' or expressing evidence insufficiency."), bullet_style))

    story.append(Paragraph(clean_text("4.1 Quantitative Performance Matrix"), h2_style))

    # Table 2: 14 Configurations
    t2_headers = ["Exp ID", "Model", "Condition", "Accuracy", "Coverage", "Comm. Acc.", "CAR", "CAR Num.", "N_inc"]
    t2_data = [[Paragraph(clean_text(h), tbl_header) for h in t2_headers]]
    
    rows_data_t2 = [
        ["E1", "Base", "Question Only", "0.055", "0.640", "0.0703", "N/A", "N/A", "N/A"],
        ["E2", "QLoRA", "Question Only", "0.520", "1.000", "0.5200", "N/A", "N/A", "N/A"],
        ["E3", "Base", "C1: Good Evidence", "0.235", "0.955", "0.2408", "N/A", "N/A", "N/A"],
        ["E4", "Base", "C2: Incomplete", "0.265", "0.935", "0.2727", "0.1667", "6", "36"],
        ["E5", "Base", "C3: Irrelevant", "0.195", "0.925", "0.2108", "N/A", "N/A", "N/A"],
        ["E6", "Base", "C4: Misleading", "0.150", "0.940", "0.1543", "N/A", "N/A", "N/A"],
        ["E7", "QLoRA", "C1: Good Evidence", "0.420", "1.000", "0.4200", "N/A", "N/A", "N/A"],
        ["E8", "QLoRA", "C2: Incomplete", "0.425", "1.000", "0.4250", "0.0000", "0", "36"],
        ["E9", "QLoRA", "C3: Irrelevant", "0.405", "1.000", "0.4050", "N/A", "N/A", "N/A"],
        ["E10", "QLoRA", "C4: Misleading", "0.360", "1.000", "0.3600", "N/A", "N/A", "N/A"],
        ["E7_ep2", "QLoRA_Ep2", "C1: Good Evidence", "0.350", "0.995", "0.3467", "N/A", "N/A", "N/A"],
        ["E8_ep2", "QLoRA_Ep2", "C2: Incomplete", "0.360", "1.000", "0.3600", "0.0000", "0", "36"],
        ["E9_ep2", "QLoRA_Ep2", "C3: Irrelevant", "0.345", "1.000", "0.3450", "N/A", "N/A", "N/A"],
        ["E10_ep2", "QLoRA_Ep2", "C4: Misleading", "0.285", "0.995", "0.2814", "N/A", "N/A", "N/A"]
    ]
    for r in rows_data_t2:
        row_cells = []
        for j, val in enumerate(r):
            style = tbl_cell_left if j in (1, 2) else tbl_cell
            row_cells.append(Paragraph(clean_text(val), style))
        t2_data.append(row_cells)
        
    t2 = Table(t2_data, colWidths=[45, 60, 110, 50, 50, 60, 45, 45, 35])
    t2.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t2)
    story.append(Paragraph(clean_text("Table 2. Quantitative evaluation matrix across all 14 experimental configurations (N = 200 test queries)."), tbl_note_style))

    story.append(Paragraph(clean_text("4.2 Paired Statistical Hypothesis Testing"), h2_style))
    story.append(Paragraph(clean_text(
        "To determine whether performance differences between the frozen base model and the QLoRA model were statistically "
        "significant, paired McNemar tests with continuity correction were run on binary correctness outcomes across all 200 matched "
        "test queries. Accuracy deltas were bounded using 10,000 paired bootstrap resamples (seed 42)."
    ), body_style))
    story.append(Paragraph(clean_text(
        "Family-wise error rate was controlled at alpha = 0.05 across the four primary evidence conditions using the strict "
        "Holm-Bonferroni step-down correction. Table 3 presents the results."
    ), body_style))

    # Table 3: Statistical Tests
    t3_headers = ["Rank", "Condition", "Delta (QLoRA - Base)", "95% Bootstrap CI", "McNemar p-value", "Holm Threshold", "Significant?"]
    t3_data = [[Paragraph(clean_text(h), tbl_header) for h in t3_headers]]
    rows_data_t3 = [
        ["1", "C4: Misleading Evidence", "+0.2100 (+21.0%)", "[0.125, 0.295]", "p = 0.00001", "0.01250", "True"],
        ["2", "C3: Irrelevant Evidence", "+0.2100 (+21.0%)", "[0.120, 0.295]", "p = 0.00001", "0.01667", "True"],
        ["3", "C1: Good Evidence", "+0.1850 (+18.5%)", "[0.095, 0.270]", "p = 0.00009", "0.02500", "True"],
        ["4", "C2: Incomplete Evidence", "+0.1600 (+16.0%)", "[0.075, 0.245]", "p = 0.00053", "0.05000", "True"]
    ]
    for r in rows_data_t3:
        row_cells = []
        for j, val in enumerate(r):
            style = tbl_cell_left if j == 1 else tbl_cell
            row_cells.append(Paragraph(clean_text(val), style))
        t3_data.append(row_cells)
        
    t3 = Table(t3_data, colWidths=[35, 115, 95, 80, 80, 55, 40])
    t3.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t3)
    story.append(Paragraph(clean_text("Table 3. Paired statistical hypothesis tests and Holm-Bonferroni step-down evaluation (alpha = 0.05, 10,000 bootstrap iterations)."), tbl_note_style))
    story.append(Paragraph(clean_text(
        "All four primary conditions passed the Holm-Bonferroni significance threshold. Under every evidence condition, "
        "QLoRA adaptation achieved a statistically significant improvement in overall decision accuracy over the frozen base reader (p < 0.001 across all tests)."
    ), body_style))
    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------------------------------------
    # 5. Figures
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("5. Visualizations and Empirical Trends"), h1_style))
    story.append(Paragraph(clean_text("The experimental findings are illustrated in five figures generated directly from the evaluation pipeline."), body_style))

    figures_dir = os.path.join(os.getcwd(), "figures")
    
    def add_pdf_figure(filename, caption_text):
        full_path = os.path.join(figures_dir, filename)
        if os.path.isfile(full_path):
            img = Image(full_path, width=5.6 * 72, height=3.15 * 72)
            img.hAlign = 'CENTER'
            cap = Paragraph(clean_text(caption_text), caption_style)
            story.append(KeepTogether([Spacer(1, 4), img, cap]))

    add_pdf_figure("training_loss_curve.png", "Figure 1. QLoRA 4-bit NF4 training loss trajectory across 200 optimizer steps (2 epochs).")
    add_pdf_figure("fig1_accuracy_comparison.png", "Figure 2. Decision accuracy across evidence conditions (C1 to C4) comparing Frozen Base and QLoRA.")
    add_pdf_figure("fig2_abstention_and_misleading.png", "Figure 3. Robustness trade-offs: Panel A shows abstention under incomplete evidence (C2); Panel B shows accuracy under misleading evidence (C4).")
    add_pdf_figure("fig3_performance_retention.png", "Figure 4. Performance retention degradation curves across C1 to C4 for Base, Epoch 1 QLoRA, and Epoch 2 QLoRA.")
    add_pdf_figure("fig4_error_distribution.png", "Figure 5. Output decision distribution under Good Evidence (C1) showing non-abstention behavior in QLoRA.")

    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------------------------------------
    # 6. Comparison with Existing Research
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("6. Comparison with Existing Research"), h1_style))
    story.append(Paragraph(clean_text("Our empirical results provide a valuable comparison against the specialized RAG robustness frameworks reviewed in Phase 1:"), body_style))
    story.append(Paragraph(clean_text(
        "1. InstructRAG (Huang et al., 2025) introduced multi-step self-training to teach models how to filter out noisy context passages. "
        "Our study tests what happens without specialized filtering networks. We found that standard QA-based QLoRA tuning maintains good "
        "accuracy under irrelevant distractors (0.405 under C3 compared to 0.420 under C1, a minor drop of 1.5%), but fails completely to abstain when necessary evidence is absent."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "2. Robust Fine-Tuning (RbFT, Deng et al., 2025) fine-tuned models on corrupted texts to prevent reliance on faulty information. "
        "Our findings demonstrate that when a model is trained exclusively on clean, authentic clinical abstracts, it remains susceptible "
        "to adversarial outcome inversions. Decision accuracy for QLoRA fell from 0.420 under C1 to 0.360 under C4 (a drop of 6.0 percentage points). "
        "Standard fine-tuning does not teach the model to question authoritative-looking contradictions in retrieved text."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "3. RAGuard (Li et al., 2025) deployed an external verification network to block poisoned retrieval inputs. Our results show that "
        "without an external verifier, internal parameter updates alone leave a residual vulnerability. When counterfactual evidence was "
        "introduced in C4, accuracy for the base model dropped from 0.235 to 0.150 (a 36.2% relative decline), while QLoRA dropped from 0.420 to 0.360 "
        "(a 14.3% relative decline). Fine-tuning provides partial buffering against noisy context, but cannot fully immunize the reader."
    ), bullet_style))
    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------------------------------------
    # 7. Ablation Studies
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("7. Ablation Studies"), h1_style))
    story.append(Paragraph(clean_text(
        "Twelve formal ablation comparisons were conducted to isolate the effects of context conditioning, evidence degradation, "
        "and training duration. Table 4 summarizes all twelve comparisons."
    ), body_style))

    # Table 4: 12 Ablations
    t4_headers = ["Ablation Study", "Comparison", "Baseline", "Conditioned", "Delta"]
    t4_data = [[Paragraph(clean_text(h), tbl_header) for h in t4_headers]]
    rows_data_t4 = [
        ["Context Effect (Base)", "E3 (C1) - E1 (Q-Only)", "0.055", "0.235", "+0.1800"],
        ["Context Effect (QLoRA)", "E7 (C1) - E2 (Q-Only)", "0.520", "0.420", "-0.1000"],
        ["Base: Incomplete Drop", "Base C2 - Base C1", "0.235", "0.265", "+0.0300"],
        ["Base: Irrelevant Drop", "Base C3 - Base C1", "0.235", "0.195", "-0.0400"],
        ["Base: Misleading Drop", "Base C4 - Base C1", "0.235", "0.150", "-0.0850"],
        ["QLoRA: Incomplete Drop", "QLoRA C2 - QLoRA C1", "0.420", "0.425", "+0.0050"],
        ["QLoRA: Irrelevant Drop", "QLoRA C3 - QLoRA C1", "0.420", "0.405", "-0.0150"],
        ["QLoRA: Misleading Drop", "QLoRA C4 - QLoRA C1", "0.420", "0.360", "-0.0600"],
        ["Duration Ablation (C1)", "E7_ep2 - E7", "0.420", "0.350", "-0.0700"],
        ["Duration Ablation (C2)", "E8_ep2 - E8", "0.425", "0.360", "-0.0650"],
        ["Duration Ablation (C3)", "E9_ep2 - E9", "0.405", "0.345", "-0.0600"],
        ["Duration Ablation (C4)", "E10_ep2 - E10", "0.360", "0.285", "-0.0750"]
    ]
    for r in rows_data_t4:
        row_cells = []
        for j, val in enumerate(r):
            style = tbl_cell_left if j in (0, 1) else tbl_cell
            row_cells.append(Paragraph(clean_text(val), style))
        t4_data.append(row_cells)
        
    t4 = Table(t4_data, colWidths=[120, 150, 65, 75, 70])
    t4.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t4)
    story.append(Paragraph(clean_text("Table 4. Summary of twelve ablation comparisons isolating context, degradation, and training duration effects."), tbl_note_style))

    story.append(Paragraph(clean_text("7.1 Analysis of Ablation Results"), h2_style))
    story.append(Paragraph(clean_text(
        "1. Context Conditioning Effect: For the frozen base model, adding 3-passage RAG context improved accuracy substantially (+0.180, rising from 0.055 to 0.235). "
        "Without fine-tuning, the base model cannot produce structured PubMedQA decisions from bare questions, so retrieved context serves as a helpful prompt scaffold. "
        "For QLoRA, Question-Only accuracy was higher (0.520) than 3-passage RAG accuracy (0.420, delta -0.100). Because a 1.5B 4-bit model has limited attention capacity, "
        "concatenating three dense MEDLINE abstracts dilutes attention and introduces distractor interference compared to answering concise questions directly from internal weights."
    ), body_style))
    story.append(Paragraph(clean_text(
        "2. Evidence Degradation Sensitivity: When evidence degraded from intact C1 to misleading C4, both models suffered accuracy drops (-0.085 for Base and -0.060 for QLoRA). "
        "Under irrelevant distractors (C3), QLoRA experienced only a slight decline (-0.015), whereas the base model dropped more sharply (-0.040). "
        "This demonstrates that QLoRA fine-tuning provides moderate resilience against off-target noise, but remains vulnerable to direct factual contradiction."
    ), body_style))
    story.append(Paragraph(clean_text(
        "3. Training Duration Ablation (Epoch 1 vs. Epoch 2): Extending QLoRA fine-tuning from 100 steps (Epoch 1) to 200 steps (Epoch 2) resulted in lower accuracy "
        "across every single evidence condition: C1 dropped by 0.070 (0.420 to 0.350), C2 dropped by 0.065 (0.425 to 0.360), C3 dropped by 0.060 (0.405 to 0.345), "
        "and C4 dropped by 0.075 (0.360 to 0.285). This confirms that a 1.5B parameter model begins overfitting training phrasing after a single epoch of fine-tuning."
    ), body_style))
    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------------------------------------
    # 8. Results, Discussion, and Limitations
    # ------------------------------------------------------------------------------------------------
    story.append(Paragraph(clean_text("8. Results, Discussion, and Limitations"), h1_style))

    story.append(Paragraph(clean_text("8.1 Resolution of Research Questions"), h2_style))
    story.append(Paragraph(clean_text(
        "1. Answer to Research Question 1 (RQ1): Yes, ordinary QLoRA fine-tuning significantly improves answer accuracy when retrieved evidence is reliable. "
        "Under Condition C1 (Good Evidence), decision accuracy increased from 0.235 (Base) to 0.420 (QLoRA), an absolute improvement of +18.5 percentage points "
        "(McNemar p = 0.00009). Fine-tuning aligns the model to extract and synthesize clinical findings from retrieved passages."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "2. Answer to Research Question 2 (RQ2): QLoRA does not make the model more vulnerable than the base model in absolute terms, but it fails to eliminate vulnerability to misleading evidence. "
        "When presented with counterfactual conclusions in C4, QLoRA achieved 0.360 accuracy compared to 0.150 for Base (delta +0.210, p = 0.00001). "
        "However, relative to its own performance on good evidence (0.420), QLoRA dropped by 6.0 percentage points. The fine-tuned model still follows "
        "authoritative-sounding false statements because it was never trained to cross-examine or verify conflicting evidence."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "3. Answer to Research Question 3 (RQ3): Mixed results. For irrelevant distractors (C3), QLoRA successfully filters out noise, maintaining 0.405 accuracy "
        "(only 1.5% below C1). However, when evidence is incomplete (C2), QLoRA completely fails to abstain. The base model achieved a Correct Abstention Rate of 16.7% "
        "(6 out of 36 valid incomplete cases), whereas QLoRA achieved 0.0% (0 out of 36 cases). Fine-tuning eliminated the model's willingness to admit uncertainty."
    ), bullet_style))

    story.append(Paragraph(clean_text("8.2 Why QLoRA Fails to Abstain on Incomplete Evidence"), h2_style))
    story.append(Paragraph(clean_text(
        "The failure of QLoRA to abstain under C2 is directly linked to the training data distribution. In PubMedQA, only 11.6% of training records have a consensus "
        "label of 'maybe', while 88.4% have definitive 'yes' or 'no' outcomes. Furthermore, every training example contained complete, intact abstracts with explicit findings. "
        "Because fine-tuning penalized the model for predicting 'maybe' on typical examples, the model learned that definitive answers are almost always expected. "
        "When tested on incomplete abstracts, QLoRA guessed 'yes' or 'no' with 100% coverage rather than expressing caution."
    ), body_style))

    story.append(Paragraph(clean_text("8.3 Project Limitations"), h2_style))
    story.append(Paragraph(clean_text("This research has four principal limitations:"), body_style))
    story.append(Paragraph(clean_text(
        "1. Model Scale and Quantization: The experiments used Qwen2.5-1.5B-Instruct in 4-bit NF4 precision. Larger models (such as 7B or 14B) running in 16-bit precision "
        "may retain stronger parametric memory capable of resisting counterfactual text more effectively."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "2. Valid Incomplete Test Sample Size: Out of 200 held-out test abstracts, only N_inc = 36 had structured conclusion sections that could be verified as cleanly stripped. "
        "While CAR was evaluated strictly on these 36 validated cases, future evaluations would benefit from larger collections of verified incomplete abstracts."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "3. Training Target Composition: Fine-tuning used standard supervised question answering without explicit adversarial examples, conflicting passages, or abstention "
        "training pairs. The observed vulnerabilities reflect standard instruction tuning rather than robustified training variants."
    ), bullet_style))
    story.append(Paragraph(clean_text(
        "4. Qualitative Audit Scope: A stratified 100-sample manual audit sheet was exported with full retrieved contexts, gold labels, and model rationales "
        "(results/manual_audit_sample.csv), leaving expert medical review as an offline qualitative evaluation."
    ), bullet_style))

    # Build the PDF using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"\nSUCCESS: PDF Document saved to {output_path}")

if __name__ == "__main__":
    build_pdf_report()
