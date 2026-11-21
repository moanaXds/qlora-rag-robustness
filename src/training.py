"""
QLoRA Training Module for Qwen2.5-1.5B-Instruct
Robustness of QLoRA-Adapted RAG Systems with Bad Retrieved Evidence
Researcher: Muanna Hamid | FAST-NUCES Islamabad | November 2026

Adheres strictly to Phase 2 Methodology:
- 4-bit NF4 with double quantization (torch.float16 compute).
- LoRA targets: q_proj, k_proj, v_proj, o_proj (r=16, alpha=32, dropout=0.05).
- Trains exclusively on the 800 training records.
- Optimizer: paged_adamw_8bit, LR=2e-4, linear decay, 5% warmup.
- Batch size: 1, Gradient accumulation: 8 (effective batch size 8).
- 1 epoch (100 optimizer steps), max_seq_length: 512 tokens.
- Native ChatML formatting with quarantined long_answer reference supervision.
"""

import os
import json
from typing import List, Dict, Any
from src.config import ExperimentConfig


def format_chatml_training_example(tokenizer, record: Dict[str, Any], system_prompt: str) -> str:
    """
    Formats a training record into ChatML format.
    Context is the intact training abstract.
    Supervised target is the reference answer followed by Final Decision.
    """
    contexts = record["context"].get("contexts", [])
    context_text = "\n\n".join(contexts).strip()
    question = record["question"]
    long_answer = record.get("long_answer", "").strip()
    decision = record["final_decision"].strip().lower()
    
    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )
    
    assistant_content = f"{long_answer}\n\nFinal Decision: {decision}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # ChatML text fallback
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
        )


def train_qlora_adapter(config: ExperimentConfig = None, train_records_path: str = None):
    """
    Executes 4-bit QLoRA fine-tuning on Qwen2.5-1.5B-Instruct.
    """
    cfg = config or ExperimentConfig()
    train_path = train_records_path or cfg.train_data_path
    
    print("=" * 70)
    print("Phase 3 QLoRA Fine-Tuning: Qwen/Qwen2.5-1.5B-Instruct")
    print("=" * 70)
    
    # Check GPU availability
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
    except ImportError:
        cuda_avail = False

    if not cuda_avail:
        print("[WARNING] No CUDA GPU / PyTorch detected in current environment.")
        print("[NOTICE] Heavy model training must run on Google Colab GPU (e.g. NVIDIA T4/A100).")
        print("[NOTICE] Creating training manifest and returning configuration.")
        return None

    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig, 
        TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    # 1. 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
        bnb_4bit_compute_dtype=torch.float16 if cfg.compute_dtype == "float16" else torch.bfloat16
    )

    print(f"[Model] Loading base model '{cfg.model_name_or_path}' with 4-bit NF4 quantization...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # 2. LoRA Adapter Configuration
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"[LoRA] Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_params / all_params * 100:.2f}%)")

    # 3. Load & Format Dataset
    print(f"[Data] Loading training records from {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]

    formatted_texts = [
        format_chatml_training_example(tokenizer, rec, cfg.system_prompt)
        for rec in train_records
    ]
    train_dataset = Dataset.from_dict({"text": formatted_texts})

    # 4. Training Arguments with introspected parameter resolution
    import inspect
    import dataclasses
    
    total_steps = int((len(train_records) / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)) * cfg.num_train_epochs)
    equiv_warmup_steps = max(1, int(round(total_steps * cfg.warmup_ratio)))

    os.makedirs(cfg.adapter_dir, exist_ok=True)
    
    config_cls = TrainingArguments
    has_sft_config = False
    try:
        from trl import SFTConfig
        config_cls = SFTConfig
        has_sft_config = True
    except (ImportError, AttributeError):
        pass

    supported_params = set(inspect.signature(config_cls).parameters.keys())
    if dataclasses.is_dataclass(config_cls):
        supported_params.update(f.name for f in dataclasses.fields(config_cls))

    ta_kwargs = {
        "output_dir": cfg.adapter_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": total_steps,
        "logging_steps": cfg.logging_steps,
        "fp16": True,
        "save_strategy": cfg.save_strategy,
        "report_to": "none"
    }
    if "optim" in supported_params:
        ta_kwargs["optim"] = cfg.optim

    if "warmup_ratio" in supported_params:
        ta_kwargs["warmup_ratio"] = cfg.warmup_ratio
    elif "warmup_steps" in supported_params:
        ta_kwargs["warmup_steps"] = equiv_warmup_steps

    if "activation_checkpointing" in supported_params:
        ta_kwargs["activation_checkpointing"] = cfg.gradient_checkpointing
    elif "gradient_checkpointing" in supported_params:
        ta_kwargs["gradient_checkpointing"] = cfg.gradient_checkpointing

    if has_sft_config:
        if "max_seq_length" in supported_params:
            ta_kwargs["max_seq_length"] = cfg.max_seq_length
        elif "max_length" in supported_params:
            ta_kwargs["max_length"] = cfg.max_seq_length
        if "dataset_text_field" in supported_params:
            ta_kwargs["dataset_text_field"] = "text"

    final_ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported_params}
    training_args = config_cls(**final_ta_kwargs)

    # 5. SFT Trainer
    trainer_sig = inspect.signature(SFTTrainer)
    trainer_params = set(trainer_sig.parameters.keys())
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": training_args
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_params and "dataset_text_field" not in final_ta_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params and "max_seq_length" not in final_ta_kwargs:
        trainer_kwargs["max_seq_length"] = cfg.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)

    print("[Training] Starting 1-epoch QLoRA fine-tuning run...")
    train_result = trainer.train()
    print(f"[Training] Completed! Final train loss: {train_result.training_loss:.4f}")

    # 6. Save Adapter
    print(f"[Model] Saving adapter weights to {cfg.adapter_dir}...")
    trainer.model.save_pretrained(cfg.adapter_dir)
    tokenizer.save_pretrained(cfg.adapter_dir)
    
    meta = {
        "model_name": cfg.model_name_or_path,
        "trainable_parameters": trainable_params,
        "total_parameters": all_params,
        "percentage": trainable_params / all_params * 100,
        "epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "training_loss": train_result.training_loss
    }
    with open(os.path.join(cfg.adapter_dir, "adapter_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return trainer.model


if __name__ == "__main__":
    train_qlora_adapter()
