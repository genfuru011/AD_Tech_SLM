#!/opt/miniconda3/envs/dpo_training/bin/python
"""
Real DPO Training Script using conda environment
Based on the original train_dpo.py but using conda Python
"""

import os
import json
import yaml
import torch
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_device():
    """Setup device for M2 Mac (MPS support)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Using Metal Performance Shaders (MPS) for M2 GPU")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ MPS not available, using CPU")
    return device

def load_jsonl_dataset(file_path: str) -> Dataset:
    """Load JSONL dataset for DPO training."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    df = pd.DataFrame(data)
    logger.info(f"📊 Loaded {len(df)} samples from {file_path}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    return dataset

def setup_model_and_tokenizer(config: dict):
    """Setup model and tokenizer with LoRA configuration."""
    model_name = config["model_name"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config.get("fp16", False) else torch.float32,
        device_map="auto" if torch.backends.mps.is_available() else None,
    )
    
    # Setup LoRA if enabled
    if config.get("use_lora", False):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            target_modules=config.get("target_modules", ["c_attn", "c_proj"]),
        )
        model = get_peft_model(model, lora_config)
        logger.info("✅ LoRA configuration applied")
    
    return model, tokenizer

def main():
    """Main training function."""
    print("🌟 Real DPO Training with conda environment")
    
    # Load configuration
    config_path = "configs/dpo_config.yaml"
    config = load_config(config_path)
    
    # Setup device
    device = setup_device()
    
    # Create output directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["logging_dir"], exist_ok=True)
    
    # Load dataset
    dataset = load_jsonl_dataset(config["dataset_path"])
    
    # Split dataset
    train_size = int(len(dataset) * config["train_split"])
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    logger.info(f"📊 Training samples: {len(train_dataset)}")
    logger.info(f"📊 Validation samples: {len(eval_dataset)}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    logger.info(f"📊 Model parameters: {model.num_parameters():,}")
    
    # Training arguments
    training_args = DPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy=config["eval_strategy"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        fp16=config.get("fp16", False),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        remove_unused_columns=config.get("remove_unused_columns", True),
        logging_dir=config["logging_dir"],
        report_to=config.get("report_to", None),  # Disable wandb/tensorboard for now
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        dataloader_pin_memory=config.get("dataloader_pin_memory", False),
        beta=float(config.get("beta", 0.1)),
        max_length=config.get("max_length", 512),
        max_prompt_length=config.get("max_prompt_length", 256),
    )
    
    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Start training
    logger.info("🚀 Starting DPO training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    logger.info(f"✅ Training completed! Model saved to {config['output_dir']}")

if __name__ == "__main__":
    main()
