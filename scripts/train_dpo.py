#!/opt/miniconda3/envs/dpo_training/bin/python
"""
DPO Training Script for AD_Tech_SLM
Optimized for MacBook Air M2 8GB with MPS support
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
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer

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

def preprocess_dataset(dataset: Dataset, tokenizer, max_length: int, max_prompt_length: int):
    """Preprocess dataset for DPO training."""
    
    def format_prompt(example):
        """Format the prompt for training."""
        return example["prompt"]
    
    def tokenize_function(examples):
        """Tokenize the examples."""
        # Format prompts
        prompts = [format_prompt(example) for example in examples]
        
        # Tokenize
        model_inputs = tokenizer(
            prompts,
            max_length=max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize chosen and rejected responses
        chosen_inputs = tokenizer(
            examples["chosen"],
            max_length=max_length - max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_inputs = tokenizer(
            examples["rejected"],
            max_length=max_length - max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Combine prompt with responses
        model_inputs["chosen_input_ids"] = torch.cat([
            model_inputs["input_ids"],
            chosen_inputs["input_ids"]
        ], dim=1)
        
        model_inputs["rejected_input_ids"] = torch.cat([
            model_inputs["input_ids"],
            rejected_inputs["input_ids"]
        ], dim=1)
        
        return model_inputs
    
    # Apply preprocessing
    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return processed_dataset

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
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, lora_config)
        logger.info("✅ LoRA configuration applied")
    
    return model, tokenizer

def main():
    """Main training function."""
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
    
    # Preprocess datasets
    train_dataset = preprocess_dataset(
        train_dataset, tokenizer, config["max_length"], config["max_prompt_length"]
    )
    eval_dataset = preprocess_dataset(
        eval_dataset, tokenizer, config["max_length"], config["max_prompt_length"]
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy=config["eval_strategy"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        fp16=config.get("fp16", False),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        remove_unused_columns=config.get("remove_unused_columns", True),
        logging_dir=config["logging_dir"],
        report_to=config.get("report_to", "tensorboard"),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        dataloader_pin_memory=config.get("dataloader_pin_memory", False),
    )
    
    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=config.get("beta", 0.1),
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