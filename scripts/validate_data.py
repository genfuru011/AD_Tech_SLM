#!/opt/miniconda3/envs/dpo_training/bin/python
"""
Data validation and preprocessing utilities for DPO training
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_dpo_dataset(file_path: str) -> bool:
    """
    Validate DPO dataset format.
    Each line should have: prompt, chosen, rejected
    """
    required_fields = {"prompt", "chosen", "rejected"}
    valid_lines = 0
    total_lines = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    
                    # Check required fields
                    if not isinstance(data, dict):
                        logger.warning(f"Line {line_num}: Not a JSON object")
                        continue
                    
                    missing_fields = required_fields - set(data.keys())
                    if missing_fields:
                        logger.warning(f"Line {line_num}: Missing fields: {missing_fields}")
                        continue
                    
                    # Check field types
                    for field in required_fields:
                        if not isinstance(data[field], str):
                            logger.warning(f"Line {line_num}: Field '{field}' is not a string")
                            continue
                    
                    # Check if fields are not empty
                    for field in required_fields:
                        if not data[field].strip():
                            logger.warning(f"Line {line_num}: Field '{field}' is empty")
                            continue
                    
                    valid_lines += 1
                    
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_num}: Invalid JSON format")
                    continue
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return False
    
    logger.info(f"✅ Validation complete: {valid_lines}/{total_lines} valid lines")
    return valid_lines > 0

def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """
    Analyze DPO dataset and provide statistics.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not data:
        logger.error("No valid data found")
        return {}
    
    df = pd.DataFrame(data)
    
    # Calculate statistics
    stats = {
        "total_samples": len(df),
        "avg_prompt_length": df["prompt"].str.len().mean(),
        "avg_chosen_length": df["chosen"].str.len().mean(),
        "avg_rejected_length": df["rejected"].str.len().mean(),
        "max_prompt_length": df["prompt"].str.len().max(),
        "max_chosen_length": df["chosen"].str.len().max(),
        "max_rejected_length": df["rejected"].str.len().max(),
    }
    
    return stats

def print_dataset_stats(file_path: str):
    """Print dataset statistics."""
    logger.info(f"📊 Analyzing dataset: {file_path}")
    
    if not validate_dpo_dataset(file_path):
        logger.error("❌ Dataset validation failed")
        return
    
    stats = analyze_dataset(file_path)
    
    if not stats:
        logger.error("❌ Could not analyze dataset")
        return
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average prompt length: {stats['avg_prompt_length']:.1f} characters")
    print(f"Average chosen length: {stats['avg_chosen_length']:.1f} characters")
    print(f"Average rejected length: {stats['avg_rejected_length']:.1f} characters")
    print(f"Max prompt length: {stats['max_prompt_length']} characters")
    print(f"Max chosen length: {stats['max_chosen_length']} characters")
    print(f"Max rejected length: {stats['max_rejected_length']} characters")
    print("="*50)

def create_sample_dataset(output_path: str, num_samples: int = 10):
    """Create a sample DPO dataset for testing."""
    
    sample_data = [
        {
            "prompt": "【テーマ】雨の日でもワクワクするニュースアプリを紹介してください",
            "chosen": "雨が降っても最新トレンドをスマホでサクッとチェック！天気と話題を同時にキャッチして、移動中も退屈知らず♪",
            "rejected": "ニュースが見られるよ！便利！"
        },
        {
            "prompt": "【テーマ】忙しい会社員が移動中に英語を学べるアプリを紹介してください",
            "chosen": "通勤電車で 3 分！AI レッスンがあなたの発音を即フィードバック💡スキマ時間で着実にスキルアップ！",
            "rejected": "英語学習ができます。おすすめです。"
        }
    ] * (num_samples // 2)
    
    # Ensure we have exactly num_samples
    sample_data = sample_data[:num_samples]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Created sample dataset with {num_samples} samples: {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <dataset_path>")
        print("       python validate_data.py --create-sample <output_path> [num_samples]")
        sys.exit(1)
    
    if sys.argv[1] == "--create-sample":
        if len(sys.argv) < 3:
            print("Usage: python validate_data.py --create-sample <output_path> [num_samples]")
            sys.exit(1)
        
        output_path = sys.argv[2]
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        create_sample_dataset(output_path, num_samples)
    else:
        dataset_path = sys.argv[1]
        print_dataset_stats(dataset_path)