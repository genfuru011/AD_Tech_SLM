#!/opt/miniconda3/envs/dpo_training/bin/python
"""
Simple data validation test without external dependencies
"""

import json
import sys

def test_dataset_format(file_path):
    """Test if the dataset has correct format."""
    required_fields = {"prompt", "chosen", "rejected"}
    valid_count = 0
    total_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_count += 1
                try:
                    data = json.loads(line.strip())
                    if all(field in data for field in required_fields):
                        valid_count += 1
                        print(f"✅ Line {line_num}: Valid")
                    else:
                        print(f"❌ Line {line_num}: Missing fields")
                except json.JSONDecodeError:
                    print(f"❌ Line {line_num}: Invalid JSON")
        
        print(f"\n📊 Results: {valid_count}/{total_count} valid lines")
        return valid_count > 0
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_data.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    print(f"🔍 Testing dataset: {dataset_path}")
    
    if test_dataset_format(dataset_path):
        print("✅ Dataset format is valid!")
    else:
        print("❌ Dataset format validation failed!")
        sys.exit(1)