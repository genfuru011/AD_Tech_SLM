#!/opt/miniconda3/envs/dpo_training/bin/python
"""
Minimal DPO Training Implementation
Without heavy dependencies that require _lzma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
import yaml

class SimpleDPOTrainer:
    """Simple DPO trainer without transformers library dependency."""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        
    def _setup_device(self):
        """Setup device for training."""
        if torch.backends.mps.is_available():
            print("🚀 Using Metal Performance Shaders (MPS)")
            return torch.device("mps")
        else:
            print("⚠️ Using CPU")
            return torch.device("cpu")
    
    def load_dataset(self, dataset_path):
        """Load DPO dataset from JSONL file."""
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"📊 Loaded {len(data)} samples")
        return data
    
    def prepare_data(self, data):
        """Prepare data for training."""
        # Simple tokenization (character-level for demo)
        def simple_tokenize(text, max_length=50):
            # Convert to character indices
            chars = list(text)[:max_length]
            # Pad with spaces
            while len(chars) < max_length:
                chars.append(' ')
            return [ord(c) % 256 for c in chars]  # Simple char encoding
        
        processed_data = []
        for item in data:
            prompt_tokens = simple_tokenize(item['prompt'])
            chosen_tokens = simple_tokenize(item['chosen'])
            rejected_tokens = simple_tokenize(item['rejected'])
            
            processed_data.append({
                'prompt': torch.tensor(prompt_tokens, dtype=torch.long),
                'chosen': torch.tensor(chosen_tokens, dtype=torch.long),
                'rejected': torch.tensor(rejected_tokens, dtype=torch.long)
            })
        
        return processed_data
    
    def create_simple_model(self, vocab_size=256, hidden_size=128):
        """Create a simple transformer-like model."""
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model=hidden_size,
                        nhead=4,
                        batch_first=True
                    ) for _ in range(num_layers)
                ])
                self.output = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, x):
                # Simple forward pass
                x = self.embedding(x)
                for layer in self.layers:
                    x = layer(x, x)  # Self-attention
                return self.output(x)
        
        return SimpleTransformer(vocab_size, hidden_size)
    
    def dpo_loss(self, model, batch):
        """Compute DPO loss."""
        prompts = batch['prompt'].to(self.device)
        chosen = batch['chosen'].to(self.device)
        rejected = batch['rejected'].to(self.device)
        
        # Get model outputs
        chosen_logits = model(chosen)
        rejected_logits = model(rejected)
        
        # Simple DPO-style loss (simplified)
        chosen_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_probs = F.log_softmax(rejected_logits, dim=-1)
        
        # Preference loss
        chosen_score = chosen_probs.mean()
        rejected_score = rejected_probs.mean()
        
        beta = float(self.config.get('beta', 0.1))
        loss = -F.logsigmoid(beta * (chosen_score - rejected_score))
        
        return loss
    
    def train_step(self, model, optimizer, batch):
        """Single training step."""
        optimizer.zero_grad()
        loss = self.dpo_loss(model, batch)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def train(self, dataset_path):
        """Main training loop."""
        print("🚀 Starting Simple DPO Training...")
        
        # Load and prepare data
        data = self.load_dataset(dataset_path)
        processed_data = self.prepare_data(data[:100])  # Use first 100 samples for demo
        
        # Create model
        model = self.create_simple_model().to(self.device)
        learning_rate = float(self.config.get('learning_rate', 1e-4))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        num_epochs = int(self.config.get('num_train_epochs', 3))
        batch_size = int(self.config.get('per_device_train_batch_size', 1))
        beta = float(self.config.get('beta', 0.1))
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Simple batching
            for i in range(0, len(processed_data), batch_size):
                batch_data = processed_data[i:i+batch_size]
                
                # Stack batch (simplified)
                if batch_data:
                    batch = {
                        'prompt': torch.stack([item['prompt'] for item in batch_data]),
                        'chosen': torch.stack([item['chosen'] for item in batch_data]),
                        'rejected': torch.stack([item['rejected'] for item in batch_data])
                    }
                    
                    loss = self.train_step(model, optimizer, batch)
                    epoch_losses.append(loss)
                    
                    if i % 10 == 0:
                        print(f"   Step {i//batch_size}: Loss = {loss:.4f}")
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            print(f"📊 Epoch {epoch+1}/{num_epochs}: Average Loss = {avg_loss:.4f}")
        
        # Save model
        output_dir = Path(self.config.get('output_dir', './outputs'))
        output_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), output_dir / 'simple_dpo_model.pt')
        print(f"✅ Model saved to {output_dir / 'simple_dpo_model.pt'}")
        
        return model

def main():
    """Main function."""
    print("🌟 Simple DPO Training Demo")
    print("   Using minimal dependencies (no transformers library)")
    
    # Load config
    config_path = "configs/dpo_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = SimpleDPOTrainer(config)
    
    # Train
    dataset_path = config['dataset_path']
    model = trainer.train(dataset_path)
    
    print("\n🎉 Simple DPO training completed!")
    print("   This is a demo implementation to test the environment.")
    print("   For production use, consider fixing the transformers library setup.")

if __name__ == "__main__":
    main()
