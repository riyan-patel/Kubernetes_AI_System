#!/usr/bin/env python3
"""
Train Flan-T5 model on MultiNLI dataset for natural language inference
Fine-tunes pre-trained Flan-T5 for entailment/contradiction/neutral classification
"""

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import json
from datetime import datetime
import logging
from typing import Dict, List

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MultiNLI label mapping
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class MultiNLIDataset(Dataset):
    """Dataset wrapper for MultiNLI"""
    
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Create input text: "premise: {premise} hypothesis: {hypothesis}"
        input_text = f"premise: {item['premise']} hypothesis: {item['hypothesis']}"
        
        # Get label - handle both string and integer labels
        label = item['label']
        if isinstance(label, int):
            # Convert integer label to text
            label_text = REVERSE_LABEL_MAP.get(label, "neutral")
        elif isinstance(label, str):
            # Already a string, use it directly
            label_text = label
        else:
            # Fallback
            label_text = "neutral"
        
        # Tokenize input (no padding - data collator will handle it)
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None  # Return as list, not tensor
        )
        
        # Tokenize label text for text-to-text generation (no padding)
        labels = self.tokenizer(
            label_text,
            max_length=10,
            padding=False,
            truncation=True,
            return_tensors=None  # Return as list, not tensor
        )
        
        # Get label ID for evaluation
        if isinstance(label, int):
            label_id = label
        else:
            label_id = LABEL_MAP.get(label, -1)
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids'],
            'label_id': label_id
        }

def get_data_loaders(tokenizer, batch_size=8, max_length=512):
    """Load MultiNLI dataset and create data loaders"""
    
    logger.info("Loading MultiNLI dataset from Hugging Face...")
    
    # Load MultiNLI dataset (automatically downloads if not cached)
    dataset = load_dataset("multi_nli")
    
    # Use matched validation set
    train_dataset = MultiNLIDataset(dataset['train'], tokenizer, max_length)
    val_dataset = MultiNLIDataset(dataset['validation_matched'], tokenizer, max_length)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute accuracy for evaluation"""
    predictions, labels = eval_pred
    
    # For text-to-text, we need to decode predictions
    # For simplicity, we'll use the label_id for accuracy
    # In practice, you'd decode the text predictions
    return {"accuracy": 0.0}  # Placeholder - would need proper decoding

def train_model(model, tokenizer, train_dataset, val_dataset, epochs=3, learning_rate=5e-5, batch_size=8):
    """Train the Flan-T5 model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get gradient accumulation steps from environment or use default
    gradient_accumulation_steps = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', 4))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='/app/models/checkpoints',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),  # Smaller eval batch
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir='/app/models/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_pin_memory=False,  # Disable pin_memory to save RAM
    )
    
    # Data collator for seq2seq (text-to-text) tasks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")
    
    # Evaluate
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")
    
    return trainer, eval_results

def save_model_and_metrics(model, tokenizer, eval_results, output_dir='/app/models'):
    """Save the trained model and metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model_save_path = os.path.join(output_dir, 'flan_t5_multinli')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Save metrics
    metrics = {
        'eval_loss': eval_results.get('eval_loss', 0),
        'eval_runtime': eval_results.get('eval_runtime', 0),
        'eval_samples_per_second': eval_results.get('eval_samples_per_second', 0),
        'training_completed_at': datetime.now().isoformat(),
        'model_type': 'flan-t5',
        'dataset': 'multi_nli',
        'task': 'natural_language_inference'
    }
    
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")
    
    return model_save_path, metrics_path

def main():
    """Main training function"""
    
    logger.info("Starting Flan-T5 MultiNLI training...")
    
    # Get config from environment variables or use defaults
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))  # Smaller batch for transformers
    EPOCHS = int(os.getenv('EPOCHS', 3))  # Fewer epochs for fine-tuning
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 5e-5))
    MODEL_NAME = os.getenv('MODEL_NAME', 'google/flan-t5-base')
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
    
    logger.info(f"Training configuration:")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Learning Rate: {LEARNING_RATE}")
    logger.info(f"  Max Length: {MAX_LENGTH}")
    
    # Load tokenizer and model
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    # Load data
    logger.info("Loading MultiNLI dataset...")
    train_dataset, val_dataset = get_data_loaders(tokenizer, BATCH_SIZE, MAX_LENGTH)
    
    # Train
    trainer, eval_results = train_model(
        model, tokenizer, train_dataset, val_dataset, 
        EPOCHS, LEARNING_RATE, BATCH_SIZE
    )
    
    # Save
    model_path, metrics_path = save_model_and_metrics(model, tokenizer, eval_results)
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
