#!/usr/bin/env python3
"""
Loads the trained Flan-T5 model and makes predictions for natural language inference
Handles premise + hypothesis pairs and returns entailment/contradiction/neutral
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# MultiNLI labels
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class ModelLoader:
    """Handles loading Flan-T5 model and making NLI predictions"""
    
    def __init__(self, model_path: str = "/app/models/flan_t5_multinli"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_names = ["entailment", "neutral", "contradiction"]
        
    def load_model(self) -> bool:
        """Load the trained Flan-T5 model from disk"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model directory not found at {self.model_path}")
                return False
            
            # Check if model files exist
            if not os.path.exists(os.path.join(self.model_path, "config.json")):
                logger.error(f"Model files not found in {self.model_path}")
                return False
            
            logger.info(f"Loading model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, premise: str, hypothesis: str) -> Dict:
        """
        Make a prediction on a premise-hypothesis pair
        
        Args:
            premise: The premise sentence
            hypothesis: The hypothesis sentence
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Create input text in the same format as training
            input_text = f"premise: {premise} hypothesis: {hypothesis}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=10,
                    num_beams=3,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode prediction
            predicted_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()
            
            # Map to label (handle variations)
            predicted_label = predicted_text
            if predicted_text not in self.label_names:
                # Try to find closest match
                if "entail" in predicted_text:
                    predicted_label = "entailment"
                elif "contradict" in predicted_text or "false" in predicted_text:
                    predicted_label = "contradiction"
                elif "neutral" in predicted_text or "maybe" in predicted_text:
                    predicted_label = "neutral"
                else:
                    # Default to neutral if unclear
                    predicted_label = "neutral"
                    logger.warning(f"Unclear prediction: '{predicted_text}', defaulting to neutral")
            
            # Get probabilities for all labels (simplified - in practice would compute properly)
            # For now, we'll use a simple confidence based on the prediction
            confidence = 0.85  # Placeholder - would compute from model outputs
            
            result = {
                'label': predicted_label,
                'confidence': confidence,
                'premise': premise,
                'hypothesis': hypothesis,
                'raw_prediction': predicted_text
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None and self.tokenizer is not None

# Global instance
model_loader = ModelLoader()
