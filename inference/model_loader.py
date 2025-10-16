#!/usr/bin/env python3
"""
loads the trained model and makes predictions
basically the same cnn as in training but for inference
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """same cnn architecture as in training"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ModelLoader:
    """handles loading the model and making predictions"""
    
    def __init__(self, model_path: str = "/app/models/trained_model.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cifar-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
    def load_model(self) -> bool:
        """load the trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                return False
            
            # create the model
            self.model = SimpleCNN(num_classes=10)
            
            # load the weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def predict(self, image_tensor: torch.Tensor) -> dict:
        """make a prediction on an image"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # get top 3 predictions for fun
                top3_prob, top3_indices = torch.topk(probabilities, 3, dim=1)
                
                result = {
                    'predicted_class': self.class_names[predicted.item()],
                    'confidence': confidence.item(),
                    'top3_predictions': [
                        {
                            'class': self.class_names[idx.item()],
                            'probability': prob.item()
                        }
                        for prob, idx in zip(top3_prob[0], top3_indices[0])
                    ]
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def is_loaded(self) -> bool:
        """check if the model is loaded"""
        return self.model is not None

# global instance we can use everywhere
model_loader = ModelLoader()
