#!/usr/bin/env python3
"""
just a simple script to train a cnn on cifar-10 images
nothing fancy, just basic image classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging

# setup logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """basic cnn for classifying images"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # first conv layer - takes rgb images and outputs 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # second conv layer - more feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # pooling to reduce size
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # pass through conv layers with relu and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # flatten for fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # final classification layer
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size=32):
    """grab the cifar-10 dataset and set it up for training"""
    
    # data augmentation for training - makes the model more robust
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # simpler transform for testing - no augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # download cifar-10 if we don't have it
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001):
    """actually train the model - this is where the magic happens"""
    
    # use gpu if available, otherwise cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    # cross entropy loss is good for classification
    criterion = nn.CrossEntropyLoss()
    # adam optimizer usually works well
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # keep track of how we're doing
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # training mode
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # standard training loop
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # track progress
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # print progress every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # calculate how well we did this epoch
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        
        # test the model
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_accuracies.append(test_acc)
        
        logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies

def save_model_and_metrics(model, train_losses, train_accuracies, test_accuracies):
    """save the trained model and some stats about how it did"""
    
    # make sure the models folder exists
    os.makedirs('/app/models', exist_ok=True)
    
    # save the model weights
    model_path = '/app/models/trained_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # save some useful info about training
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_test_accuracy': test_accuracies[-1],
        'training_completed_at': datetime.now().isoformat()
    }
    
    metrics_path = '/app/models/training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")
    
    return model_path, metrics_path

def main():
    """main function - runs everything"""
    
    logger.info("Starting AI model training...")
    
    # get config from environment variables or use defaults
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    EPOCHS = int(os.getenv('EPOCHS', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    
    logger.info(f"Training configuration: Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}")
    
    # load the data
    logger.info("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # create our model
    logger.info("Creating model...")
    model = SimpleCNN(num_classes=10)
    
    # print some info about the model
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # train it!
    logger.info("Starting training...")
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, EPOCHS, LEARNING_RATE
    )
    
    # save everything
    model_path, metrics_path = save_model_and_metrics(model, train_losses, train_accuracies, test_accuracies)
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()

