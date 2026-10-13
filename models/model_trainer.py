"""
Unified PyTorch Model Training Engine with Early Stopping and Checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from config.settings import DEVICE, ARTIFACTS_DIR

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-4,
    model_name: str = "resnet18",
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains a model with AdamW optimizer, CrossEntropyLoss, and saves the best model checkpoint.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    if save_path is None:
        save_path = str(ARTIFACTS_DIR / f"spina_bifida_{model_name}.pt")
        
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epoch_times": []
    }
    
    print(f"\n--- Training {model_name} on {DEVICE} for {epochs} Epochs ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # Train pass
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        scheduler.step()
        
        train_loss = running_loss / (total + 1e-6)
        train_acc = correct / (total + 1e-6)
        
        # Validation pass
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / (val_total + 1e-6)
        val_acc = val_correct / (val_total + 1e-6)
        epoch_dur = time.time() - epoch_start
        
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["epoch_times"].append(round(epoch_dur, 2))
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f} ({epoch_dur:.1f}s)")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
    total_time = time.time() - start_time
    print(f"✅ Training completed in {total_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%. Saved to {save_path}")
    
    # Reload best weights
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, history
