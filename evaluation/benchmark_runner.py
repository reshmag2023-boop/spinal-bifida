"""
Unified Multi-Model Benchmark Runner.
Trains and evaluates Custom CNN, ResNet-18, EfficientNet-B0, and ViT
on the Spina Bifida dataset, recording real computed metrics in model_artifacts/benchmarks.json.
"""

import os
import glob
import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List

from config.settings import (
    NORMAL_DIR, SB_DIR, ARTIFACTS_DIR, DEVICE,
    SUPPORTED_MODELS, IMG_SIZE
)
from preprocessing.transforms import (
    UltrasoundDataset, get_train_transforms, get_val_transforms
)
from models.custom_cnn import SpinaBifidaCNN
from models.resnet_model import SpinaBifidaResNet
from models.efficientnet_model import SpinaBifidaEfficientNet
from models.vit_model import SpinaBifidaViT
from models.model_trainer import train_model
from models.model_evaluator import evaluate_model

def get_dataset_splits(val_ratio: float = 0.20, seed: int = 42):
    """Gathers all normal and Spina Bifida image paths with stratified train/val split."""
    normal_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
        normal_files.extend(glob.glob(str(NORMAL_DIR / ext)))
        
    sb_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
        sb_files.extend(glob.glob(str(SB_DIR / ext)))
        
    all_files = normal_files + sb_files
    all_labels = [0] * len(normal_files) + [1] * len(sb_files)
    
    print(f"Total dataset found: {len(normal_files)} Normal, {len(sb_files)} Spina Bifida.")
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=val_ratio, random_state=seed, stratify=all_labels
    )
    
    return train_files, val_files, train_labels, val_labels

def run_all_benchmarks(
    epochs: int = 6, 
    batch_size: int = 32, 
    models_to_run: List[str] = None
) -> Dict[str, Any]:
    """
    Executes training and evaluation across specified models, saving real results.
    """
    if models_to_run is None:
        models_to_run = ["resnet18", "custom_cnn", "efficientnet_b0", "vit_b_16"]
        
    train_files, val_files, train_labels, val_labels = get_dataset_splits()
    
    train_dataset = UltrasoundDataset(train_files, train_labels, transform=get_train_transforms())
    val_dataset = UltrasoundDataset(val_files, val_labels, transform=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Preserve existing benchmarks if present
    out_file = ARTIFACTS_DIR / "benchmarks.json"
    benchmark_results = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                benchmark_results = json.load(f)
        except Exception:
            benchmark_results = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*50}\nBENCHMARKING MODEL: {model_name}\n{'='*50}")
        if model_name == "resnet18":
            model = SpinaBifidaResNet(backbone="resnet18", pretrained=True)
            lr = 1e-4
        elif model_name == "custom_cnn":
            model = SpinaBifidaCNN()
            lr = 3e-4
        elif model_name == "efficientnet_b0":
            model = SpinaBifidaEfficientNet(pretrained=True)
            lr = 1e-4
        elif model_name in ["vit_transformer", "vit_b_16"]:
            model = SpinaBifidaViT(pretrained=True, freeze_backbone=True, unfreeze_last_n_blocks=2)
            lr = 2e-4
        else:
            continue
            
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            model_name=model_name
        )
        
        eval_metrics = evaluate_model(trained_model, val_loader)
        eval_metrics["history"] = history
        eval_metrics["model_architecture"] = model_name
        eval_metrics["epochs_trained"] = epochs
        eval_metrics["val_samples"] = len(val_files)
        
        # Strip long arrays from summary view for clean json
        y_true_sample = eval_metrics.pop("y_true", [])[:200]
        y_prob_sample = eval_metrics.pop("y_prob", [])[:200]
        eval_metrics["sample_predictions"] = {
            "y_true": y_true_sample,
            "y_prob": y_prob_sample
        }
        
        benchmark_results[model_name] = eval_metrics
        print(f"Results for {model_name}: Accuracy={eval_metrics['accuracy']*100:.2f}%, Sensitivity={eval_metrics['sensitivity']*100:.2f}%, Specificity={eval_metrics['specificity']*100:.2f}%, AUC={eval_metrics['roc_auc']:.4f}, Latency={eval_metrics['inference_time_ms']}ms")
        
    # Save to disk
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
        
    print(f"\n[OK] All benchmarks finished and saved to {out_file}")
    return benchmark_results

if __name__ == "__main__":
    run_all_benchmarks(epochs=4)
