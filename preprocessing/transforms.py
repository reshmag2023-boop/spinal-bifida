"""
Image transformation and augmentation pipelines for Fetal Ultrasound data.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
import os

from config.settings import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

def get_train_transforms(img_size: Tuple[int, int] = IMG_SIZE) -> transforms.Compose:
    """Returns PyTorch data augmentation pipeline for training."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_transforms(img_size: Tuple[int, int] = IMG_SIZE) -> transforms.Compose:
    """Returns deterministic PyTorch preprocessing pipeline for validation/testing."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def apply_clahe(image_np: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance ultrasound acoustic contrast.
    """
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    elif len(image_np.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image_np)
    return image_np

def load_and_preprocess_image(
    image_path: str, 
    img_size: Tuple[int, int] = IMG_SIZE, 
    use_clahe: bool = True
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Loads an ultrasound image from disk and returns:
    1. PyTorch Tensor formatted for model inference (1, 3, 224, 224)
    2. Raw RGB numpy image resized (224, 224, 3)
    3. CLAHE enhanced RGB numpy image (224, 224, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not decode image at {image_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    
    img_clahe = apply_clahe(img_resized) if use_clahe else img_resized.copy()
    
    # Transform to tensor
    transform = get_val_transforms(img_size)
    pil_img = Image.fromarray(img_clahe)
    tensor = transform(pil_img).unsqueeze(0)
    
    return tensor, img_resized, img_clahe

class UltrasoundDataset(Dataset):
    """
    PyTorch Dataset for Normal and Spina Bifida Ultrasound Images.
    Supports on-the-fly augmentation, CLAHE enhancement, and label assignment.
    """
    def __init__(
        self, 
        file_paths: List[str], 
        labels: List[int], 
        transform: Optional[transforms.Compose] = None,
        use_clahe: bool = True
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform or get_val_transforms()
        self.use_clahe = use_clahe

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Create a blank fallback image to prevent crashing
            img_bgr = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        
        if self.use_clahe:
            img_processed = apply_clahe(img_resized)
        else:
            img_processed = img_resized
            
        pil_img = Image.fromarray(img_processed)
        tensor = self.transform(pil_img)
        
        return tensor, label, path
