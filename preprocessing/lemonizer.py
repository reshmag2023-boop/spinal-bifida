"""
Enhanced Morphological Lemon Sign Generator for Fetal Ultrasound Scans.
Preserves the physics-based frontal cranial bone inward pinch deformation,
adds speckle noise simulation, and supports custom perturbation strengths.
"""

import cv2
import numpy as np
import os
import glob
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple
from pathlib import Path

class LemonSignGenerator:
    """
    Generates synthetic Cranial Lemon Sign deformations on fetal ultrasound scans.
    Simulates the frontal bone flattening and inward scalloping seen in Chiari II / Spina Bifida.
    """
    def __init__(self, random_seed: Optional[int] = 42):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    @staticmethod
    def add_ultrasound_noise(image: np.ndarray, noise_factor: float = 0.4) -> np.ndarray:
        """Adds random acoustic speckle noise to mimic ultrasound attenuation."""
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * noise_factor
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_lemon_deformation(
        self, 
        image: np.ndarray, 
        pinch_strength: Optional[float] = None,
        add_noise: bool = True,
        add_blur: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Applies morphological inward cranial pinch to top 50% of the image.
        Returns: (deformed_image, actual_pinch_strength_used)
        """
        rows, cols = image.shape[:2]
        
        if pinch_strength is None:
            # Random strength between 0.0015 (moderate) and 0.0035 (severe)
            pinch_strength = random.uniform(0.0015, 0.0035)
            
        center_x, center_y = cols / 2.0, rows / 4.0
        
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        dx, dy = x_coords - center_x, y_coords - center_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Apply pinch specifically to frontal cranial zone (upper 55%)
        mask = y_coords < (rows / 1.8)
        factor = 1.0 + pinch_strength * dist
        
        map_x = np.where(mask, center_x + (dx * factor), x_coords).astype(np.float32)
        map_y = y_coords.astype(np.float32)
        
        distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        if add_noise and random.random() > 0.5:
            distorted = self.add_ultrasound_noise(distorted, noise_factor=0.35)
            
        if add_blur and random.random() > 0.5:
            distorted = cv2.GaussianBlur(distorted, (5, 5), 0)
            
        return distorted, pinch_strength

    def process_file(
        self, 
        img_path: str, 
        output_dir: str, 
        prefix: str = "lemon_v2_"
    ) -> Optional[str]:
        """Processes a single file and saves output."""
        try:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            distorted, _ = self.apply_lemon_deformation(img)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{prefix}{filename}")
            cv2.imwrite(save_path, distorted)
            return save_path
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    def generate_dataset(
        self, 
        input_folder: str, 
        output_folder: str, 
        target_count: int = 1200,
        max_workers: int = 8
    ) -> List[str]:
        """Batch generates synthetic Spina Bifida images from normal scans."""
        files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
            files.extend(glob.glob(os.path.join(input_folder, ext)))
            
        if not files:
            raise FileNotFoundError(f"No source images found in {input_folder}")
            
        print(f"Generating {target_count} Spina Bifida samples from {len(files)} source scans...")
        os.makedirs(output_folder, exist_ok=True)
        
        task_files = []
        while len(task_files) < target_count:
            task_files.extend(files)
        task_files = task_files[:target_count]
        
        generated_paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(lambda f: self.process_file(f, output_folder), task_files)
            for res in results:
                if res:
                    generated_paths.append(res)
                    
        print(f"Successfully generated {len(generated_paths)} Spina Bifida samples in {output_folder}")
        return generated_paths
