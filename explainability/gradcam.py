"""
Grad-CAM and Grad-CAM++ Implementation for PyTorch Models.
Generates Class Activation Maps highlighting the anatomical regions
influencing the deep learning model's Spina Bifida prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional

class GradCAM:
    """
    Standard Gradient-weighted Class Activation Mapping (Grad-CAM).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = 1
    ) -> np.ndarray:
        """
        Generates normalized Grad-CAM heatmap [0.0, 1.0] for the target class.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by pooled gradients
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        
        # ReLU to keep only features that have positive influence
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)
            
        # Resize to input dimensions (224, 224)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        return heatmap_resized

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ with second and third order gradient weighting for finer localization.
    """
    def generate_heatmap(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = 1
    ) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[:, target_class]
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        grads = self.gradients[0] # (C, H, W)
        activations = self.activations[0] # (C, H, W)
        
        # Grad-CAM++ alpha weights calculation
        grads_power_2 = torch.pow(grads, 2)
        grads_power_3 = torch.pow(grads, 3)
        sum_activations = torch.sum(activations, dim=[1, 2], keepdim=True)
        
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + 1e-7)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))
        
        weights = torch.sum(aij * F.relu(grads), dim=[1, 2])
        
        cam = torch.sum(weights[:, None, None] * activations, dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        else:
            cam = np.zeros_like(cam)
            
        return cv2.resize(cam, (224, 224))
