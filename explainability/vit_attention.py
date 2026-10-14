"""
Vision Transformer (ViT) Explainability and Attention Rollout Engine.
Implements spatial [CLS] self-attention map extraction and Attention Rollout
(Abnar & Zuidema, 2020) for ultrasound lesion localization in Spina Bifida.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List


class ViTAttentionExplainer:
    """
    Computes explainability maps from Vision Transformer self-attention matrices.
    Supports single-layer CLS-to-patch attention and multi-layer Attention Rollout.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def get_attention_heatmap(
        self, 
        input_tensor: torch.Tensor, 
        method: str = "cls_attention"
    ) -> np.ndarray:
        """
        Extracts a 2D spatial attention heatmap (224x224) normalized to [0.0, 1.0].
        
        Args:
            input_tensor: Input image tensor (1, 3, 224, 224)
            method: 'cls_attention' for final block CLS token attention,
                    or 'attention_rollout' for recursive inter-layer token flow.
                    
        Returns:
            np.ndarray: (224, 224) float32 array in [0.0, 1.0].
        """
        self.model.eval()
        with torch.no_grad():
            if method == "attention_rollout":
                heatmap = self._compute_attention_rollout(input_tensor)
            else:
                heatmap = self._compute_cls_attention(input_tensor)
                
        # Resize to standard ultrasound image dimensions (224, 224)
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
        return heatmap_resized

    def _compute_cls_attention(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Computes CLS token spatial attention from model's get_attention_map method."""
        if hasattr(self.model, "get_attention_map"):
            attn_grid = self.model.get_attention_map(input_tensor) # (1, 14, 14)
            return attn_grid[0].detach().cpu().numpy().astype(np.float32)
        
        # Fallback if get_attention_map not directly implemented
        return np.ones((14, 14), dtype=np.float32) * 0.5

    def _compute_attention_rollout(
        self, 
        input_tensor: torch.Tensor, 
        discard_ratio: float = 0.85, 
        head_fusion: str = "mean"
    ) -> np.ndarray:
        """
        Calculates recursive Attention Rollout:
        A_hat = 0.5 * W_att + 0.5 * I
        Rollout = product_{l=1}^L A_hat_l
        """
        if not hasattr(self.model, "vit") or not hasattr(self.model.vit, "blocks"):
            return self._compute_cls_attention(input_tensor)

        B = input_tensor.shape[0]
        vit = self.model.vit
        
        try:
            x_tokens = vit.patch_embed(input_tensor)
            if hasattr(vit, "cls_token"):
                cls_token = vit.cls_token.expand(B, -1, -1)
                x_tokens = torch.cat((cls_token, x_tokens), dim=1)
            if hasattr(vit, "pos_embed") and vit.pos_embed is not None:
                x_tokens = x_tokens + vit.pos_embed
            if hasattr(vit, "pos_drop"):
                x_tokens = vit.pos_drop(x_tokens)

            num_tokens = x_tokens.shape[1]
            rollout = torch.eye(num_tokens, device=input_tensor.device).unsqueeze(0).repeat(B, 1, 1)

            for blk in vit.blocks:
                B_tok, N_tok, C_tok = x_tokens.shape
                qkv = blk.attn.qkv(x_tokens).reshape(B_tok, N_tok, 3, blk.attn.num_heads, C_tok // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)

                if head_fusion == "mean":
                    attn_fused = attn.mean(dim=1)
                elif head_fusion == "max":
                    attn_fused = attn.max(dim=1)[0]
                else:
                    attn_fused = attn.min(dim=1)[0]

                # Threshold weak attention connections
                flat = attn_fused.view(B, -1)
                val, _ = torch.topk(flat, int(flat.size(-1) * discard_ratio), largest=False)
                threshold = val[:, -1].view(B, 1, 1)
                attn_fused = torch.where(attn_fused < threshold, torch.zeros_like(attn_fused), attn_fused)

                # Add residual identity connection and normalize
                I = torch.eye(num_tokens, device=input_tensor.device).unsqueeze(0)
                a_hat = 0.5 * attn_fused + 0.5 * I
                a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
                rollout = torch.bmm(a_hat, rollout)
                x_tokens = blk(x_tokens)

            # Extract CLS token attention across spatial patches (excluding CLS self-interaction)
            cls_rollout = rollout[:, 0, 1:]
            num_patches = cls_rollout.shape[1]
            grid_size = int(num_patches ** 0.5)
            rollout_grid = cls_rollout.reshape(B, grid_size, grid_size)
            
            # Normalize to [0, 1]
            r_min = rollout_grid.amin(dim=(1, 2), keepdim=True)
            r_max = rollout_grid.amax(dim=(1, 2), keepdim=True)
            norm_rollout = (rollout_grid - r_min) / (r_max - r_min + 1e-7)
            return norm_rollout[0].detach().cpu().numpy().astype(np.float32)
            
        except Exception as e:
            print(f"Attention rollout computation fallback: {e}")
            return self._compute_cls_attention(input_tensor)
