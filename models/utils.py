"""
Utility functions and helper classes for Probabilistic CLIP models.

This module contains supporting utilities, attention mechanisms, and pooling
operations used by the probabilistic CLIP architecture.
"""

import torch
import torch.nn as nn
from .gpo import GPO


def soft_clamp_tanh(x, min_val, max_val, k=1.0):
    """
    Implements soft clipping using tanh.

    Args:
        x (torch.Tensor): Input tensor.
        min_val (float): Desired minimum value.
        max_val (float): Desired maximum value.
        k (float): Scaling factor controlling steepness of curve. Larger k approaches hard clipping.
    """
    # Calculate center and range of interval
    center = (max_val + min_val) / 2.0
    half_range = (max_val - min_val) / 2.0

    # Apply soft clipping using tanh
    # k * (x - center) controls input scaling, determining where steep part of curve occurs
    scaled_tanh = torch.tanh(k * (x - center) / half_range)

    # Map tanh output from (-1, 1) to (min_val, max_val)
    return center + half_range * scaled_tanh


class CrossAttentionModulationBlock(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, num_heads=8):
        super().__init__()
        # Project text features to image dimension, used to generate K and V
        self.kv_projection = nn.Linear(text_dim, image_dim)
        # Image features themselves are used as Q, so no projection needed
        self.attention = nn.MultiheadAttention(embed_dim=image_dim, num_heads=num_heads, batch_first=True)

    def forward(self, image_feature, text_feature):
        # image_feature (Query): [B, L, 1024]
        # text_feature: [B, 768]
        
        # 1. Project text features to generate Key and Value
        # [B, 768] -> [B, 1024]
        text_kv = self.kv_projection(text_feature)
        
        # 2. Add sequence dimension, since Attention expects (B, Seq_Len, Dim)
        # [B, 1024] -> [B, 1, 1024]
        text_kv = text_kv.unsqueeze(1)
        
        # 3. Perform cross-attention
        # Q comes from image, K and V come from text
        # Output shape: [B, L, 1024]
        modulation_signal, _ = self.attention(
            query=image_feature,
            key=text_kv,
            value=text_kv
        )
        
        return modulation_signal


class UncertaintyPooler(nn.Module):
    """
    Uncertainty pooler that combines projection and GPO (Generalized Pooling Operator).
    
    Note: The output represents log(variance), not variance itself.
    This log-space representation provides numerical stability and is expected by the loss function.
    """
    def __init__(self, img_dim, embed_size, sigma_ln_init=0.01, sigma_ln_init_bias=0):
        super().__init__()
        self.gpo = GPO(32, 32)
        self.proj = nn.Linear(img_dim, embed_size)
        
        # Add sigma_ln_init mechanism for uncertainty control
        self.sigma_ln_init = sigma_ln_init
        if sigma_ln_init is not None:
            self.std_ln = nn.LayerNorm(embed_size)
            # Remove the "safe" adjustment that makes values too large
            # safe_sigma_ln_init = max(sigma_ln_init, 0.1)
            nn.init.constant_(self.std_ln.weight, sigma_ln_init)  # Use original small value
            nn.init.constant_(self.std_ln.bias, sigma_ln_init_bias)
        else:
            self.std_ln = nn.Identity()

    def _initialize_weights(self):
        """
        Initialize weights using CLIP-style initialization to prevent NaN values.
        Based on the reference implementation and CLIP's initialization strategy.
        """
        # Initialize projection layer with proper scaling
        # Use standard deviation proportional to input dimension size (similar to CLIP)
        std = self.proj.in_features ** -0.5
        nn.init.normal_(self.proj.weight, std=std)
        
        # Initialize projection bias based on sigma_ln_init availability
        if self.proj.bias is not None:
            if self.sigma_ln_init is None:
                # As there is no layer norm, set the bias of the linear layer to -4 to prevent too large std
                nn.init.constant_(self.proj.bias, -4.0)
            else:
                nn.init.zeros_(self.proj.bias)

        # Initialize GPO components properly
        # GRU initialization (PyTorch default is usually fine, but ensure it's set properly)
        for name, param in self.gpo.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        # GPO linear layer initialization
        nn.init.xavier_uniform_(self.gpo.linear.weight)
        # Note: GPO linear layer has no bias (bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [B, L, D] - Hidden states from uncertainty layers
            
        Returns:
            torch.FloatTensor: [B, D] - log(variance) for each feature dimension
        """

        # x: [B, L, D]
        x = self.proj(x)  # [B, L, D]
        
        # GPO expects features and lengths
        # For vision transformers, all sequences have the same length
        batch_size, seq_len, _ = x.shape
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
        
        x, _ = self.gpo(x, lengths)  # [B, D] 
        
        x = self.std_ln(x)  # Apply sigma LayerNorm for uncertainty control
        
        # Clamp log(var) to prevent extreme values that cause NaN
        # Reasonable range: log(var) in [-10, 10] means var in [4.5e-5, 22026]
        # In UncertaintyPooler.forward(), change clamping range
        x_clamped = soft_clamp_tanh(x, min_val=-10.0, max_val=-8.0)
        
        # Add random noise during training using reparameterization trick
        if self.training:
            # The noise scale should be comparable to the uncertainty value itself
            # Since x_clamped is log(var), we add noise in log-space
            # Convert log(var) to std: std = sqrt(exp(log(var))) = exp(log(var)/2)
            log_std = x_clamped * 0.5  # log(std) = log(var) / 2
            
            # Sample noise from standard normal distribution
            eps = torch.randn_like(x_clamped)
            
            # Scale noise by the predicted standard deviation (reparameterization trick)
            # noise_scale = exp(log(std)) = std
            noise_scale = torch.exp(log_std) 
            
            # Apply noise: log(var_noisy) ≈ log(var + noise)
            # For small noise, log(var + noise) ≈ log(var) + noise/var
            # But we use a more stable approach: add noise proportional to std in log space
            noise = eps * noise_scale * 0.1  # Scale down noise to prevent destabilization
            
            # Add noise to the log(var) values
            x_clamped = x_clamped + noise
            
            # Re-clamp after adding noise to ensure we stay in valid range
            x_clamped = soft_clamp_tanh(x_clamped, min_val=-10.0, max_val=-8.0)
        
        return x_clamped  # This outputs log(var) with optional noise injection
