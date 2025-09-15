"""
Main model interface for Probabilistic CLIP with Composed Image Retrieval.

This module contains the primary interface class ProbCLIP_CIR that orchestrates
training and inference for composed image retrieval tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer
from util.transforms import targetpad_transform

from .registry import register_model
from .encoders import ProbCLIPModel
from .loss import NeighborhoodDeviationLoss
from .utils import soft_clamp_tanh


@register_model("prob_clip")
class ProbCLIP_CIR(nn.Module):

    def __init__(self, path="clip-vit-large-patch14", neighborhood_loss_weight=0.01, k_neighbors=5,
                 target_ratio=1.25, image_size=224, local_files_only=True, 
                 alpha=0.9, beta=0.1, **kwargs):
        """
        Initialize ProbCLIPModel from a pretrained CLIP model path.
        
        Args:
            path (str): Path to pretrained CLIP model directory or model name.
            neighborhood_loss_weight (float): Weight for the NeighborhoodDeviationLoss.
            k_neighbors (int): Number of neighbors to consider for neighborhood loss.
            target_ratio (float): Target aspect ratio for TargetPad preprocessing.
            image_size (int): Final image dimension for preprocessing.
            local_files_only (bool): Whether to load model from local files only.
            alpha, beta (float): Weight parameter for loss combination.
            **kwargs: Additional keyword arguments (for future extensibility).
        """
        # Load base CLIP model to get the config
        super().__init__()
        self.backbone = ProbCLIPModel.from_pretrained(path, local_files_only=local_files_only)
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=local_files_only)
        self.preprocess = targetpad_transform(target_ratio=target_ratio, dim=image_size)
        
        # Loss configuration
        self.neighborhood_loss_weight = neighborhood_loss_weight
        self.alpha = alpha
        self.beta = beta
        
        # Initialize additional loss function
        self.neighborhood_loss = NeighborhoodDeviationLoss(k_neighbors=k_neighbors)


    def tokenize(self, text, **kwargs):
        """Tokenize text input for the model"""
        return self.tokenizer(text, **kwargs)


    def encode_query(self, ref_imgs, input_ids):
        text_features = self.backbone.get_text_features(input_ids)
        ref_img_features = self.backbone.get_image_features(ref_imgs, feat_modulate=text_features['mean'])

        query_mean = F.normalize(text_features['mean'] + ref_img_features['mean'])
        
        # Combine uncertainties in log-space for numerical stability
        # Note: uncertainty poolers output log(var), so we work in log-space
        log_var_text = text_features['var']  # log(var_text) = log(var_text)
        log_var_img = ref_img_features['var']  # log(var_img) = log(var_img)
        
        # Combine log-variances using log-sum-exp for numerical stability
        # log(var_text + var_img) = logsumexp([log(var_text), log(var_img)])
        log_combined_var = torch.logsumexp(torch.stack([log_var_text, log_var_img], dim=0), dim=0)
        
        # Use the combined log(var) directly
        # Apply final clamping to ensure the result stays in reasonable range
        query_var = log_combined_var

        return {
            'mean': query_mean,
            'var': query_var  # This is log(var)
        }

    def encode_target(self, tgt_imgs):
        tgt_img_features = self.backbone.get_image_features(tgt_imgs)
        return {
            'mean': F.normalize(tgt_img_features['mean']),
            'var': tgt_img_features['var']  # This is log(var)
        }
    
    def warmup(self, ref_imgs, input_ids, tgt_imgs):
        """
        Warmup method using the same loss function as CLIP_CIR.
        Uses deterministic cross-entropy loss instead of probabilistic loss.
        This can be used during the initial training phase for stability.
        
        Args:
            ref_imgs: Reference images
            input_ids: Text input IDs
            tgt_imgs: Target images
            
        Returns:
            Cross-entropy loss (same as CLIP_CIR)
        """
        # Get query and target features (only use the mean, ignore var)
        query_features = self.encode_query(ref_imgs, input_ids)
        target_features = self.encode_target(tgt_imgs)
        
        # Extract only the mean features for deterministic computation
        q_feat = query_features['mean']  # [B, D]
        t_feat = target_features['mean']  # [B, D]
        
        # Compute logits using the same approach as CLIP_CIR
        logits = 100 * q_feat @ t_feat.T  # [B, B]
        labels = torch.arange(ref_imgs.shape[0], dtype=torch.long, device=ref_imgs.device)
        
        # Use cross-entropy loss (same as CLIP_CIR)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    
    def forward(
        self,
        ref_imgs,
        input_ids,
        tgt_imgs,
    ):
        query_features = self.encode_query(ref_imgs, input_ids)
        target_features = self.encode_target(tgt_imgs)
        
        # Main CoPE probabilistic matching loss with query as input1 and target as input2
        main_loss = self.backbone.loss(query_features, target_features)
        
        # Neighborhood deviation loss to regularize uncertainty predictions
        # Apply to both query and target features
        neighborhood_loss_query = self.neighborhood_loss(query_features, query_features)
        neighborhood_loss_target = self.neighborhood_loss(target_features, target_features)
        neighborhood_loss = (neighborhood_loss_query + neighborhood_loss_target) / 2
        
        # Probabilistic component
        probabilistic_loss = main_loss + self.neighborhood_loss_weight * neighborhood_loss
        
        # Deterministic component (cross-entropy using mean features)
        q_feat = query_features['mean']  # [B, D]
        t_feat = target_features['mean']  # [B, D]
        logits = 100 * q_feat @ t_feat.T  # [B, B]
        labels = torch.arange(ref_imgs.shape[0], dtype=torch.long, device=ref_imgs.device)
        deterministic_loss = F.cross_entropy(logits, labels)
        
        # Combine losses with alpha and beta weights
        total_loss = self.alpha * deterministic_loss + self.beta * probabilistic_loss
        
        return total_loss
