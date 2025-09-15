"""
Loss functions for Probabilistic CLIP models.

This module contains all loss function implementations used for training
probabilistic CLIP models, including the main CoPE loss and auxiliary losses.
"""

import torch
import torch.nn as nn


class ClosedFormSampledDistanceLoss(nn.Module):
    def __init__(
            self,
            init_shift=5,
            init_negative_scale=5,
            vib_beta=0,
            smoothness_alpha=0,
            prob_distance='csd',
            **kwargs):
        super().__init__()

        shift = init_shift * torch.ones(1)
        negative_scale = init_negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.vib_beta = vib_beta
        self.smoothness_alpha = smoothness_alpha
        self.prob_distance = prob_distance

        self.bceloss = nn.BCEWithLogitsLoss()

    def _recompute_matched(self, matched, logits, smoothness=0):
        """ Recompute the `matched` matrix if the smoothness value is given.
        """
        if not smoothness:
            return matched, None
        else:
            logits = logits.view(matched.size())
            # XXX Warning: all negative pairs will return weird results
            gt_labels, gt_indices = torch.max(matched, dim=1)
            gt_vals = logits[:, gt_indices].diag()
            pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1))
            new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices))
            _matched = matched.clone()
            _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices]

            return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)

    def _compute_prob_matching_loss(self, logits, matched, smoothness=0):
        matched, n_pseudo_gts = self._recompute_matched(matched, logits, smoothness)
        loss = self.bceloss(logits, matched)

        return {
            'loss': loss,
            'n_pseudo_gts': n_pseudo_gts,
        }

    # In ClosedFormSampledDistanceLoss._compute_closed_form_loss()
    def _compute_closed_form_loss(self, input1, input2, matched, smoothness=0):
        """ Closed-form probabilistic matching loss -- See Eq (1) and (2) in the paper.
        
        Args:
            input1, input2: dicts with 'mean' and 'var' keys
                - 'mean': feature means
                - 'var': log(variance) values
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1)
        # Note: input1['var'] and input2['var'] are log(var), so torch.exp() converts to var
        sigma_pdist = ((torch.exp(input1['var']).unsqueeze(1) + torch.exp(input2['var']).unsqueeze(0))).sum(-1)

        # Add monitoring
        # if self.training:
        #     print(f"Loss components - mu_pdist: {mu_pdist.mean().item():.3f}, sigma_pdist: {sigma_pdist.mean().item():.3f}")
        #     print(f"Var ranges - input1: [{input1['var'].min().item():.3f}, {input1['var'].max().item():.3f}]")
        #     print(f"Var ranges - input2: [{input2['var'].min().item():.3f}, {input2['var'].max().item():.3f}]")
    
        logits = mu_pdist + sigma_pdist
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(logits, matched, smoothness=smoothness)
        return loss_dict

    def forward(self, input1, input2, matched=None):
        """
        Args:
            input1, input2: dicts with 'mean' and 'var' keys
                - 'mean': feature means
                - 'var': log(variance) values
            matched: ground truth matching matrix (optional)
        """
        if matched is None:
            matched = torch.eye(len(input1['mean'])).to(input1['mean'].device)
        
        loss_dict = self._compute_closed_form_loss(input1, input2, matched=matched)
        loss = loss_dict['loss']
        
        return loss


class CoPELoss(nn.Module):
    """
    Implements a loss function based on the distance metric:
    d(z1, z2) = ||μ1 - μ2||^2 + ||σ1 - σ2||^2 + 2D * σ̄1 * σ̄2
    """
    def __init__(
            self,
            init_shift=5,
            init_negative_scale=5,
            **kwargs):
        """
        Initializes the loss module.

        Args:
            init_shift (float): Initial value for the learnable shift parameter.
            init_negative_scale (float): Initial value for the learnable scale parameter.
        """
        super().__init__()

        # Learnable parameters to scale and shift the distance into a logit
        shift = init_shift * torch.ones(1)
        negative_scale = init_negative_scale * torch.ones(1)

        self.shift = nn.Parameter(shift)
        self.negative_scale = nn.Parameter(negative_scale)

        # Standard binary cross-entropy loss for probabilistic matching
        self.bceloss = nn.BCEWithLogitsLoss()

    def _compute_cope_loss(self, input1, input2, matched):
        """
        Computes the loss based on the new distance metric.

        Args:
            input1, input2: Dictionaries with 'mean' and 'var' keys.
                - 'mean': Feature means of shape (N, D) or (M, D)
                - 'var': log(variance) values of shape (N, D) or (M, D)
            matched: Ground truth matching matrix of shape (N, M).
        """
        # Extract means and convert log(var) to var
        mu1, mu2 = input1['mean'], input2['mean']
        sigma1_sq, sigma2_sq = torch.exp(input1['var']), torch.exp(input2['var'])
        sigma1, sigma2 = torch.sqrt(sigma1_sq), torch.sqrt(sigma2_sq)
        
        # Ensure consistent device
        device = mu1.device
        mu2 = mu2.to(device)
        sigma2 = sigma2.to(device)
        matched = matched.to(device)

        # Get the dimension D from the feature size
        D = mu1.shape[-1]

        # --- Calculate the three terms of the distance metric ---

        # Term 1: Squared Euclidean distance between means
        # ||μ1 - μ2||^2
        # Resulting shape: (N, M)
        term1 = torch.sum((mu1.unsqueeze(1) - mu2.unsqueeze(0)) ** 2, dim=-1)

        # Term 2: Squared Euclidean distance between standard deviations
        # ||σ1 - σ2||^2
        # Resulting shape: (N, M)
        term2 = torch.sum((sigma1.unsqueeze(1) - sigma2.unsqueeze(0)) ** 2, dim=-1)

        # Term 3: Cross-term involving the average standard deviations
        # 2 * D * σ̄1 * σ̄2
        # First, calculate the average std for each sample: σ̄ = (1/D) * Σ(σ_k)
        sigma1_bar = torch.mean(sigma1, dim=-1) # Shape: (N,)
        sigma2_bar = torch.mean(sigma2, dim=-1) # Shape: (M,)
        
        # Then, compute the cross-term using broadcasting
        # Resulting shape: (N, M)
        term3 = 2 * D * sigma1_bar.unsqueeze(1) * sigma2_bar.unsqueeze(0)

        # --- Combine terms to get the final distance ---
        distance = term1 + term2 + term3
        
        # Transform the distance into logits for the BCE loss
        # A smaller distance should result in a higher logit (more likely to be a match)
        logits = -self.negative_scale * distance + self.shift

        # Compute the final loss
        loss = self.bceloss(logits, matched)

        return {'loss': loss}


    def forward(self, input1, input2, matched=None):
        """
        Forward pass for the loss calculation.

        Args:
            input1, input2: Dictionaries with 'mean' and 'var' keys.
            matched (torch.Tensor, optional): Ground truth matching matrix. 
                                              If None, assumes an identity matrix 
                                              for self-comparison.
        """
        if matched is None:
            # Create an identity matrix for the case of comparing a batch to itself
            num_samples = len(input1['mean'])
            matched = torch.eye(num_samples).to(input1['mean'].device)

        loss_dict = self._compute_cope_loss(input1, input2, matched=matched)
        
        return loss_dict['loss']


class NeighborhoodDeviationLoss(nn.Module):
    def __init__(self, k_neighbors=5, **kwargs):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.mse_loss = nn.MSELoss()

    def _compute_neighborhood_deviation_loss(self, input1, input2, matched=None):
        """
        Compute MSE between model's predicted standard deviation and 
        standard deviation of K-neighboring means in the batch.
        
        Args:
            input1, input2: dicts with 'mean' and 'var' keys
                - 'mean': feature means [batch_size, feature_dim]
                - 'var': log(variance) values [batch_size, feature_dim]
        """
        # Use input1 for both predicted std and neighborhood computation
        # Convert log(variance) to standard deviation: std = sqrt(exp(log_var))
        predicted_std = torch.sqrt(torch.exp(input1['var']))  # [batch_size, feature_dim]
        
        means = input1['mean']  # [batch_size, feature_dim]
        batch_size = means.shape[0]
        
        # Compute pairwise distances between means
        # Distance matrix: [batch_size, batch_size]
        means_expanded = means.unsqueeze(1)  # [batch_size, 1, feature_dim]
        means_tiled = means.unsqueeze(0)     # [1, batch_size, feature_dim]
        distances = torch.norm(means_expanded - means_tiled, dim=2)  # [batch_size, batch_size]
        
        # For each sample, find K nearest neighbors (excluding itself)
        k = min(self.k_neighbors, batch_size - 1)  # Ensure k doesn't exceed available neighbors
        
        if k == 0:
            # If no neighbors available, return zero loss
            return torch.tensor(0.0, device=means.device, requires_grad=True)
        
        # Get indices of k nearest neighbors (excluding self)
        # Set diagonal to infinity to exclude self-distances
        distances_masked = distances + torch.eye(batch_size, device=distances.device) * float('inf')
        _, neighbor_indices = torch.topk(distances_masked, k, dim=1, largest=False)  # [batch_size, k]
        
        # Gather neighbor means for each sample
        batch_indices = torch.arange(batch_size, device=means.device).view(-1, 1).expand(-1, k)
        neighbor_means = means[neighbor_indices]  # [batch_size, k, feature_dim]
        
        # Compute standard deviation of neighbor means for each sample
        neighbor_std = torch.std(neighbor_means, dim=1)  # [batch_size, feature_dim]
        
        # Compute MSE between predicted std and neighbor std
        loss = self.mse_loss(predicted_std, neighbor_std)
        
        return loss

    def forward(self, input1, input2, matched=None):
        loss = self._compute_neighborhood_deviation_loss(input1, input2, matched=matched)
        return loss
