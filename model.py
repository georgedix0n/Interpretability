"""
Toy Model for Sparse Feature Learning and Mechanistic Interpretability

This module implements a simple autoencoder architecture to study how neural networks
learn to represent features under different sparsity conditions, inspired by 
Anthropic's work on mechanistic interpretability.
"""

import torch
import torch.nn as nn
from typing import Tuple

class SparseFeatureGenerator:
    """
    Generates synthetic data with controlled sparsity levels.
    
    This simulates the kind of sparse feature activations observed in
    biological and artificial neural networks, where only a subset of
    features are active at any given time.
    """
    
    def __init__(self, n_features: int, sparsity_levels: list):
        """
        Args:
            n_features: Number of possible features in the data
            sparsity_levels: List of sparsity probabilities (0 = dense, 1 = fully sparse)
        """
        self.n_features = n_features
        self.sparsity_levels = torch.tensor(sparsity_levels)
        self.n_instances = len(sparsity_levels)
        
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generate a batch of data with instance-specific sparsity.
        
        Returns:
            Tensor of shape (n_instances, batch_size, n_features)
        """
        # Generate random feature values
        values = torch.rand(self.n_instances, batch_size, self.n_features)
        
        # Create sparsity mask (True = keep feature, False = zero it out)
        mask = torch.rand(self.n_instances, batch_size, self.n_features) > self.sparsity_levels[:, None, None]
        
        return values * mask

class ToyAutoencoder(nn.Module):
    """
    Simple 2-layer autoencoder with tied weights, designed to study
    feature learning under sparsity constraints.
    
    The architecture is intentionally simple to make the learned representations
    interpretable, following the principle that understanding toy models
    can provide insights into more complex systems.
    """
    
    def __init__(self, n_instances: int, n_features: int, n_hidden: int):
        """
        Args:
            n_instances: Number of parallel model instances (one per sparsity level)
            n_features: Input/output dimensionality
            n_hidden: Hidden layer size (creates information bottleneck)
        """
        super().__init__()
        self.n_instances = n_instances
        self.n_features = n_features
        self.n_hidden = n_hidden
        
        # Weight matrix shared between encoder and decoder (tied weights)
        # Shape: (n_instances, n_hidden, n_features)
        self.W = nn.Parameter(torch.empty(n_instances, n_hidden, n_features))
        nn.init.xavier_normal_(self.W)
        
        # Bias term
        self.b = nn.Parameter(torch.zeros(n_instances, n_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode to hidden layer, then decode back.
        
        The tied weights (W and W^T) create a symmetric architecture
        that encourages learning meaningful feature representations.
        """
        # Encoder: project input to hidden space
        # For each instance, compute h = W @ x
        hidden = torch.einsum('ihf,ibf->ibh', self.W, x)
        
        # Decoder: reconstruct from hidden representation
        # Reconstructed = W^T @ h
        reconstruction = torch.einsum('ihf,ibh->ibf', self.W, hidden)
        
        # Apply ReLU and add bias
        return torch.nn.functional.relu(reconstruction + self.b[:, None, :])