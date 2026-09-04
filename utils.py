"""
Utility functions for analyzing and visualizing the learned representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import os

def analyze_representations(model) -> Dict:
    """
    Analyze the learned weight matrices to extract insights about
    what features the model learned to represent.
    
    Returns:
        Dictionary containing various metrics about the learned representations
    """
    W = model.W.detach().cpu().numpy()  # (n_instances, n_hidden, n_features)
    n_instances = W.shape[0]
    
    analysis = {
        'W': W,
        'effective_features': [],
        'feature_overlap': [],
        'weight_norms': [],
        'gram_matrices': []
    }
    
    for i in range(n_instances):
        Wi = W[i]  # (n_hidden, n_features)
        
        # Compute effective number of features (based on weight magnitude)
        feature_magnitudes = np.linalg.norm(Wi, axis=0)
        effective_threshold = 0.1 * feature_magnitudes.max()
        n_effective = np.sum(feature_magnitudes > effective_threshold)
        analysis['effective_features'].append(n_effective)
        
        # Compute feature overlap (how similar are the learned features)
        normalized = Wi / (np.linalg.norm(Wi, axis=0) + 1e-8)
        gram = normalized.T @ normalized
        analysis['gram_matrices'].append(gram)
        
        # Average absolute cosine similarity between features
        off_diagonal = gram[~np.eye(gram.shape[0], dtype=bool)]
        analysis['feature_overlap'].append(np.abs(off_diagonal).mean())
        
        # Weight norms
        analysis['weight_norms'].append(np.linalg.norm(Wi))
    
    return analysis

def visualize_results(model, generator, analysis: Dict, history: Dict, save_dir: str = 'results'):
    """
    Create comprehensive visualizations of the results.
    
    Args:
        model: Trained ToyAutoencoder model
        generator: SparseFeatureGenerator instance
        analysis: Dictionary from analyze_representations
        history: Training history dictionary
        save_dir: Directory to save results
    """
    
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    W = analysis['W']
    n_instances = W.shape[0]
    sparsity_levels = generator.sparsity_levels.numpy()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid: 3 rows, n_instances columns
    gs = fig.add_gridspec(3, n_instances, hspace=0.3, wspace=0.3)
    
    sparsity_labels = [f"S={s:.3f}" for s in sparsity_levels]
    
    for i in range(n_instances):
        Wi = W[i]  # (n_hidden, n_features)
        
        # Row 1: Feature directions as 2D arrows
        ax1 = fig.add_subplot(gs[0, i])
        colors = plt.cm.viridis(np.linspace(0, 0.9, Wi.shape[1]))
        
        for f in range(Wi.shape[1]):
            ax1.arrow(0, 0, Wi[0, f], Wi[1, f], 
                     head_width=0.05, length_includes_head=True,
                     color=colors[f], linewidth=2, alpha=0.8)
        
        # Add feature labels
        for f in range(Wi.shape[1]):
            if np.linalg.norm(Wi[:, f]) > 0.1:
                ax1.text(Wi[0, f]*1.1, Wi[1, f]*1.1, f'F{f}', 
                        fontsize=8, alpha=0.7)
        
        lim = max(1.5, np.abs(Wi).max() * 1.2)
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_aspect('equal')
        ax1.axhline(0, color='gray', lw=0.5, alpha=0.5)
        ax1.axvline(0, color='gray', lw=0.5, alpha=0.5)
        ax1.set_title(f'Feature Directions\n{sparsity_labels[i]}', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Row 2: Gram matrix (feature relationships)
        ax2 = fig.add_subplot(gs[1, i])
        gram = analysis['gram_matrices'][i]
        im = ax2.imshow(gram, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_xticks(range(gram.shape[0]))
        ax2.set_yticks(range(gram.shape[0]))
        ax2.set_xlabel('Feature index', fontsize=8)
        ax2.set_ylabel('Feature index', fontsize=8)
        ax2.set_title(f'Feature Relationships\n(W^T W)', fontsize=10)
        
        # Add colorbar for the last column
        if i == n_instances - 1:
            plt.colorbar(im, ax=ax2)
        
        # Row 3: Feature importance/magnitude
        ax3 = fig.add_subplot(gs[2, i])
        feature_magnitudes = np.linalg.norm(Wi, axis=0)
        bars = ax3.bar(range(len(feature_magnitudes)), feature_magnitudes, 
                      color=plt.cm.viridis(np.linspace(0, 0.9, len(feature_magnitudes))))
        ax3.set_xlabel('Feature index', fontsize=8)
        ax3.set_ylabel('Magnitude', fontsize=8)
        ax3.set_title('Feature Importance', fontsize=10)
        ax3.set_xticks(range(len(feature_magnitudes)))
    
    # Add main title
    fig.suptitle('Analysis of Learned Representations Across Sparsity Levels\n' +
                 'Toy Model for Mechanistic Interpretability', 
                 fontsize=16, y=1.02)
    
    # Save figure
    plt.savefig(os.path.join(save_dir, 'representation_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot training curve
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['step'], history['loss'], linewidth=2, color='blue')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training Convergence', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to '{save_dir}/' directory")