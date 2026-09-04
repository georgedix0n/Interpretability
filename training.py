"""
Training script for the sparse feature learning model.

This script trains multiple instances of the autoencoder in parallel,
each with a different sparsity level, to study how sparsity affects
the learned feature representations.
"""

import torch
import torch.optim as optim
from model import SparseFeatureGenerator, ToyAutoencoder
from utils import visualize_results, analyze_representations
import os

def train_model(
    n_instances: int = 5,
    n_features: int = 5,
    n_hidden: int = 2,
    sparsity_levels: list = [0.0, 0.7, 0.9, 0.99, 0.999],
    batch_size: int = 1024,
    n_steps: int = 10000,
    learning_rate: float = 1e-3,
    log_every: int = 500,
    seed: int = 42
):
    """
    Train the autoencoder across multiple sparsity levels.
    
    The key insight being explored: how does increasing sparsity
    change what features the model learns to represent?
    
    Args:
        n_instances: Number of parallel model instances
        n_features: Number of features in the data
        n_hidden: Hidden layer size (bottleneck)
        sparsity_levels: List of sparsity probabilities
        batch_size: Batch size for training
        n_steps: Number of training steps
        learning_rate: Learning rate for optimizer
        log_every: How often to log progress
        seed: Random seed for reproducibility
    
    Returns:
        model: Trained model
        generator: Data generator
        training_history: Dictionary with loss history
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Initialize data generator and model
    generator = SparseFeatureGenerator(n_features, sparsity_levels)
    model = ToyAutoencoder(n_instances, n_features, n_hidden)
    
    # Loss function: weighted MSE
    importance = torch.ones(n_features)
    
    def loss_fn(predictions: torch.Tensor, targets: torch.Tensor, 
                importance: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss for each instance."""
        squared_error = importance * (targets - predictions) ** 2
        return squared_error.sum(dim=-1).mean(dim=-1)  # Shape: (n_instances,)
    
    # Setup optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    
    # Training loop
    training_history = {'loss': [], 'step': []}
    
    print("Starting training...")
    print(f"Training {n_instances} instances with sparsity levels: {sparsity_levels}")
    print("-" * 60)
    
    for step in range(n_steps):
        # Generate fresh batch
        x = generator.generate_batch(batch_size)
        
        # Forward pass
        predictions = model(x)
        
        # Compute loss (sum over instances)
        loss = loss_fn(predictions, x, importance).sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Log progress
        if step % log_every == 0:
            training_history['loss'].append(loss.item())
            training_history['step'].append(step)
            print(f"Step {step:6d} | Total Loss: {loss.item():.6f}")
    
    print("-" * 60)
    print("Training complete!")
    
    return model, generator, training_history

def main():
    """Main execution: train model and visualize results."""
    
    # Configuration
    config = {
        'n_instances': 5,
        'n_features': 5,
        'n_hidden': 2,  # Small hidden layer forces feature selection
        'sparsity_levels': [0.0, 0.7, 0.9, 0.99, 0.999],
        'batch_size': 1024,
        'n_steps': 10000,
        'learning_rate': 1e-3,
        'log_every': 500,
        'seed': 42
    }
    
    # Train the model
    model, generator, history = train_model(**config)
    
    # Analyze and visualize results
    print("\nAnalyzing learned representations...")
    analysis = analyze_representations(model)
    
    # Create visualizations
    visualize_results(model, generator, analysis, history)
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    for i, sparsity in enumerate(config['sparsity_levels']):
        effective_features = analysis['effective_features'][i]
        print(f"\nSparsity {sparsity:.3f}:")
        print(f"  - Learned {effective_features} effective features")
        print(f"  - Feature overlap: {analysis['feature_overlap'][i]:.3f}")
        print(f"  - Weight norm: {analysis['weight_norms'][i]:.3f}")
    
    return model, analysis

if __name__ == "__main__":
    model, analysis = main()