
import torch
import torch.nn as nn
from models.bilinear_utils import BilinearLayer
import numpy as np

class BilinearClassifier(nn.Module):
    """
    Complete MNIST classifier with bilinear layer.
    
    Architecture (from paper Figure 21):
        Input (flatten 28×28=784) → Embed (d_embed) → Bilinear (d_hidden) → Output (10)
    
    Args:
        d_embed: Embedding dimension (default: 512)
        d_hidden: Hidden dimension for bilinear layer (default: 512)
        d_output: Output dimension (number of classes, default: 10)
        input_noise: Std of Gaussian noise to add during training (default: 0.5)
        use_bias: Whether to use bias in bilinear layer (default: False)
    """

    def __init__(self, 
                d_input: int = 28*28,
                d_embed: int =512,
                d_hidden: int =512, 
                d_out: int =10,
                input_noise: int =0.5,
                use_bias: bool=False):
        super().__init__()

        self.input_noise = input_noise
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_input, d_embed, bias=False)
        )

        self.bilinear = BilinearLayer(
                            d_input=d_embed,
                            d_hidden=d_hidden,
                            d_output=d_out,
                            bias=use_bias
                        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input images of shape (batch, 1, 28, 28) or (batch, 784)
        
        Returns:
            logits: Output logits of shape (batch, 10)
        """
        
        # Add Gaussian noise during training (key regularization from paper!)
        if self.training and self.input_noise > 0:
            x = x + torch.randn_like(x) * self.input_noise
        
        # Embedding
        h = self.embed(x)  # (batch, d_embed)
        
        # Bilinear layer
        logits = self.bilinear(h)  # (batch, 10)
        
        return logits
    
    def get_interaction_matrix(self, digit):
        """
        Get interaction matrix for a specific digit class.
        
        Args:
            digit: Digit class (0-9)
        
        Returns:
            Q: Interaction matrix in embedding space (d_embed, d_embed)
        """
        return self.bilinear.get_interaction_matrix(digit)
    
    def analyze_digit(self, digit):
        """
        Perform eigendecomposition analysis for a specific digit.
        
        This is the core interpretability method from the paper!
        
        Args:
            digit: Digit class to analyze (0-9)
        
        Returns:
            dict with eigenvalues, eigenvectors, and statistics
        """
        # Get interaction matrix
        Q = self.get_interaction_matrix(digit)
        Q_np = Q.cpu().numpy()
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(Q_np)
        
        # Sort by absolute magnitude (most important)
        sorted_idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Statistics
        pos_eigs = eigenvalues[eigenvalues > 1e-6]
        neg_eigs = eigenvalues[eigenvalues < -1e-6]
        
        # Effective rank (eigenvalues > 1% of max)
        threshold = 0.01 * np.abs(eigenvalues[0])
        effective_rank = np.sum(np.abs(eigenvalues) > threshold)
        
        return {
            'digit': digit,
            'Q': Q_np,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'num_positive': len(pos_eigs),
            'num_negative': len(neg_eigs),
            'max_eigenvalue': eigenvalues[0],
            'min_eigenvalue': eigenvalues[-1],
            'effective_rank': effective_rank,
        }

