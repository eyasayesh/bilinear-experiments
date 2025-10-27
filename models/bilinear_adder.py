import torch
import torch.nn as nn
from models.bilinear_utils import BilinearLayer
import numpy as np

class BilinearAdder(nn.Module):
    """
    Bilinear layer that adds two inputs using a bilinear transformation.
    
    Architecture:
        Input (d_input = 2xP) -> Bilinear (d_hidden) -> Output (P)
    
    Args:
        d_input: Input dimension (default: 2*113)
        d_hidden: Hidden dimension for bilinear layer (default: 100)
        P: mod value and output dimension (default: 113)
        use_bias: Whether to use bias in bilinear layer (default: False)
    """

    def __init__(self, 
                d_input: int = 2*113,
                d_hidden: int = 100, 
                P: int = 113,
                use_bias: bool=False):
        super().__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.P = P

        self.bilinear = BilinearLayer(
                            d_input=d_input,
                            d_hidden=d_hidden,
                            d_output=P,
                            bias=use_bias
                        )
        
    def forward(self, x):
        """
        Forward pass through the bilinear adder.
        
        Args:
            x: Input tensor of shape (batch, 2*P)
        
        Returns:
            logits: Output logits of shape (batch, P)
        """
        
        logits = self.bilinear(x)
        
        return logits
    
    def get_interaction_matrix(self, output_idx):
        """
        Extract interaction matrix Q for a specific output dimension.
        
        The bilinear layer computes: x^T Q x for output dimension 'output_idx'
        where Q = Σ_a P[output_idx, a] * W[a] ⊗ V[a]
        
        Args:
            output_idx: Which output dimension (0 to P-1)
        Returns:
            Q: Interaction matrix of shape (d_input, d_input)
        """

        Q = self.bilinear.get_interaction_matrix(output_idx)
        return Q
    
    def analyze_mod_digit(self, digit):
        """
        Perform eigendecomposition analysis for a specific modulo digit.
        
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
        sorted_idx = np.argsort(eigenvalues)[::-1]
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
            'interaction_matrix': Q_np,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'num_positive': len(pos_eigs),
            'num_negative': len(neg_eigs),
            'max_eigenvalue': eigenvalues[0],
            'min_eigenvalue': eigenvalues[-1],
            'effective_rank': effective_rank,
        }
