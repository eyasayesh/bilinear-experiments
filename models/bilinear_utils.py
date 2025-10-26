
import torch
import torch.nn as nn


class BilinearLayer(nn.Module):
    """Bilinear layer that computes interactions different parts of the input and itself
    Args:
        d_input: Input dimension
        d_hidden: Hidden dimension
        d_output: Output dimension
        bias: Whether to include bias term
    """

    def __init__(self, d_input: int, d_hidden: int, d_output: int, bias: bool = False):
        super().__init__()
        
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        
        #The weight matrices for bilinear operation
        self.W = nn.Linear(d_input, d_hidden, bias=False)
        self.V = nn.Linear(d_input, d_hidden, bias=False)

        #downproject to output dimension
        self.P = nn.Linear(d_hidden, d_output, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bilinear layer
        Args:
            x: first input tensor of shape (batch_size, input_dim)
        Returns:
            output: output tensor of shape (batch_size, output_dim)
        """
        w_out = self.W(x)  # (batch_size, d_hidden)
        v_out = self.V(x)  # (batch_size, d_hidden)

        hidden = w_out * v_out  # Element-wise multiplication (batch_size, d_hidden)
        output = self.P(hidden)  # (batch_size, d_output)

        return output
    
    def get_interaction_matrix(self, output_idx):
        """
        Extract interaction matrix Q for a specific output dimension.
        
        The bilinear layer computes: x^T Q x for output dimension 'output_idx'
        where Q = Σ_a P[output_idx, a] * W[a] ⊗ V[a]
        
        Args:
            output_idx: Which output dimension (0 to d_output-1)
        
        Returns:
            Q: Interaction matrix of shape (d_input, d_input)
        """
        device = self.W.weight.device
        
        # Get weight matrices
        W = self.W.weight.data  # (d_hidden, d_input)
        V = self.V.weight.data  # (d_hidden, d_input)
        P = self.P.weight.data  # (d_output, d_hidden)
        
        # Get down-projection weights for this output
        p = P[output_idx, :]  # (d_hidden,)
        
        # Compute interaction matrix: Q = Σ_a p[a] * W[a] ⊗ V[a]
        Q = torch.zeros(self.d_input, self.d_input, device=device)
        
        for a in range(self.d_hidden):
            # Outer product: W[a] ⊗ V[a]
            outer = torch.outer(W[a], V[a])  # (d_input, d_input)
            
            # Weight by down-projection and accumulate
            Q += p[a] * outer
        
        # Symmetrize (only symmetric part contributes to x^T Q x)
        Q = 0.5 * (Q + Q.T)
        
        return Q
    
    def get_bilinear_tensor(self):
        """
        Get the full bilinear tensor B of shape (d_output, d_input, d_input).
        
        B[k, i, j] = Σ_a P[k,a] * W[a,i] * V[a,j]
        
        Returns:
            B: Tensor of shape (d_output, d_input, d_input)
        """
        device = self.W.weight.device
        
        W = self.W.weight.data  # (d_hidden, d_input)
        V = self.V.weight.data  # (d_hidden, d_input)
        P = self.P.weight.data  # (d_output, d_hidden)
        
        # Initialize tensor
        B = torch.zeros(self.d_output, self.d_input, self.d_input, device=device)
        
        # Compute each output slice
        for k in range(self.d_output):
            for a in range(self.d_hidden):
                # Weighted outer product
                B[k] += P[k, a] * torch.outer(W[a], V[a])
            
            # Symmetrize
            B[k] = 0.5 * (B[k] + B[k].T)
        
        return B
    
