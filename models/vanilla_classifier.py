
import torch.nn as nn
import torch.nn.functional as F

class VanillaClassifier(nn.Module):
    """A simple feedforward MNIST classifier
        - Input: 28x28=784 dim vector corresponding to an MNIST image
        - Embedding Layer: Linear layer with n_embed hidden units
        - FFN Layer: Linear layer with n_h hidden units
        - Head Layer: Linear layer with 10 output units (one per class)
    """

    def __init__(self, n_input = 28*28, n_embed=256, n_h=128, n_classes=10):
        super().__init__()

        # image to embedding layer
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, n_embed, bias=False)
        )

        #vanilla MLP
        self.mlp = nn.Sequential(  
            nn.Linear(n_embed, n_h, bias=False),
            nn.ReLU(),
        )

        #hidden layer to logits
        self.head = nn.Linear(n_h, n_classes, bias=False)

    def forward(self, x):
        """Forward pass through the network
        Args:
            x: input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            logits: output tensor of shape (batch_size, n_classes)
        """
        embeddings = self.embed(x)
        embeddings = self.mlp(embeddings)
        logits = self.head(embeddings)

        
        return logits
