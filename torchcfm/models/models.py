import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class GradModel(torch.nn.Module):
    """
    this is what we use for calculating the div of velocity probably
    """
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class Graph_like_transformer(nn.Module):
    """
    A simplified model for gene expression data that avoids expensive 
    graph operations but still captures gene relationships.
    
    This model uses:
    1. Initial feature extraction per gene
    2. Self-attention for capturing relationships
    3. Global pooling and final prediction
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: int = 0,
        time_varying: bool = False,
        w: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature extraction for each gene
        self.feature_extraction = nn.Sequential(
            nn.Linear(1, w//2),
            nn.LayerNorm(w//2),
            nn.ReLU(),
            nn.Linear(w//2, w),
            nn.LayerNorm(w),
            nn.ReLU(),
        )
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(
                EfficientSelfAttention(
                    w=w, 
                    dropout=dropout
                )
            )
            
        # Final prediction
        self.output_layers = nn.Sequential(
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(w, out_dim)
        )
        
        self.dim = (dim+1) if time_varying else dim
        
    def forward(self, gene_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        gene_vector : torch.Tensor, shape (batch_size, dim)
            A batch of gene expression values.
            
        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, out_dim)
        """
        batch_size, dim = gene_vector.shape
        assert dim == self.dim, "Mismatch between gene_vector size and model's dim"
        
        # Reshape for per-gene processing
        x = gene_vector.unsqueeze(-1)  # [batch_size, dim, 1]
        
        # Extract features for each gene
        x = self.feature_extraction(x)  # [batch_size, dim, w]
        
        # Apply self-attention layers
        for attn_layer in self.attention_layers:
            x = attn_layer(x)
            
        # Global average pooling across genes
        x = x.mean(dim=1)  # [batch_size, w]
        
        # Final prediction
        output = self.output_layers(x)  # [batch_size, out_dim]
        
        return output


class EfficientSelfAttention(nn.Module):
    """
    Memory-efficient self-attention implementation with linear complexity
    """
    def __init__(self, w: int, dropout: float = 0.1):
        super().__init__()
        
        # Using a smaller projection dimension for efficiency
        self.projection_dim = max(32, w // 4)
        
        # Create query, key, value projections
        self.query = nn.Linear(w, self.projection_dim)
        self.key = nn.Linear(w, self.projection_dim)
        self.value = nn.Linear(w, self.projection_dim)
        
        # Output projection
        self.output = nn.Linear(self.projection_dim, w)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(w)
        self.norm2 = nn.LayerNorm(w)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(w, w * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(w * 2, w)
        )
    
    def forward(self, x):
        # x shape: [batch_size, dim, w]
        residual = x
        
        # Normalize first
        x = self.norm1(x)
        
        # Project queries, keys, values
        q = self.query(x)  # [batch_size, dim, projection_dim]
        k = self.key(x)    # [batch_size, dim, projection_dim]
        v = self.value(x)  # [batch_size, dim, projection_dim]
        
        # Efficient attention using associative property:
        # Instead of QK^T * V which is O(n²), we do Q * (K^T * V) which is O(n)
        # First normalize K for numerical stability
        k = F.normalize(k, p=2, dim=2)
        
        # Compute context vector: (K^T * V) then apply softmax
        context = torch.bmm(k.transpose(1, 2), v)  # [batch_size, projection_dim, projection_dim]
        
        # Apply query
        attn_output = torch.bmm(q, context)  # [batch_size, dim, projection_dim]
        
        # Project back to original dimension
        attn_output = self.output(attn_output)
        
        # Add residual connection and apply dropout
        x = residual + self.dropout(attn_output)
        
        # Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        
        return x