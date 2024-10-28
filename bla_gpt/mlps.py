import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GeGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = self.c_gate(x)
        x = self.c_fc(x)
        x = F.gelu(gate) * x
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = self.c_gate(x)
        x = self.c_fc(x)
        x = F.silu(gate) * x
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Negout_MLP(nn.Module):
    def __init__(self, config):
        """Negout MLP
        I developed Negout when I was a child
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc_neg = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        pos = self.c_fc(x)
        neg = self.c_fc_neg(x)
        x = torch.where(neg > pos, -neg, pos)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Maxout_MLP(nn.Module):
    def __init__(self, config):
        """Maxout MLP implementation
        Uses maxout activation function that takes the maximum value
        across multiple linear transformations
        """
        super().__init__()
        self.num_pieces = 2  # number of pieces for maxout

        # Create multiple linear layers for maxout
        self.c_fc = nn.ModuleList(
            [
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
                for _ in range(self.num_pieces)
            ]
        )

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Apply each linear transformation
        pieces = [fc(x) for fc in self.c_fc]

        # Stack the pieces and take maximum along the pieces dimension
        x = torch.stack(pieces, dim=-1)
        x = torch.max(x, dim=-1)[0]

        # Project back to original dimension and apply dropout
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
