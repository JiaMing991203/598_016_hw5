import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from mamba import MambaConfig, Mamba
from atten_model import LayerNorm, Block


class HybridA(nn.Module):
    def __init__(self, vocab_size: int,
                 config: MambaConfig,
                 max_len: int = 11,
                 attn_layers: int = 1) -> None:
        super().__init__()
        assert config.n_layers == 1
        self.embed_dim = config.d_model
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.mamba = Mamba(config)
        self.atten_layer = Block(self.embed_dim, max_len)
        self.ln = LayerNorm(self.embed_dim, True)
        self.head = nn.Linear(self.embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.mamba(x)
        x = self.ln(self.atten_layer(x))
        x = self.head(x)
        return x
    
    
    
class HybridA(nn.Module):
    def __init__(self, vocab_size: int,
                 config: MambaConfig,
                 max_len: int = 11,
                 attn_layers: int = 1) -> None:
        super().__init__()
        assert config.n_layers == 1
        self.embed_dim = config.d_model
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.mamba1 = Mamba(config)
        self.atten_layer = Block(self.embed_dim, max_len)
        self.ln = LayerNorm(self.embed_dim, True)
        self.mamba2 = Mamba(config)
        self.head = nn.Linear(self.embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.mamba1(x)
        x = self.ln(self.atten_layer(x))
        x = self.mamba2(x)
        x = self.head(x)
        return x