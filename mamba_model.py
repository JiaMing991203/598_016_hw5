import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from mamba import ResidualBlock, MambaConfig

class MambaTwo(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(2)])
        self.head = nn.Linear(config.d_model, config.d_state)
    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, vocab_size)

        for layer in self.layers:
            x = layer(x)

        #x = self.norm_f(x)
        x = self.head(x)
        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches