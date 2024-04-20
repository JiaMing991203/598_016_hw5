import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from mamba import MambaConfig, Mamba
from atten_model import LayerNorm

class MambaTwo(nn.Module):
    def __init__(self, config: MambaConfig, vocab_size: int):
        super().__init__()

        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.mambda = Mamba(config)
        self.head = nn.Linear(config.d_model, vocab_size)
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