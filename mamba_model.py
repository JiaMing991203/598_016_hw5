import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from mamba import MambaConfig, Mamba, RMSNorm
from atten_model import LayerNorm

class MambaTwo(nn.Module):
    def __init__(self, config: MambaConfig, vocab_size: int):
        super().__init__()

        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.mambda = Mamba(config)
        self.norm_f = RMSNorm(self.config.d_model)

        # self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.head = nn.Linear(config.d_model, vocab_size)
    def forward(self, x):
        # x : (B, L, D)

        # logits : (B, L, vocab_size)

        x = self.embed(x)
        x = self.mambda(x)
        x = self.norm_f(x)
        logits = self.head(x)
        return logits
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches