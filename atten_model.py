import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
# from mamba import ResidualBlock, MambaConfig
# self attention block
class Block(nn.Module):
    def __init__(self, embed_dim, max_len=11):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.c_attn = nn.Linear(embed_dim, embed_dim*3)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
    def forward(self, x):
        T = x.size(1)
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y
    

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class BaseNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, 
                 is_pe = False, max_len=11, 
                 attn_layers=2, block=None,
                 **kwargs):
        super(BaseNet, self).__init__()
        if block is None:
            raise ValueError("block type should be provided.")
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.is_pe = is_pe
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = nn.Embedding(max_len, embed_dim) if is_pe else None
        self.att = nn.ModuleList([block(embed_dim, max_len, **kwargs) for _ in range(attn_layers)])
        self.ln = nn.ModuleList([LayerNorm(embed_dim, True) for _ in range(attn_layers)])
        self.head = nn.Linear(embed_dim, vocab_size)
    
        print(f"BaseNet with {attn_layers} layers of {block} blocks")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Positional Encoding: {is_pe}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Context length: {max_len}")
        
    def forward(self, x):
        b, t = x.size()
        x = self.embed(x)
        if self.is_pe:
            pos = torch.arange(0, t, dtype=torch.long, device=x.device)
            pe_emb = self.pe(pos) if self.is_pe else 0
            x = x + pe_emb
        for layer, ln in zip(self.att, self.ln):
            x = ln(layer(x) + x)
            print(f'{x.size()=}')
        x = self.head(x)
        return x
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        # Create a rotation matrix.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        rotation_matrix = np.zeros((d_model, d_model))
        for i in range(d_model):
            for j in range(d_model):
                rotation_matrix[i, j] = np.cos(i * j * 0.01)
        self.rotation_matrix = torch.from_numpy(rotation_matrix).to(self.device)

        # Create a positional embedding matrix.
        positional_embedding = np.zeros((max_seq_len, d_model))
        for i in range(max_seq_len):
            for j in range(d_model):
                positional_embedding[i, j] = np.cos(i * j * 0.01)
        self.positional_embedding = torch.from_numpy(positional_embedding).to(self.device)

    def forward(self, x):
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """

        # Add the positional embedding to the input tensor.
        x += self.positional_embedding

        # Apply the rotation matrix to the input tensor.
        x = torch.matmul(x, self.rotation_matrix)

        return x




class AttnRope(nn.Module):
    def __init__(self, vocab_size, embed_dim, 
                 max_len=11, attn_layers=2, block=None,
                 **kwargs):
        super().__init__()
        if block is None:
            raise ValueError("block type should be provided.")
        self.vocab_size = vocab_size
        self.max_len = max_len
        # self.is_pe = is_pe
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = nn.ModuleList([RotaryPositionalEmbedding(embed_dim, max_len) for _ in range(attn_layers)])
        self.att = nn.ModuleList([block(embed_dim, max_len, **kwargs) for _ in range(attn_layers)])
        self.ln = nn.ModuleList([LayerNorm(embed_dim, True) for _ in range(attn_layers)])
        self.head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        for pe_emb, layer, ln in zip(self.pe, self.att, self.ln):
            x = ln(layer(pe_emb(x)))
        x = self.head(x)
        return x
    
    
def get_rotary_matrix(context_len: int, embedding_dim: int) -> torch.Tensor:
    """
    Generate the Rotary Matrix for ROPE

    Args:
        context_len (int): context len
        embedding_dim (int): embedding dim

    Returns:
        torch.Tensor: the rotary matrix of dimension context_len x embedding_dim x embedding_dim
    """
    R = torch.zeros((context_len, embedding_dim, embedding_dim), requires_grad=False)
    positions = torch.arange(1, context_len+1).unsqueeze(1)
    # Create matrix theta (shape: context_len  x embedding_dim // 2)
    slice_i = torch.arange(0, embedding_dim // 2)
    theta = 10000. ** (-2.0 * (slice_i.float()) / embedding_dim) 
    m_theta = positions * theta
    # Create sin and cos values
    cos_values = torch.cos(m_theta)
    sin_values = torch.sin(m_theta)
    # Populate the rotary matrix R using 2D slicing
    R[:, 2*slice_i, 2*slice_i] = cos_values
    R[:, 2*slice_i, 2*slice_i+1] = -sin_values
    R[:, 2*slice_i+1, 2*slice_i] = sin_values
    R[:, 2*slice_i+1, 2*slice_i+1] = cos_values
    return R


class RoPEBlock(nn.Module):
    def __init__(self, embed_dim, max_len=11):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.c_attn = nn.Linear(embed_dim, embed_dim*3)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len)))
        self.rope = get_rotary_matrix(max_len, embedding_dim=embed_dim)
    def forward(self, x):
        T = x.size(1)
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        
        queries_rot = (k.transpose(0,1) @ self.rope).transpose(0,1)
        keys_rot = (k.transpose(0,1) @ self.rope).transpose(0,1)
        y = torch.nn.functional.scaled_dot_product_attention(q, keys_rot, queries_rot, is_causal=True)
        return y
    
