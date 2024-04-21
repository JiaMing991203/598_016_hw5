from dataclasses import dataclass, fields, asdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba, MambaConfig, RMSNorm

"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

# TODO generate function : batch size != 1 ? (for now B=1)
# TODO generate function : top-p sampling

@dataclass
class MambaLMConfig(MambaConfig):
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8

    def __post_init__(self):
        super().__post_init__()

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)



class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig,
                 n_layers: int = 2,
                 d_model: int = 8):
        super().__init__()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()
        self.config.d_model = d_model
        self.config.n_layers = n_layers
        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x = self.mamba(x)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits
    
    def step(self, token, caches):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits, caches
    
    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [(None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=input_ids.device)) for _ in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.step(input_ids[:, i], caches) # (batch_size, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                
        outputs = [tokenizer.decode(output.tolist()) for output in input_ids]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs
    