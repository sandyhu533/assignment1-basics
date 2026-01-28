"""Transformer blocks and language model."""

import torch
import torch.nn as nn

from cs336_basics.model.layers import (
    CausalMultiHeadSelfAttention,
    Linear,
    Embedding,
    RMSNorm,
    Swiglu,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.mha = CausalMultiHeadSelfAttention.with_rope(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ff = Swiglu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x):
        norm1 = self.norm(x)
        seq_len = x.shape[-2]
        att = self.mha(norm1, torch.arange(0, seq_len, device=x.device))
        x = x + att
        norm2 = self.norm2(x)
        ff = self.ff(norm2)
        x = x + ff
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layer: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layer)
            ]
        )
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.liner = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.liner(x)
        return x
