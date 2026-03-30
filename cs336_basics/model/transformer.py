"""Transformer blocks and language model."""

import torch
import torch.nn as nn

from cs336_basics.data import batch
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

    def forward(self, x, token_positions=None, past_kv=None, use_cache=False):
        norm1 = self.norm(x)
        if token_positions is None:
            seq_len = x.shape[-2]
            batch_size = x.shape[0]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        if use_cache:
            att, new_kv = self.mha(norm1, token_positions, past_kv=past_kv, use_cache=True)
        else:
            att = self.mha(norm1, token_positions)
            new_kv = None
        x = x + att
        norm2 = self.norm2(x)
        ff = self.ff(norm2)
        x = x + ff
        if use_cache:
            return x, new_kv
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

    def forward(self, x, token_positions=None, past_kv_list=None, use_cache=False):
        x = self.embeddings(x)
        new_kv_list = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past_kv = past_kv_list[i] if past_kv_list is not None else None
            if use_cache:
                x, new_kv = block(x, token_positions=token_positions, past_kv=layer_past_kv, use_cache=True)
                new_kv_list.append(new_kv)
            else:
                x = block(x, token_positions=token_positions)
        x = self.norm(x)
        x = self.liner(x)
        if use_cache:
            return x, new_kv_list
        return x
