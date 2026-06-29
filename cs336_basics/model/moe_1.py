import torch
from torch import nn
from cs336_basics.model.transformer import *

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_shared, n_routed, top_k, device=None):
        super().__init__()
        self.router = Linear(d_model, n_routed, device)
        self.shared_experts = nn.ModuleList(
            SwiGLU(d_model, d_ff, device) for _ in range(n_shared)
        )
        self.routed_experts = nn.ModuleList(
            SwiGLU(d_model, d_ff, device) for _ in range(n_routed)
        )
        self.top_k = top_k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape # [B, L, D]
        x = x.reshape(-1, D) # [T, D]
        probs = self.router(x).softmax(dim=-1) # [T, D]
        gate, indices = probs.topk(self.top_k, dim=-1) # [T, top_k]; [T, top_k]
        self.last_router_probs = probs
        self.last_router_indices = indices
        
        out = torch.zeros_like(x) # [T, D]
        for shared_exp in self.shared_experts:
            out += shared_exp(x) # [T, D]
        for e, exp in enumerate(self.routed_experts):
            token_mask = (indices == e) # [T, top_k]
            sel = token_mask.any(dim=-1) # [T]
            if not sel.any():
                continue
            token_ids = sel.nonzero(as_tuple=True)[0] # [T] -select-> [N] (N <= T)
            w = (gate * token_mask).sum(dim=-1)[token_ids] # [T, top_k] -> [T] -> [N]
            contrib = exp(x[token_ids]) # [T, D] -> [N, D] -> [N, D]
            out = out.index_add(0, token_ids, w.unsqueeze(-1)*contrib) # [N, D] +> [T, D]
        return out.reshape(B, L, D) # [B, L, D]

class MoELayerBalanced(MoELayer):
    def __init__(self, d_model, d_ffn, n_shared, n_routed, top_k,
                 balance_coeff: float = 0.001, device=None):
        super().__init__(d_model, d_ffn, n_shared, n_routed, top_k, device)
        self.balance_coeff = balance_coeff

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x = x.reshape(-1, D)
        out = torch.zeros_like(x)
        for exp in self.shared_experts:
            out = out + exp(x)
        
        probs:torch.Tensor = self.router(x).softmax(dim=-1)
        gate, indices = probs.topk(self.top_k, dim=-1)
        self.last_router_probs = probs
        self.last_router_indices = indices
        
        T = B*L
        n_routed = len(self.routed_experts)
        p = 1/T*probs.sum(dim=0) # (n_routed)
        f = torch.zeros_like(p)
        
        if n_routed == 0:
            return out, x.new_zeros(())
        
        for eid, exp in enumerate(self.routed_experts):
            mask = (indices == eid) # (T, top_k)
            sel = mask.any(dim=-1) # (T)
            if not sel.any(): continue
            token_ids = sel.nonzero(as_tuple=True)[0] # (T) -> (N)
            w = (gate * mask).sum(dim=-1, keepdim=True)[token_ids] # (T, top_k) -> (T, 1) -> (N, 1)
            contrib = exp(x[token_ids])
            out = out.index_add(dim=0, index=token_ids, source=w*contrib)
            f[eid] += token_ids.shape[0]
        
        f = n_routed/(self.top_k*T)*f
        loss = self.balance_coeff*(f*p).sum(dim=-1)
        
        return out.reshape(B, L, D), loss