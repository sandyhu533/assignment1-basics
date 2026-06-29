import torch
from torch import nn
from cs336_basics.model.transformer import *

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_shared, n_routed, top_k, device=None):
        super().__init__()
        self.router = Linear(d_model, n_routed)
        self.shared_experts = nn.ModuleList(
            SwiGLU(d_model, d_ffn, device) for _ in range(n_shared)
        )
        self.routed_experts = nn.ModuleList(
            SwiGLU(d_model, d_ffn, device) for _ in range(n_routed)
        )
        self.top_k = top_k
    
    def forward(self, x:torch.Tensor):
        B, L, D = x.shape
        x = x.reshape(-1, D)
        out = torch.zeros_like(x)
        
        for expert in self.shared_experts:
            out += expert(x)
        
        probs = self.router(x).softmax(dim=-1)
        gate, indices = probs.topk(self.top_k, dim=-1)
        self.last_router_indices = indices
        self.last_router_probs = probs
    
        for eid, expert in enumerate(self.routed_experts):
            mask:torch.Tensor = (indices == eid)
            sel = mask.any(dim=-1)
            if not sel.any(dim=-1): continue
            token_ids = sel.nonzero(as_tuple=True)[0]
            w = (gate * mask).sum(dim=-1, keepdim=True)[token_ids] # (N, 1)
            contrib = expert(x[token_ids]) # (N, D)
            out = out.index_add(dim=0, index=token_ids, source=w*contrib)
        
        return out.reshape(B, L, D)

class MoELayerBalanced(MoELayer):
    def __init__(self, d_model, d_ffn, n_shared, n_routed, top_k,
                 balance_coeff: float = 0.001, device=None):
        super().__init__(d_model, d_ffn, n_shared, n_routed, top_k, device=device)
        self.balance_coeff = balance_coeff
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = super().forward(x)
        n_routed = len(self.routed_experts)
        if n_routed == 0:
            return out, x.new_zeros(())
        
        probs, indices = self.last_router_probs, self.last_router_indices
        T, _ = probs.shape
        f = torch.zeros(n_routed, device=x.device)
        f = f.scatter_add(dim=-1, index=indices.reshape(-1), src=torch.ones(indices.numel(), device=x.device))
        f = (n_routed / (self.top_k * T)) * f
        loss = (self.balance_coeff * (f * probs.mean(0))).sum()
        return out, loss