import torch
from torch import nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super().__init__()
        weight = torch.zeros((out_features, in_features), device=device)
        std = (2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(weight, 0, std, -3*std, 3*std)
        self.weight = nn.Parameter(weight)
    
    # flops: 
    #       2 * ... * out_feautures * in_features
    # activation:
    #       [self.weight], x
    def forward(self, x):
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim, device=None):
        super().__init__()
        weight = torch.zeros((num_embedding, embedding_dim), device=device)
        nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        self.weight = nn.Parameter(weight)
    
    # flops: 0
    # activation: token_ids
    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None):
        super().__init__()
        weight = torch.ones((d_model,), device=device)
        self.weight = nn.Parameter(weight)
        self.eps = eps
    
    # flops: 
    #       4BLD (pow, sum, divied, *)
    #       3BL (divied, add, sqrt)
    # activation:
    #       [self.weight]: D
    #       x/rms, x: 2BLD
    #       rms: BL
    def forward(self, x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        d_model = x.size(-1)
        rms = torch.sqrt(1/d_model*x.pow(2).sum(dim=-1,keepdim=True)+self.eps)
        return (x/rms*self.weight).to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device)
        self.w3 = Linear(d_model, d_ff, device)
        self.w2 = Linear(d_ff, d_model, device)
    
    # flops:
    #       6BLDF (w1(), w2(), w3())
    #       6BLF (sigmoid-neg, divide, add, e^(-x); gate*w3x; w1x*sigmoid)
    # activation:
    #       [self.w1.weight, self.w2.weight, self.w3.weight]: 3DF
    #       x: BLD
    #       sigmoid, gate, w1x, w3x, gate*w3x: 5BLF
    def forward(self, x:torch.Tensor):
        w1x = self.w1(x)
        gate = w1x*torch.sigmoid(w1x)
        return self.w2(gate*self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        k = torch.arange(0, d_k, 2, device=device)
        seqs = torch.arange(0, max_seq_len, 1, device=device)
        angles = torch.outer(seqs, 1/theta**(k/d_k))
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)
    
    def forward(self, x, token_positions):
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]
        pairs = rearrange(x, "... (d two) -> ... d two", two=2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        new_even = cos*even-sin*odd
        new_odd = sin*even+cos*odd
        new_pairs = torch.stack([new_even, new_odd], dim=-1)
        return rearrange(new_pairs, "... d two -> ... (d two)")
    
# flops:
#           amax, -, exp, sum, divide
# activations:
#           x, xexp, xexpsum, maxv
def softmax(x:torch.Tensor, dim=-1):
    maxv = x.amax(dim=dim, keepdim=True) #x, maxv
    xexp = (x-maxv).exp() #xexp
    return xexp / xexp.sum(dim=dim, keepdim=True) #xexp, xexpsum

# flops:
#       4BLLD: q@k, attn@v
#       7BLL*num_heads: divied, mask, softmax(5)
# activations:
#       q, k, v: 3BLD
#       softmax:
#               x(qk_masked), xexp: 2BLL*num_heads
#               xexpsum, maxv: 2BL*num_heads
#       mask: LL
#       attn: BLL*num_heads
def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None):
    d_k = q.size(-1)
    qk = einsum(q, k, "... query d_k, ... key d_k -> ... query key")/d_k**0.5 # q, k, d_k**0,5
    if mask is not None:
        qk_masked = torch.where(mask, qk, float('-inf')) # mask
    attn = softmax(qk_masked, dim=-1)
    return einsum(attn, v, "... query key, ... key d_k -> ... query d_k") # attn, v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, d_model, device)
        self.v_proj = Linear(d_model, d_model, device)
        self.output_proj = Linear(d_model, d_model, device)
        self.rope = rope
        self.num_heads = num_heads
        
    # flops:
    #       rope: neglect
    #       8BLDD: qkv proj
    # ---------------------------------
    #       4BLLD: q@k, attn@v
    #       7BLL*num_heads: divied, mask, softmax(5)
    # activations:
    #       [self.q_proj, self.k_proj, self.v_proj, self.output_proj]
    #       x: BLD
    #       res: BLD
    #       q, k, v: 3BLD
    # ---------------------------------
    #       softmax:
    #               x(qk_masked), xexp: 2BLL*num_heads
    #               xexpsum, maxv: 2BL*num_heads
    #       mask: LL
    #       attn: BLL*num_heads
    def forward(self, x, token_positions=None):
        q = rearrange(self.q_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        context_length = v.size(-2)
        if token_positions is not None and self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.tril(torch.ones((context_length, context_length),dtype=torch.bool, device=q.device))
        res = scaled_dot_product_attention(q, k, v, mask)
        res = rearrange(res, "... num_heads context_length d_k -> ... context_length (num_heads d_k)")
        return self.output_proj(res)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, rope=None, device=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device)
        self.ffn = SwiGLU(d_model, d_ff, device)
    
    # activations:
    #       2*ln: x/rms, x: 2BLD
    #           rms: BL
    #       attn:     
    #           x: BLD
    #           res: BLD
    #           q, k, v: 3BLD
    # ---------------------------------
    #           softmax:
    #               x(qk_masked), xexp: 2BLL*num_heads
    #               xexpsum, maxv: 2BL*num_heads
    #           mask: LL
    #           attn: BLL*num_heads
    #       ffn:
    #           x: BLD
    #           sigmoid, gate, w1x, w3x, gate*w3x: 5BLF
    
    
    # total activations:
    # BL - 2, LL 0- 1, BL*num_heads - 2
    # BLD - 10
    # BLL*num_heads - 3
    # BLF - 5
    def forward(self, x, token_positions=None):
        y = self.ln1(x)
        y = self.attn(y, token_positions)
        x = x + y
        y = self.ln2(x)
        y = self.ffn(y)
        return x + y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, d_ff, num_heads, rope=None, device=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, rope, device) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device)
    
    def forward(self, token_ids, token_positions=None):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
        