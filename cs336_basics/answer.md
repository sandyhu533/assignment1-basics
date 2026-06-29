# Transformer Accounting

vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288
d_k = 64

## Trainable Params

embedding: vocab_size * d_model = 80,411,200
each transformer block:
- norm * 2 = d_model * 2 = 3,200
- attn = d_model * d_model * 4 = 10,240,000
- fnn = d_model * d_ff * 3 = 20,582,400

total_blocks = 30,825,600 * 48 = 1,479,628,800
final_norm: d_model = 1,600
output_proj: d_model * vocab_size = 80,411,200

total = 1,640,452,800
memory (FP32) = 1,640,452,800 * 4Byte = 6.11GB

## Flops of one forward pass

input token: context_length = 1024
embedding: 0 (lookup only)
each transformer block:
- norm * 2 = 4*d_model*context_length*2 = 13,107,200
- att
    - proj (qkvo): 2*context_length*d_model*d_model*4= 20,971,520,000
    - q*k: 2*num_heads*context_length*context_length*d_k=3,355,443,200
    - scores*v: 2*num_heads*d_k*context_length*context_length=3,355,443,200
    - softmax: 5*num_heads*context_length*context_length=131,072,000
    - rope and mask: neglect
- ffn
    - proj (w1-3): 2*context_length*d_ff*d_model*3=42,152,755,200
    - silu & sigmoid: context_length*d_ff*6=26,345,472

total_blocks = 70,005,686,272*48 = 3,360,272,941,056
final_norm: 4*d_model*context_length = 6,553,600
output_proj: 2*vocab_size*context_length*d_model=164,682,137,600

total = 3,524,961,632,256 = 3.52TFlops

sanity check: Flops ~ 2*N*tokens = 2*1,640,452,800*1024 = 3,359,647,334,400 = 3.35TFlops

if we have KV Cache, for one decode:
- the flops will be total_flops / context_length = 3,280,905,600 = 3.28GFlops


## Peak memory required by AdamW

parameters = N
activations = ?
gradients = N
optimizer states (m, v) = 2*N

### Activations

embedding: c x b x d_model

transformer block:
- rmsnorm * 2 = c x b x d_model x 2
- ffn
1. w1x, w3 = c x b x d_ff x 2
2. w2 = c x b x c_model
3. **Silu*W3** = b x c x d_ff
- attn
1. q, k, v proj = c x b x d_model * 3
2. scores =  b x c x c x num_heads
3. softmax = b x c x c x num_heads
4. qkv result = c x b x d_model
5. out proj = c x b x d_model

final_norm: c x b x d_model
final_proj: c x b x vocab_size

## Flops of one Adamw step

weight decay: 2*N #x, -
m: 3*N
v: 4*N
moment adjusted update: 5*N
