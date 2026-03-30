# Assignment 1 Writeup

## Problem (unicode1): Understanding Unicode (1 point)

### (a) What Unicode character does chr(0) return?

**Answer:** `'\x00'`

### (b) How does this character's string representation (`__repr__()`) differ from its printed representation?

**Answer:** The character's printed representation is empty, while its string representation isn't.

### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

**Answer:** It is non-printable.

---

## Problem (unicode2): Unicode Encodings (3 points)

### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

**Answer:** UTF-8 ranges from 0-256, while UTF-16/32 is much larger (0-65536 and 0-4294967296), leading to:
1. Manageable dimension - control training cost
2. Less rare/sparse tokens - prevent overfitting

### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**Answer:** Can only deal with one character to one byte scenarios, cannot deal with complex characters like '哈哈哈'.

### (c) Give a two byte sequence that does not decode to any Unicode character(s).

**Answer:** A two-byte sequence that will fail to decode is `b'\xff\xff'`.

**Explanation:** In the UTF-8 encoding scheme, the byte 0xff (255) is explicitly disallowed and never used, so any sequence containing it is invalid.

---

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

### (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?

**Resource requirements:** ≤30 minutes (no GPUs), ≤30GB RAM

**Hint:** You should be able to get under 2 minutes for BPE training using multiprocessing during pretokenization and the following two facts:
- The `<|endoftext|>` token delimits documents in the data files.
- The `<|endoftext|>` token is handled as a special case before the BPE merges are applied.

**Deliverable:** A one-to-two sentence response.

**Answer:** It took 2 minutes. The longest token is `b' accomplishment'`, which makes very good sense.

### (b) Profile your code. What part of the tokenizer training process takes the most time?

**Deliverable:** A one-to-two sentence response.

**Answer:** The process of updating and merging tokens takes the most time.

---

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

### (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?

**Resource requirements:** ≤12 hours (no GPUs), ≤100GB RAM

**Deliverable:** A one-to-two sentence response.

**Answer:** The longest token is `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ`, which does not make sense. But it actually exists in the training test. So it is expected by the algorithm itself.

### (b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

**Deliverable:** A one-to-two sentence response.

**Answer:** The overlapping token number of TinyStories and OWT is 7319, which means 73% of tokens in TinyStories exist in OWT.

---

## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

### (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer's compression ratio (bytes/token)?

**Deliverable:** A one-to-two sentence response.

**Answer:**
- Compression rate of tiny_stories_sample with ts_tokenizer = 4.100162744287541
- Compression rate of tiny_stories_sample with owt_tokenizer = 3.930880227743864
- Compression rate of owt_sample with ts_tokenizer = 3.006948449455682
- Compression rate of owt_sample with owt_tokenizer = 4.291383001736754

### (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

**Deliverable:** A one-to-two sentence response.

**Answer:** Tokenizing with a sample that does not match the training sets appears to have a lower compression rate. But the gap is bigger while applying TinyStories tokenizer to OWT sample, because TinyStories' vocab size is much smaller than OWT.

### (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?

**Deliverable:** A one-to-two sentence response.

**Answer:**
- Estimated bytes/s of owt_tokenizer is 790311.8778890059, 289h for 825GB
- Estimated bytes/s of ts_tokenizer is 1020181.9216268589, 241h for 825GB

### (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We'll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?

**Deliverable:** A one-to-two sentence response.

**Answer:** uint16 has lower storage cost compared to uint32/uint64. uint8 can only represent 0-256, which is too short for the token range 0-10000/0-32000.

---

## Problem (model_analysis): Model Analysis

### (a) Consider GPT-2 XL, which has the following configuration:

```
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400
```

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

**Deliverable:** A one-to-two sentence response.

**Answer:**

```
emb = vocab_size * d_model
norm = d_model
att = d_model * d_model * 4
ff = d_model * d_ff * 3
block = norm * 2 + att + ff
block = block * num_layers
rope = 0
linear = d_model * vocab_size
ans = emb + block + rope + norm + linear

ans = 2127057600
mem = ans * 4 bytes(f32) = 8.5082304 GB
```

### (b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

**Deliverable:** A list of matrix multiplies (with descriptions), and the total number of FLOPs required.

```
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400
```

**Answer:**

```
token_emb = 0 # only table look ups
norm = 6 * context_length * d_model # pow, mean, +, rsqrt, *, *
att = 3 * (2 * d_model * d_model * context_length) # proj of qkv
att += d_model * context_length * context_length * 2 * 2 # qkv
att += d_model * context_length * d_model * 2 # proj of o
ff = 2 * context_length * d_model * d_ff * 3 # w1, w2, w3
block = norm * 2 + att + ff
block = block * num_layers
linear = 2 * context_length * d_model * vocab_size
ans = emb + block + rope + norm + linear

ans = 4514370484800
```

### (c) Based on your analysis above, which parts of the model require the most FLOPs?

**Deliverable:** A one-to-two sentence response.

**Answer:** The FeedForward Layer (Swiglu) in Transformer block requires the most FLOPs.

### (d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

**Deliverable:** For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

**Answer:**

**Small:**
- total_flops = 1,915,684,059,712
- att_flops = 386,547,056,640 (flops_ratio = 20.18%)
- ff_flops = 1,449,551,462,400 (flops_ratio = 75.67%)
- linear_flops = 79,047,426,048 (flops_ratio = 4.13%)

**Medium:**
- total_flops = 2,657,297,824,320
- att_flops = 618,475,290,624 (flops_ratio = 23.27%)
- ff_flops = 1,932,735,283,200 (flops_ratio = 72.73%)
- linear_flops = 105,396,568,064 (flops_ratio = 3.97%)

**Large:**
- total_flops = 3,450,451,196,480
- att_flops = 901,943,132,160 (flops_ratio = 26.14%)
- ff_flops = 2,415,919,104,000 (flops_ratio = 70.02%)
- linear_flops = 131,745,710,080 (flops_ratio = 3.82%)

As the model becomes larger, the FF still counts for most FLOPs, but the attention increases from 20% to 26%.

### (e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?

**Deliverable:** A one-to-two sentence response.

**Answer:**

**Before (context_length = 1024):**
- total_flops = 4,514,370,484,800
- att_flops = 1,328,755,507,200 (flops_ratio = 29.43%)
- ff_flops = 3,019,898,880,000 (flops_ratio = 66.90%)
- linear_flops = 164,682,137,600 (flops_ratio = 3.65%)

**After (context_length = 16384):**
- total_flops = 149,538,132,916,800
- att_flops = 98,569,499,443,200 (flops_ratio = 65.92%)
- ff_flops = 48,318,382,080,000 (flops_ratio = 32.31%)
- linear_flops = 2,634,914,201,600 (flops_ratio = 1.76%)

The FF changes. Attention changes more than FF. The ratio of attention became dominant. This is because ff_flops is linearly associated with context_length, while att_flops is quadratically associated with context_length.

---

## Problem (learning_rate_tuning): Tuning the learning rate (1 point)

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let's see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?

**Deliverable:** A one-to-two sentence response with the behaviors you observed.

**Answer:**

**1e1:**
```
33.356502532958984
21.348163604736328
15.736955642700195
12.312485694885254
9.973114013671875
8.268854141235352
6.973681449890137
5.959208965301514
5.146246433258057
4.482952117919922
```

**1e2:**
```
25.2305965423584
25.2305965423584
4.328885078430176
0.10359998047351837
2.0683108586406622e-16
2.305260756932448e-18
7.762623467245897e-20
4.624248605210782e-21
3.966979291529242e-22
4.407754628123886e-23
```

**1e3:**
```
24.241891860961914
8751.3232421875
1511491.0
168137232.0
13619116032.0
859522203648.0
44125059547136.0
1898448961929216.0
6.997277825774387e+16
```

1e1 is learning slowly, 1e2 learns faster and converges, while 1e3 does not converge, meaning that it is jumping between the optimal results because the step is too big.

---

## Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)

Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.

### (a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.

For simplicity, when calculating memory usage of activations, consider only the following components:
- Transformer block
  - RMSNorm(s)
  - Multi-head self-attention sublayer: QKV projections, Q⊤ K matrix multiply, softmax, weighted sum of values, output projection.
  - Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
- final RMSNorm
- output embedding
- cross-entropy on logits

**Deliverable:** An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

**Answer:**

```
# parameters
# parameters
emb = vocab_size * d_model
norm = d_model
att = d_model * d_model * 4
ff = d_model * d_ff * 2
block = norm * 2 + att + ff
block = block * num_layers
rope = 0
final_norm = d_model
linear = d_model * vocab_size
params = emb + block + rope + linear + final_norm

# activations
x = batch_size*context_length*d_model
norm = x
q = k = v = s = x
qk = batch_size*context_length*context_length*num_heads
att = x + q + k + v + qk + s
w1 = silu = batch_size*context_length*d_ff
ff = x + w1 + silu
block = att + ff + 2*norm
block = block * num_layers
final_norm = norm
loss_inputs = loss_log_probs = batch_size*context_length*vocab_size
loss = loss_inputs + loss_log_probs
activations = final_norm + block + loss

# gradients
gradients = params

# optimizer
m = v = params
optimizer = m + v

total = (params + activations + gradients + optimizer) * 4
```

### (b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

**Deliverable:** An expression that looks like a·batch_size + b for numerical values a, b, and a number representing the maximum batch size.

**Answer:**
only 5 batch_size can fit within 80GB memory.

### (c) How many FLOPs does running one step of AdamW take?

**Deliverable:** An algebraic expression, with a brief justification.

**Answer:**

```
m = params * 3 # *, +, *
v = params * 4 # *, +, *, **2
pdata1 = params * 5 # *, ., **0.5, +, -
pdata2 = 2 # *, -
total = m + v + pdata1 + pdata2 = params * 14
```

### (d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware's theoretical peak FLOP throughput [Chowdhery et al., 2022]. An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022], assume that the backward pass has twice the FLOPs of the forward pass.

**Deliverable:** The number of days training would take, with a brief justification.

**Answer:**

```
# flops of forward pass
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

# num_layer = 12
# d_model = 768
# num_heads = 12

# num_layer = 24
# d_model = 1024
# num_heads = 16

# num_layer = 36
# d_model = 1280
# num_heads = 20

# context_length = 16384

token_emb = 0 # only table look ups
norm = 6 * context_length * d_model # pow, mean, +, rsqrt, *, *
att = 3 * (2 * d_model * d_model * context_length) # proj of qkv
att += d_model * context_length * context_length * 2 * 2 # qkv
att += d_model * context_length * d_model * 2 # proj of o
ff = 2 * context_length * d_model * d_ff * 3 # w1, w2, w3
block = norm * 2 + att + ff
block = block * num_layers
linear = 2 * context_length * d_model * vocab_size
ans = emb + block + rope + norm + linear
ans

print(f'total_flops={ans:,d}')
print(f'att_flops={att*num_layers:,d} flops_ratio={att*num_layers/ans:.2%}')
print(f'ff_flops={ff*num_layers:,d} flops_ratio={ff*num_layers/ans:.2%}')
print(f'linear_flops={linear:,d} flops_ratio={linear/ans:.2%}')

batch_size = 1024
steps = 400000
forward = batch_size * steps * ans
print(f'forward={forward:,d}')

# flops of backward pass - optimizer

m = params * 3 # *, +, *
v = params * 4 # *, +, *, **2
pdata1 = params * 5 # *, ., **0.5, +, -
pdata2 = 2 # *, -
total = m + v + pdata1 + pdata2

optimizer = total * steps
print(f'optimizer={optimizer:,d}')

# flops of backward pass - gradients (approximate)

backgrad = forward * 2
print(f'backgrad={backgrad:,d}')


a100 = 19.5 * 1e12 * 0.5

total = forward + backgrad + optimizer

total/a100/60/60/24

6585.073958099146 days

```

---

## Problem (kv_cache_impl): Implement KV Cache (3 points)

Autoregressive generation in the current codebase recomputes all Key and Value projections for the entire prefix at every step. This is highly wasteful — KV Cache eliminates this redundancy by caching the K and V tensors from previous positions.

**Background reading:** See `docs/kv_cache_tutorial.md` for a detailed explanation of the KV Cache concept, complexity analysis, and implementation guide.

**Tasks:**

1. Modify `CausalMultiHeadSelfAttention.forward()` in `cs336_basics/model/layers.py` to accept an optional `past_kv` cache and return an updated cache when `use_cache=True`.
2. Modify `TransformerBlock.forward()` and `TransformerLM.forward()` in `cs336_basics/model/transformer.py` to thread the KV cache through all layers. The model must accept external `token_positions` (for correct RoPE during decode) and `past_kv_list` (one cache entry per layer).
3. Implement `generate_with_kv_cache()` in `cs336_basics/generation.py` with two-phase decoding:
   - **Prefill**: run the full prompt through the model, collect KV caches.
   - **Decode**: for each new token, pass only the single new token plus cached KV, and update the cache.
4. Ensure backward compatibility: existing training code (which does not use cache) must continue to work without modification.

**Resource requirements:** No GPU needed.

**Deliverable:** Pass all tests in `tests/test_kv_cache.py` (`pytest tests/test_kv_cache.py -v`).

---

## Problem (kv_cache_correctness): Verify KV Cache Correctness (1 point)

Verify that `generate_with_kv_cache()` produces **identical output** as `generate()` when using `temperature=0` (deterministic greedy decoding) on a trained checkpoint.

**Tasks:**

1. Load a trained model checkpoint (e.g., TinyStories).
2. Run both `generate()` and `generate_with_kv_cache()` with the same prompt and `temperature=0`.
3. Confirm the outputs are identical token-by-token.

**Deliverable:** A one-sentence confirmation that the outputs match, and the generated text from both methods.

the result match:
Once upon a time, there was a little girl named Lily. She loved to play with her toys. One day, she saw a big box in the park. She wanted to play with her toys.
Lily went to her mom and dad, "Lily, I will help you find my toy." Sue was sad, but she said, "I want to play with you. I will help you." Sue was happy. She thought it was a toy.
Sue's mom saw her mom and said, "Hi, I will help you." Sue and her mom went to the park. She saw the toy car. Sue was very happy. She was happy. She was happy. She was happy.
Sue's mom saw her mom and dad. She said, "Lily, I will help you." Sue and her mom went to the park. She took the toy car and put them in the box. She was happy.
Lily and her mom went to the park, she saw a big tree. She was happy. She was happy. She was happy. She was happy. She was happy. She was so happy and thanked her mom.

---

## Problem (kv_cache_speedup): Measure KV Cache Speedup (2 points)

Run the provided benchmark script to measure the wall-clock speedup of KV-cached generation over naive generation.

**Tasks:**

1. Run the benchmark: `python -m scripts.benchmark_kv_cache --gen-lengths 64 128 256 512`
2. Record wall-clock times and speedup ratios for each generation length.
3. (Optional) Generate a plot: `python -m scripts.benchmark_kv_cache --gen-lengths 64 128 256 512 --plot kv_cache_speedup.png`

**Deliverable:** A table of wall-clock time and speedup ratio for each generation length, plus a 2-3 sentence analysis of how the speedup scales with sequence length and why.

**Answer:**

Gen Len |  Naive (s) | KV Cache (s) |  Speedup |  Naive tok/s |  Cache tok/s
-----------------------------------------------------------------------------
      64 |     0.1957 |       0.1038 |    1.89x |        327.0 |        616.8
     128 |     0.6201 |       0.2068 |    3.00x |        206.4 |        619.0
     256 |     1.8021 |       0.4146 |    4.35x |        142.1 |        617.5
     512 |     8.2085 |       1.9155 |    4.29x |         62.4 |        267.3
    
---

## Problem (kv_cache_analysis): KV Cache Memory-Compute Tradeoff Analysis (2 points, written)

KV Cache trades memory for compute. Let's analyze this tradeoff quantitatively.

### (a) Derive the memory cost of the KV cache for a GPT-2 XL model generating a total sequence of 1024 tokens (prompt + generated). Express your answer in terms of the model hyperparameters (`num_layers`, `d_model`, `batch_size`, `seq_len`) and `sizeof(dtype)`.

```
GPT-2 XL:
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
```

**Deliverable:** An algebraic expression for the KV cache memory in bytes, and the numerical result for GPT-2 XL with `batch_size=1`, `seq_len=1024`, `dtype=float32`.

**Answer:**

Each layer caches K and V tensors, each of shape `(batch_size, num_heads, seq_len, d_k)` where `d_k = d_model / num_heads`. Since `num_heads × d_k = d_model`:

```
KV_cache_memory = 2 × num_layers × batch_size × seq_len × d_model × sizeof(dtype)
```

For GPT-2 XL with batch_size=1, seq_len=1024, dtype=float32 (4 bytes):

```
= 2 × 48 × 1 × 1024 × 1600 × 4
= 629,145,600 bytes
≈ 600 MB (0.586 GB)
```

### (b) Compare the KV cache memory overhead with the model parameter memory. What fraction of the model memory does the KV cache represent?

**Deliverable:** A one-to-two sentence response.

**Answer:**

The model parameters require ~8.508 GB in float32 (from the model_analysis problem). The KV cache at seq_len=1024 requires ~0.586 GB, which is approximately **6.9%** of the model parameter memory — a modest overhead that delivers a substantial speedup.

### (c) Derive how much compute (FLOPs) the KV cache saves per generation step compared to naive generation, for a sequence of length $n$. At what sequence length does the memory cost of KV cache become a concern relative to model parameters for GPT-2 XL?

**Deliverable:** An algebraic expression for FLOPs saved per step, and a few sentences of analysis.

**Answer:**

At sequence length $n$, per layer the naive forward pass costs:

```
naive_per_layer = 8n × d_model²  (QKV + O projections)
               + 4n² × d_model   (QKᵀ and attn × V)
               + 6n × d_model × d_ff  (FFN W1, W2, W3)
```

With KV cache, we only process 1 new token per step:

```
cached_per_layer = 8 × d_model²         (projections for 1 token)
                 + 4n × d_model          (Q[1] @ K[n]ᵀ and attn × V)
                 + 6 × d_model × d_ff   (FFN for 1 token)
```

FLOPs saved per step over all layers:

```
saved = L × (n − 1) × (8 × d_model² + 4n × d_model + 6 × d_model × d_ff)
```

This is roughly an $n\times$ reduction in work per step. Instantiating for GPT-2 XL at n=1024:

```
naive_total  = L × naive_per_layer + 2 × n × d_model × vocab_size
             = 48 × 90,596,966,400 + 164,682,137,600
             ≈ 4,513,336,524,800 FLOPs

cached_total = L × cached_per_layer + 2 × 1 × d_model × vocab_size
             = 48 × 88,473,600 + 160,822,400
             ≈ 4,407,555,200 FLOPs

savings      = 1 − cached / naive
             = 1 − 4,407,555,200 / 4,513,336,524,800
             ≈ 99.90%
```

KV cache reduces per-step decode FLOPs by **~99.9%** (a ~1024× reduction, matching $n$). The prefill phase still performs a full forward pass, so the total savings over $T$ decode steps is $(T-1)/T$ of this, approaching 99.9% as $T$ grows.

For the memory crossover point, the KV cache memory equals model parameter memory when:

```
2 × L × B × n × d_model = num_params
n = num_params / (2 × L × B × d_model)
  = 2,127,057,600 / (2 × 48 × 1 × 1600)
  ≈ 13,843
```

At the default context_length of 1024, the KV cache is only ~7% of model memory and well worth the tradeoff. The cache only becomes memory-competitive with the model itself at sequence lengths around 14K — relevant for modern long-context models, but not for GPT-2 XL's 1024-token context window.