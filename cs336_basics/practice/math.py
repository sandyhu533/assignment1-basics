from collections import defaultdict

def round_d_ff(d_model):
    return (d_model / 3 * 8 // 64) * 64

def transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    res = defaultdict(int)
    res["token_embedding"] = vocab_size*d_model
    res["attention"] = d_model*d_model*4
    res["attention"] *= num_layers
    res["ffn"] = d_model*d_ff*3
    res["ffn"] *= num_layers
    res["norms"] = d_model * (2*num_layers+1)
    res["lm_head"] = d_model * vocab_size
    res["total"] = sum(res.values())
    return res

def transformer_forward_flops(batch_size, context_length, vocab_size,
                              num_layers, d_model, num_heads, d_ff=None) -> dict:
    if d_ff is None: d_ff = round_d_ff(d_model)
    res = defaultdict(int)
    res["attention_qkvo"] = 2*context_length*batch_size*d_model*d_model
    res["attention_qkvo"] *= 4*num_layers
    res["attention_scores"] = 2*context_length*context_length*batch_size*d_model
    res["attention_scores"] *= num_layers
    res["attention_softmax_v"] = 2*context_length*batch_size*d_model*context_length
    res["attention_softmax_v"] *= num_layers
    res["ffn"] = 2*context_length*batch_size*d_ff*d_model
    res["ffn"] *= 3*num_layers
    res["lm_head"] = 2*context_length*batch_size*vocab_size*d_model
    res["total"] = sum(res.values())
    return res

def activation_elements(batch_size, context_length, vocab_size,
                        num_layers, d_model, num_heads, d_ff=None) -> dict:
    res = defaultdict(int)
    res["attention"] = batch_size*context_length*d_model*5
    res["attention"] *= num_layers
    res["attention_T2"] = batch_size*context_length*context_length*num_heads*3
    res["attention_T2"] *= num_layers
    res["ffn"] = 5*batch_size*context_length*d_ff+batch_size*context_length*d_model
    res["ffn"] *= num_layers
    res["lm_head"] = batch_size*context_length*d_model
    res["norms"] = batch_size*context_length*d_model*2
    res["norms"] *= (2*num_layers+1)
    res["ce"] = batch_size*context_length*vocab_size*2
    res["total"] = sum(res.values())
    return res

def adamw_peak_memory_bytes(batch_size, context_length, vocab_size,
                            num_layers, d_model, num_heads,
                            d_ff=None, bytes_per_element=4) -> dict:
    res = defaultdict(int)
    res["params"] = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)["total"]*bytes_per_element
    res["gradients"] = res["params"]
    res["optimizer"] = res["params"]*2
    res["activations"] = activation_elements(batch_size, context_length, vocab_size, num_layers, d_model, num_heads, d_ff)["total"]*bytes_per_element
    res["total"] = sum(res.values())
    return res

def adamw_step_flops(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    res = defaultdict(int)
    params = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)["total"]
    res["weight_decay"] = params*2
    res["m_update"] = params*3
    res["v_update"] = params*4
    res["param_update"] = params*5
    res["total"] = sum(res.values())
    return res
    