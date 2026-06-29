from collections import defaultdict

def round_d_ff(d_model: int) -> int:
    return (8/3*d_model // 64) * 64

def transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    if d_ff==None: d_ff=round_d_ff(d_model)
    dic = defaultdict(int)
    dic["token_embedding"] = vocab_size*d_model
    dic["norms"] = d_model*(2*num_layers+1)
    dic["attention"] = d_model*d_model*4*num_layers #qkvo
    dic["ffn"] = d_model*d_ff*3*num_layers #w1-3
    dic["lm_head"] = d_model*vocab_size
    dic["total"] = sum(dic.values())
    return dic

def transformer_forward_flops(batch_size, context_length, vocab_size,
                              num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    dic["attention_qkvo"] = 4*2*batch_size*context_length*d_model*d_model*num_layers
    dic["attention_scores"] = 2*batch_size*context_length*context_length*d_model*num_layers
    dic["attention_softmax_v"] = 2*batch_size*context_length*d_model*context_length*num_layers
    dic["ffn"] = 3*2*batch_size*context_length*d_model*d_ff*num_layers
    dic["lm_head"] = 2*batch_size*context_length*vocab_size*d_model
    dic["total"] = sum(dic.values())
    return dic

def activation_elements(batch_size, context_length, vocab_size,
                        num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    dic["attention"] = 5*batch_size*context_length*d_model*num_layers #qkvo, qkv result
    dic["attention_T2"] = 2*batch_size*context_length*context_length*num_heads*num_layers# scores, softmax
    dic["ffn"] = 3*batch_size*context_length*d_ff*num_layers+ batch_size*context_length*d_model*num_layers #w1,w3, silu + w2
    dic["norms"] = batch_size*context_length*d_model*(2*num_layers+1)
    dic["logits"] = batch_size*context_length*vocab_size
    dic["total"] = sum(dic.values())
    return dic

# "medium": {
#             "d_model": 1024,
#             "d_ff": 4096,
#             "num_layers": 24,
#             "num_heads": 16
#         },
# context_lengt=512, back_size=4
def activation_elements_amp(batch_size, context_length, vocab_size,
                        num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    dic["attention"] = 5*batch_size*context_length*d_model*num_layers #qkvo, qkv result
    dic["attention_T2"] = 2*batch_size*context_length*context_length*num_heads*num_layers# scores, softmax
    dic["ffn"] = 3*batch_size*context_length*d_ff*num_layers+ batch_size*context_length*d_model*num_layers #w1,w3, silu + w2
    dic["norms"] = batch_size*context_length*d_model*(2*num_layers+1)
    dic["logits"] = batch_size*context_length*vocab_size
    dic["total"] = sum(dic.values())
    return dic

def adamw_peak_memory_bytes(batch_size, context_length, vocab_size,
                            num_layers, d_model, num_heads,
                            d_ff=None, bytes_per_element=4):
    dic = defaultdict(int)
    dic["params"] = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)["total"]*bytes_per_element
    dic["gradients"] = dic["params"]
    dic["optimizer"] = 2*dic["params"]
    dic["activations"] = activation_elements(batch_size, context_length, vocab_size, num_layers, d_model, num_heads, d_ff)["total"]*bytes_per_element
    dic["total"] = sum(dic.values())
    return dic

def adamw_step_flops(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    param = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)["total"]
    dic["weight_decay"] = 2*param
    dic["m_update"]=3*param
    dic["v_update"]=4*param
    dic["param_update"]=5*param
    dic["total"]=sum(dic.values())
    return dic