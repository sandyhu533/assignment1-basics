from collections import defaultdict

def round_d_ff(d_model: int) -> int:
    return (8/3*d_model//64)*64

def transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    if d_ff == None: d_ff = round_d_ff(d_model)
    dic = defaultdict(int)
    dic['token_embedding'] = vocab_size*d_model
    dic['attention'] = d_model*d_model*4*num_layers
    dic['ffn'] = d_model*d_ff*3*num_layers
    dic['norms'] = d_model*(2*num_layers+1)
    dic['lm_head'] = d_model*vocab_size
    dic['total'] = sum(dic.values())
    return dic

def transformer_forward_flops(batch_size, context_length, vocab_size,
                              num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    dic['attention_qkvo'] = 4*2*batch_size*context_length*d_model*d_model*num_layers
    dic['attention_scores'] = 2*batch_size*context_length*context_length*d_model*num_layers
    dic['attention_softmax_v'] = 2*batch_size*context_length*context_length*d_model*num_layers
    dic['ffn'] = 3*2*batch_size*context_length*d_model*d_ff*num_layers
    dic['lm_head'] = 2*batch_size*context_length*vocab_size*d_model
    dic['total'] = sum(dic.values())
    return dic

def activation_elements(batch_size, context_length, vocab_size,
                        num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    dic['attention'] = 5*batch_size*context_length*d_model*num_layers #qkvo + x result
    dic['attention_T2'] = 3*batch_size*context_length*context_length*num_heads*num_layers # qk_masked, xexp, attn
    dic['ffn'] = 5*batch_size*context_length*d_ff + batch_size*context_length*d_model # sigmoid, gate, w1x, w3x, gate*w3x ; x
    dic['ffn'] *= num_layers
    dic['norms'] = 2*batch_size*context_length*d_model*(num_layers*2+1)
    dic['lm_head'] = batch_size*context_length*d_model
    dic['ce'] = 2*batch_size*context_length*vocab_size
    dic['total'] = sum(dic.values())
    return dic

def adamw_peak_memory_bytes(batch_size, context_length, vocab_size,
                            num_layers, d_model, num_heads,
                            d_ff=None, bytes_per_element=4) -> dict:
    dic = defaultdict(int)
    dic['params'] = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)['total']*4
    dic['gradients'] = dic['params']
    dic['optimizer'] = 2*dic['params']
    dic['activations'] = activation_elements(batch_size, context_length, vocab_size, num_layers, d_model, num_heads, d_ff)['total']*4
    dic['total'] = sum(dic.values())
    return dic

def adamw_step_flops(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    params = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)['total']
    dic = defaultdict(int)
    dic['weight_decay'] = 2*params
    dic['m_update'] = 3*params
    dic['v_update'] = 4*params
    dic['param_update'] = 5*params
    dic['total'] = sum(dic.values())
    return dic