from multiprocessing import Process

from .utils import get_model, calc_sparsity
from .utils import process_layer_loop, process_patch_layer_loop
from .utils import process_dense_layer, process_patch_dense_layer


def check_sparsity(model_name, model=None):
    model = get_model(model_name) if model is None else model
    layers = model.model.layers
    q_sparsity, k_sparsity, v_sparsity, o_sparsity, gate_sparsity, up_sparsity, down_sparsity = [], [], [], [], [], [], []
    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        q_sparsity.append(calc_sparsity(layer.self_attn.q_proj.weight.data))
        k_sparsity.append(calc_sparsity(layer.self_attn.k_proj.weight.data))
        v_sparsity.append(calc_sparsity(layer.self_attn.v_proj.weight.data))
        o_sparsity.append(calc_sparsity(layer.self_attn.o_proj.weight.data))
        gate_sparsity.append(calc_sparsity(layer.mlp.gate_proj.weight.data))
        up_sparsity.append(calc_sparsity(layer.mlp.up_proj.weight.data))
        down_sparsity.append(calc_sparsity(layer.mlp.down_proj.weight.data))

    sparse_func = {
        "Min": lambda x : 100 * min(x),
        "Max": lambda x : 100 * max(x),
        "Average": lambda x : 100 * sum(x) / len(x)
    }
    for func_name in sparse_func:
        for layer_name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
            print(f"{func_name} sparsity for {layer_name}_proj: {sparse_func[func_name](eval(f'{layer_name}_sparsity'))}")


def check_proj(model_name, layer_name, threshold_ratio=0.5, model=None):
    model = get_model(model_name) if model is None else model
    layers = model.model.layers
    if layer_name in ['q', 'k', 'v', 'o']:
        block_name = 'self_attn'
    elif layer_name in ['gate', 'up', 'down']:
        block_name = 'mlp'
    else:
        raise ValueError(f"Invalid layer_name: {layer_name}")
    proc_func = process_patch_layer_loop if layer_name == 'down' else process_layer_loop

    total_cycles, total_flops = 0, 0
    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        pe_cycles, flops = proc_func(model_name, eval(f"layer.{block_name}.{layer_name}_proj.weight.data"), layer_index, f'{layer_name}_proj', threshold_ratio)
        total_cycles += pe_cycles
        total_flops += flops
    print(f"Avg {layer_name}_proj cycles: {total_cycles / len(layers)}")
    print(f"Avg {layer_name}_proj flops: {total_flops / len(layers)}")


def check_attn(model_name, threshold_ratio=0.5, model=None):
    model = get_model(model_name) if model is None else model
    for layer_name in ['q', 'k', 'v', 'o']:
        check_proj(model_name, layer_name, threshold_ratio, model)


def check_mlp(model_name, threshold_ratio=0.5, model=None):
    model = get_model(model_name) if model is None else model
    for layer_name in ['gate', 'up', 'down']:
        check_proj(model_name, layer_name, threshold_ratio, model)


def check_dense(model_name, model=None):
    model = get_model(model_name) if model is None else model
    layers = model.model.layers
    layer = layers[0]
    process_dense_layer(layer.self_attn.q_proj.weight.data, 'q_proj')
    process_dense_layer(layer.self_attn.k_proj.weight.data, 'k_proj')
    process_dense_layer(layer.self_attn.v_proj.weight.data, 'v_proj')
    process_dense_layer(layer.self_attn.o_proj.weight.data, 'o_proj')
    process_dense_layer(layer.mlp.gate_proj.weight.data, 'gate_proj')
    process_dense_layer(layer.mlp.up_proj.weight.data, 'up_proj')
    process_dense_layer(layer.mlp.down_proj.weight.data, 'down_proj')
    process_patch_dense_layer(layer.mlp.down_proj.weight.data, 'down_proj')


def check_model(model_name, threshold_ratio=0.5, model=None):
    model = get_model(model_name) if model is None else model
    layers = model.model.layers
    p_list = []

    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        p_list.append(Process(target=process_layer_loop, args=(model_name, layer.self_attn.q_proj.weight.data, layer_index, 'q_proj', threshold_ratio)))
        p_list.append(Process(target=process_layer_loop, args=(model_name, layer.self_attn.k_proj.weight.data, layer_index, 'k_proj', threshold_ratio)))
        p_list.append(Process(target=process_layer_loop, args=(model_name, layer.self_attn.v_proj.weight.data, layer_index, 'v_proj', threshold_ratio)))
        p_list.append(Process(target=process_layer_loop, args=(model_name, layer.self_attn.o_proj.weight.data, layer_index, 'o_proj', threshold_ratio)))
        p_list.append(Process(target=process_layer_loop, args=(model_name, layer.mlp.gate_proj.weight.data, layer_index, 'gate_proj', threshold_ratio)))
        p_list.append(Process(target=process_layer_loop, args=(model_name, layer.mlp.up_proj.weight.data, layer_index, 'up_proj', threshold_ratio)))
        p_list.append(Process(target=process_patch_layer_loop, args=(model_name, layer.mlp.down_proj.weight.data, layer_index, 'down_proj', threshold_ratio)))

    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
