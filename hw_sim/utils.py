import pickle

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .reorder import calc
from .ViTCoD_comp import sparse_linear_simulate, sparse_linear_flops


def get_model(model_name):
    def skip(*args, **kwargs):
        pass
    nn.init.kaiming_uniform_ = skip
    nn.init.uniform_ = skip
    nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    return model


def calc_sparsity(layer_weight):
    layer_params = layer_weight.numel()
    layer_pruned = layer_params - torch.count_nonzero(layer_weight)
    return layer_pruned / layer_params


def process_sparse_layer(model_name, layer_weight, layer_index, layer_name, threshold_ratio=0.5):
    print(f"Process layer-{layer_index}-{layer_name}")
    save_file_name = f"{model_name}_masks/{threshold_ratio}_layer_{layer_index}_{layer_name}.pkl"

    try:
        layer_info = pickle.load(open(save_file_name, 'rb'))
        cnt_d = layer_info['dense_ratio']
        cnt_e = layer_info['sparse_ratio']
        SpMM_PE_cycles = layer_info['SpMM_PE_cycles']
        print(f"Computation Cycles: {SpMM_PE_cycles}")

        if 'SpMM_FLOPs' not in layer_info:
            SpMM_FLOPs = sparse_linear_flops(layer_info['sparse_weight_mask'], layer_info['num_global_cols'])
            layer_info['SpMM_FLOPs'] = SpMM_FLOPs
            pickle.dump(layer_info, open(save_file_name, 'wb'))
        else:
            SpMM_FLOPs = layer_info['SpMM_FLOPs']
        print(f"Computation FLOPs: {SpMM_FLOPs}")
    except:
        bool_weight = layer_weight == 0
        bool_weight = bool_weight.numpy()
        threshold = threshold_ratio * bool_weight.shape[0]
        cnt_d, cnt_e, sparse_weight_mask, num_global_cols = calc(bool_weight, threshold)
        SpMM_PE_cycles = sparse_linear_simulate(sparse_weight_mask, num_global_cols)
        SpMM_FLOPs = sparse_linear_flops(sparse_weight_mask, num_global_cols)
        pickle.dump({
            'sparse_weight_mask': sparse_weight_mask,
            'num_global_cols': num_global_cols,
            'dense_ratio': cnt_d,
            'sparse_ratio': cnt_e,
            'SpMM_PE_cycles': SpMM_PE_cycles,
            'SpMM_FLOPs': SpMM_FLOPs,
            'layer_shape': [layer_weight.shape[0], layer_weight.shape[1]]
        }, open(save_file_name, 'wb'))

    return cnt_d, cnt_e, SpMM_PE_cycles, SpMM_FLOPs


def process_layer_loop(model_name, layer_weight, layer_index, layer_name, threshold_ratio=0.5):
    proj_ratio = threshold_ratio
    try_times = 0
    min_pe_cycles = float('inf')
    min_cycle_flops = float('inf')

    while True:
        D, E, pe_cycles, flops = process_sparse_layer(model_name, layer_weight, layer_index, layer_name, proj_ratio)
        if pe_cycles < min_pe_cycles:
            min_pe_cycles = pe_cycles
            min_cycle_flops = flops
        if D/E >= 0.6:
            proj_ratio += 0.1
            try_times += 1
        elif D/E <= 0.4:
            proj_ratio -= 0.1
            try_times += 1
        else:
            break

        if try_times == 5:
            break
        else:
            continue
    print(f'layer_{layer_index}_{layer_name}, Dense: {D} ({D/E * 100:.2f}%), Sparse: {E-D} ({(E-D)/E * 100:.2f}%), Total: {E}')

    return min_pe_cycles, min_cycle_flops


def process_patch_layer_loop(model_name, layer_weight, layer_index, layer_name, threshold_ratio=0.5):
    rows, cols = layer_weight.shape[0], layer_weight.shape[1]
    assert cols > rows
    patch_idx, total_pe_cycles, total_flops = 0, 0, 0
    for begin_idx in range(0, cols, rows):
        end_idx = min(cols, begin_idx + rows)
        patch_weight = layer_weight[:, begin_idx:end_idx]
        pe_cycles, flops = process_layer_loop(model_name, patch_weight, layer_index, f"{layer_name}_Patch-{patch_idx}", threshold_ratio)
        patch_idx += 1
        total_pe_cycles += pe_cycles
        total_flops += flops

    return total_pe_cycles, total_flops


def process_patch_dense_layer(layer_weight, layer_name):
    rows, cols = layer_weight.shape[0], layer_weight.shape[1]
    assert cols > rows
    patch_idx, total_pe_cycles, total_flops = 0, 0, 0
    for begin_idx in range(0, cols, rows):
        end_idx = min(cols, begin_idx + rows)
        patch_weight = layer_weight[:, begin_idx:end_idx]
        pe_cycles = sparse_linear_simulate(patch_weight, patch_weight.shape[1])
        flops = sparse_linear_flops(patch_weight, patch_weight.shape[1])
        patch_idx += 1
        total_pe_cycles += pe_cycles
        total_flops += flops
    print(f"{layer_name} dense cycles: {total_pe_cycles}")
    print(f"{layer_name} flops: {flops}")


def process_dense_layer(layer_weight, layer_name):
    SpMM_PE_cycles = sparse_linear_simulate(layer_weight, layer_weight.shape[1])
    SpMM_FLOPs = sparse_linear_flops(layer_weight, layer_weight.shape[1])
    print(f"{layer_name} dense cycles: {SpMM_PE_cycles}")
    print(f"{layer_name} flops: {SpMM_FLOPs}")