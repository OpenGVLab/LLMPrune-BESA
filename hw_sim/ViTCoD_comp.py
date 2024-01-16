import math
import numpy as np
from scipy.sparse import coo_matrix

from .PE_array import PE_array


def sparse_linear_simulate(sparse_weight_mask, num_global_cols, PE_width=64, PE_height=64):
    my_PE = PE_array(PE_width, PE_height)
    inp = np.random.random((sparse_weight_mask.shape[1], 1))

    mask = sparse_weight_mask
    global_tokens = int(num_global_cols)
    sparser = coo_matrix(1 - mask[:, global_tokens:])
    sparser = np.column_stack((sparser.row, sparser.col))
    dense_ratio = global_tokens * sparse_weight_mask.shape[0] / (len(sparser) + global_tokens * sparse_weight_mask.shape[0])
    dense_PE_width = int(my_PE.width * dense_ratio)
    sparse_PE_width = my_PE.width - dense_PE_width

    # ############## dense pattern weight * inp ##############
    dense_SpMM_PE_cycles = 0
    if dense_PE_width > 0:
        for _ in range(math.ceil((inp.shape[0] * inp.shape[1] * global_tokens) / (dense_PE_width * my_PE.height))):
            dense_SpMM_PE_cycles += 1
    print('Dense SpMM PE caclulation | cycles: {}'.format(dense_SpMM_PE_cycles))

    # ############## sparse pattern weight * inp ##############
    # acumulation
    num_list = []
    accumulator = 0
    prev_cout_index = 0
    for _cout_index, _cin_index in sparser:
        if _cout_index == prev_cout_index:
            accumulator += 1
        else:
            num_list.append(accumulator)
            accumulator = 1
        prev_cout_index = _cout_index
    num_list.append(accumulator)

    # ############## sparse pattern weight * inp ##############
    sparse_SpMM_PE_cycles = 0
    for row_num in num_list:
        sparse_SpMM_PE_cycles += row_num * inp.shape[1]
    if sparse_PE_width > 0:
        sparse_SpMM_PE_cycles = math.ceil(sparse_SpMM_PE_cycles / (sparse_PE_width * my_PE.height))
    print('Sparse SpMM PE caclulation | cycles: {}'.format(sparse_SpMM_PE_cycles))  

    SpMM_PE_cycles = max(sparse_SpMM_PE_cycles, dense_SpMM_PE_cycles)
    print(f"Computation Cycles: {SpMM_PE_cycles}")
    return SpMM_PE_cycles


def sparse_linear_flops(sparse_weight_mask, num_global_cols):
    inp = np.random.random((sparse_weight_mask.shape[1], 1))

    mask = sparse_weight_mask
    global_tokens = int(num_global_cols)
    sparser = coo_matrix(1 - mask[:, global_tokens:])
    sparser = np.column_stack((sparser.row, sparser.col))
    dense_flops = global_tokens * sparse_weight_mask.shape[0]

    # ############## sparse pattern weight * inp ##############
    num_list = []
    accumulator = 0
    prev_cout_index = 0
    for _cout_index, _cin_index in sparser:
        if _cout_index == prev_cout_index:
            accumulator += 1
        else:
            num_list.append(accumulator)
            accumulator = 1
        prev_cout_index = _cout_index
    num_list.append(accumulator)

    sparse_flops = 0
    for row_num in num_list:
        sparse_flops += row_num * inp.shape[1]

    total_flops = dense_flops + sparse_flops
    print(f"Computation FLOPs: {total_flops}")
    return total_flops
