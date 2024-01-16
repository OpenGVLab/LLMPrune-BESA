import torch
import torch.nn as nn
from torch.autograd import Function
import mask_gen_cuda


class MaskGen(Function):
    @staticmethod
    def forward(ctx, sort_index, mask_shape, top_k):
        mask=torch.ones(mask_shape, device=sort_index.device)
        x = mask_gen_cuda.mask_gen_forward(mask, sort_index, top_k)
        return x[0]

    @staticmethod
    def backward(ctx, grad_output):
        return None


mask_gen_op = MaskGen.apply


class MaskGen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sort_index, mask_shape, top_k):
        mask=mask_gen_op(sort_index, mask_shape, top_k)
        return mask


mask_gen = MaskGen()