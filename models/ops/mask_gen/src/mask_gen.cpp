#include "mask_gen.h"
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// == Forward
std::vector<torch::Tensor> mask_cuda_forward(torch::Tensor mask,         // parameter: K*group_num, C
                                             torch::Tensor sorted_index, // tensor : B, N, C
                                             torch::Tensor top_k)        // tensor: B, N, K
{
  CHECK_INPUT(mask);
  CHECK_INPUT(sorted_index);
  CHECK_INPUT(top_k);

  return MaskData_ongpu(mask, sorted_index, top_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("mask_gen_forward", &mask_cuda_forward, "score forward (CUDA)");
}