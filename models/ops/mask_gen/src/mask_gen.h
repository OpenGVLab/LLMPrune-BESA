#ifndef _Score_CUDA
#define _Score_CUDA
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> mask_cuda_forward(torch::Tensor mask,         // query t: N, H, W, C1
                                             torch::Tensor sorted_index, // scene : N, H, W, C1
                                             torch::Tensor top_k);       // scene : N, H, W, C1

std::vector<at::Tensor> MaskData_ongpu(at::Tensor mask,         // query t: N, H, W, C1
                                       at::Tensor sorted_index, // scene : N, H, W, C1
                                       at::Tensor top_k);       // scene : N, H, W, C1

#endif