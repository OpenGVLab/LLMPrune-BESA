#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include "mask_gen.h"
#include <stdio.h>

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32
#define MAX_H 8

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t


template <typename scalar_t>
__global__ void MaskData_kernal(
  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> mask, // B, N1, 4, H, dim
  torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> index, //B, N1, K*4, H
  torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> top_k //B, N1, 4, K*4, H
  ){

  int n = blockIdx.x;
  int g = blockIdx.y;
  
  if(g<top_k[n])
    mask[n][index[n][g]] = 0;


}


std::vector<torch::Tensor> MaskData_ongpu(torch::Tensor mask, // B, N1, 4, H, dim
  torch::Tensor sorted_index, // B, N2, H, dim
  torch::Tensor top_k) // B, N1, K, 4, H
{

    const auto N = mask.size(0);
    const auto G = mask.size(1);
 

    //auto mask = torch::zeros({B, N1, 4, K, H},torch::device(torch::kCUDA));
    
   int shared_memory_per_block = 0;
    
    dim3 totalBlocks(N, G, 1);
    dim3 threadsPerBlock(THREADS_PER_WARP);
    AT_DISPATCH_FLOATING_TYPES(mask.type(), "MaskData_kernal", ([&] {
      MaskData_kernal<scalar_t><<<totalBlocks, threadsPerBlock, shared_memory_per_block * sizeof(scalar_t)>>>(
          mask.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          sorted_index.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
          top_k.packed_accessor32<long,1,torch::RestrictPtrTraits>());
    }));
  return {mask};

}
