import torch
import torch.nn as nn

from mask_gen.functions.mask_gen import mask_gen


class MaskGenPytorch(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, sort_index, mask_shape, top_k):
        mask=torch.ones(mask_shape, device=sort_index.device)
        
        for i in range(mask_shape[0]):
            k=top_k[i]
            mask[i,sort_index[i,:k]]=0
        return mask


if __name__ == "__main__":
    import time
    rows, columns = 4096, 4096
    shape = (rows, columns)
    mask_gen_pytorch=MaskGenPytorch()

    row_blocksize = 8
    row_number = rows // row_blocksize
    row_sparsities = torch.rand(row_number).cuda()
    row_block_prune_num = (row_sparsities * columns).to(dtype=torch.long)
    row_prune_num = row_block_prune_num.reshape(-1, 1).repeat(1, row_blocksize).reshape(-1)

    mask_prob=torch.rand(shape).cuda()
    sort_index=torch.argsort(mask_prob, dim=1, descending=True)
    t=time.time()
    mask=mask_gen(sort_index, shape, row_prune_num)
    t2=time.time()
    mask2=mask_gen_pytorch(sort_index, shape, row_prune_num)
    t3=time.time()
    assert (mask==mask2).all()
    print('time for cuda op: ', t2-t)
    print('time for pytorch op: ', t3-t2)
