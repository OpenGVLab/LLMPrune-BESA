import math

import torch
import torch.nn as nn
import mask_gen_cuda


class SparseLinear(nn.Module):
    def __init__(self, layer, metric_type='Wanda', wise_dim='row') -> None:
        super().__init__()
        self.layer = layer
        self.linear_func = nn.functional.linear
        self.register_buffer('weight', layer.weight)
        if layer.bias is not None:
            self.register_buffer('bias', layer.bias)
        else:
            self.bias = None
        self.param_num = self.weight.numel()

        self.nsamples = 0
        self.use_lora = False
        self.learn_sparsity = False
        self.rows = self.weight.data.shape[0]
        self.columns = self.weight.data.shape[1]
        self.device = self.weight.device

        self.wise_dim = wise_dim
        assert self.wise_dim in ['row', 'column'], f"Invalid wise dim: {wise_dim}"

        self.metric_type = metric_type
        if metric_type == 'Wanda':
            self.scaler_row = torch.zeros((self.columns), device=self.device)
        elif metric_type == 'SparseGPT' or metric_type == 'SparseGPT-Git':
            self.Hessian = torch.zeros((self.columns, self.columns), device=self.device)
        elif metric_type == 'Weight':
            pass
        else:
            raise NotImplementedError(f"Invalid metric type: {metric_type}")
    
    def add_batch(self, inp, out):
        if self.metric_type == 'Weight':
            return

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        if self.metric_type == 'Wanda':
            self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        elif self.metric_type == 'SparseGPT' or self.metric_type == 'SparseGPT-Git':
            self.Hessian *= self.nsamples / (self.nsamples + tmp)
        else:
            raise NotImplementedError(f"Invalid metric type: {self.metric_type}")

        self.nsamples += tmp

        if self.metric_type == 'Wanda':
            self.scaler_row += torch.norm(inp.float(), p=2, dim=1) ** 2  / self.nsamples
        elif self.metric_type == 'SparseGPT' or self.metric_type == 'SparseGPT-Git':
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.Hessian += inp.matmul(inp.t())
        else:
            raise NotImplementedError(f"Invalid metric type: {self.metric_type}")
    
    def get_w_metric(self):
        # NOTE: in lora pruning, the importance metric will be: lora_B @ grad(lora_A) + grad(lora_B) @ lora_A - grad(lora_B) @ grad(lora_A)
        if self.metric_type == 'Weight':
            self.W_metric = torch.abs(self.weight)
        elif self.metric_type == 'Wanda':
            self.W_metric = torch.abs(self.weight) * torch.sqrt(self.scaler_row.reshape((1,-1)))
            self.scaler_row = None
        elif self.metric_type == 'SparseGPT' or self.metric_type == 'SparseGPT-Git':
            percdamp = 0.01 # Percent of the average Hessian diagonal used for dampening
            dead = torch.diag(self.Hessian) == 0
            self.Hessian[dead, dead] = 1
            # NOTE: SparseGPT updates the weight in cols of zero diag
            self.weight.data[:, dead] = 0
            damp = percdamp * torch.mean(torch.diag(self.Hessian))
            diag = torch.arange(self.columns, device=self.device)
            self.Hessian[diag, diag] += damp
            Hinv = torch.linalg.cholesky(self.Hessian)
            Hinv = torch.cholesky_inverse(Hinv)
            # NOTE: use cholesky_ex as Hinv many not be complex Hermitian matrix
            Hinv, _ = torch.linalg.cholesky_ex(Hinv, upper=True)
            Hinv = torch.diag(Hinv).reshape((1, -1))
            if self.metric_type == 'SparseGPT-Git':
                self.W_metric = self.weight ** 2 / Hinv ** 2
            else:
                self.W_metric = self.weight ** 2 / Hinv

        if self.wise_dim == 'row':
            self.sort_indices = torch.sort(self.W_metric, dim=-1, stable=True)[1]
        elif self.wise_dim == 'column':
            self.sort_indices = torch.sort(self.W_metric, dim=0, stable=True)[1]
        else:
            raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")

    def init_learn_sparsity(self, sparsity_step=0.01, prune_n=0, prune_m=0, blocksize=-1, sigmoid_smooth=False, lora_rank=-1, lora_alpha=1):
        self.prune_n, self.prune_m = prune_n, prune_m
        self.get_w_metric()
        torch.cuda.empty_cache()

        if hasattr(self, 'sparsity'):
            self.block_wise = False
            self.learn_sparsity = False
            W_mask = self.get_weight_mask().detach()
            self.weight.data *= W_mask.to(dtype=self.weight.dtype)
            self.finish_learn_sparsity()
            return

        self.learn_sparsity = True
        self.block_wise = blocksize != -1
        self.sigmoid_smooth = sigmoid_smooth
        self.sparsity_candidates = torch.arange(1.0, -1 * sparsity_step, -1 * sparsity_step, device=self.device)
        self.sparsity_candidates[-1] = 0.0
        if self.block_wise:
            self.blocksize = blocksize
            if self.wise_dim == 'row':
                assert self.rows % blocksize == 0, "Row blocksize should be fully divided by the number of rows"
                self.blocknum = self.rows // blocksize
            elif self.wise_dim == 'column':
                assert self.columns % blocksize == 0, "Column blocksize should be fully divided by the number of rows"
                self.blocknum = self.columns // blocksize
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
            self.sparsity_probabilities = nn.Parameter(torch.zeros((self.blocknum, self.sparsity_candidates.shape[0]), device=self.device))
        else:
            self.sparsity_probabilities = nn.Parameter(torch.zeros_like(self.sparsity_candidates, device=self.device))
        self.update_sparsity()

        map_dim_size = self.columns if self.wise_dim == 'row' else self.rows if self.wise_dim == 'column' else -1
        self.prob_map_matrix = torch.zeros((len(self.sparsity_candidates), map_dim_size), device=self.device)
        for i in range(len(self.sparsity_candidates)):
            self.prob_map_matrix[i, :int(map_dim_size * self.sparsity_candidates[i].item())] = 1

        self.use_lora = lora_rank != -1
        if self.use_lora:
            assert type(lora_rank) is int and 0 < lora_rank < min(self.rows, self.columns), f"Invalid Lora rank: {lora_rank}"
            self.lora_A = nn.Parameter(torch.zeros((lora_rank, self.columns), device=self.device))
            self.lora_B = nn.Parameter(torch.zeros((self.rows, lora_rank), device=self.device))
            self.lora_scaling = lora_alpha / lora_rank

    def finish_learn_sparsity(self):
        if self.learn_sparsity:
            if self.use_lora:
                lora_weight = (self.lora_B.data @ self.lora_A.data).detach() * self.lora_scaling
                self.weight.data += lora_weight.to(self.weight.dtype)
                self.lora_A = None
                self.lora_B = None
                self.lora_scaling = None
            self.update_sparsity()
            prune_mask = self.get_prune_mask().detach()
            self.weight.data *= prune_mask
        self.learn_sparsity = False

        self.W_metric = None
        self.scaler_row = None
        self.sort_indices = None
        self.sparsities = None
        self.prob_map_matrix = None
        self.sparsity_candidates = None
        self.sparsity_probabilities = None
        self.sparsity_probabilities_softmax = None
        torch.cuda.empty_cache()

    def update_sparsity(self):
        if self.sigmoid_smooth:
            self.sparsity_probabilities_softmax = self.sparsity_probabilities.sigmoid().softmax(dim=-1)
        else:
            self.sparsity_probabilities_softmax = self.sparsity_probabilities.softmax(dim=-1)

        if self.block_wise:
            self.sparsities = self.sparsity_probabilities_softmax @ self.sparsity_candidates
            self.sparsity = self.sparsities.mean()
        else:
            self.sparsity = torch.matmul(self.sparsity_candidates, self.sparsity_probabilities_softmax)
        return self.sparsity

    def get_weight_mask(self):
        W_mask = torch.ones((self.rows, self.columns), device=self.device)
        if self.prune_n != 0:
            # structured n:m sparsity
            for ii in range(self.columns):
                if ii % self.prune_m == 0:
                    tmp = self.W_metric[:, ii:(ii + self.prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], 0)
        elif self.block_wise:
            # block wise unstructured pruning
            if self.wise_dim == 'row':
                row_block_prune_num = (self.sparsities * self.columns).to(dtype=torch.long)
                row_prune_num = row_block_prune_num.reshape((-1, 1)).repeat(1, self.blocksize).reshape(-1)
                W_mask = mask_gen_cuda.mask_gen_forward(W_mask, self.sort_indices, row_prune_num)[0]
            elif self.wise_dim == 'column':
                column_block_prune_num = (self.sparsities * self.rows).to(dtype=torch.long)
                column_prune_num = column_block_prune_num.reshape((-1, 1)).repeat(1, self.blocksize).reshape(-1)
                W_mask = mask_gen_cuda.mask_gen_forward(W_mask.t().contiguous(), self.sort_indices.t().contiguous(), column_prune_num)[0]
                W_mask = W_mask.t().contiguous()
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
        else:
            # unstructured pruning
            if self.wise_dim == 'row':
                indices = self.sort_indices[:, :int(self.columns * self.sparsity)]
                W_mask.scatter_(1, indices, 0)
            elif self.wise_dim == 'column':
                indices = self.sort_indices[:int(self.rows * self.sparsity), :]
                W_mask.scatter_(0, indices, 0)
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
        return W_mask

    def get_prob_mask(self):
        P_mask = torch.zeros((self.rows, self.columns), device=self.device)
        probabilities = 1 - (self.sparsity_probabilities_softmax @ self.prob_map_matrix)
        if not self.block_wise:
            if self.wise_dim == 'row':
                probabilities = probabilities.repeat(self.rows, 1)
            elif self.wise_dim == 'column':
                probabilities = probabilities.reshape((-1, 1)).repeat(1, self.columns)
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
        else:
            if self.wise_dim == 'row':
                probabilities = probabilities.reshape((self.blocknum, 1, self.columns))
                probabilities = probabilities.repeat(1, self.blocksize, 1)
            elif self.wise_dim == 'column':
                probabilities = probabilities.reshape((self.rows, self.blocknum, 1))
                probabilities = probabilities.repeat(1, 1, self.blocksize)
            else:
                raise NotImplementedError(f"Invalid wise dim: {self.wise_dim}")
            probabilities = probabilities.reshape((self.rows, self.columns))
        probabilities = probabilities.to(dtype=P_mask.dtype)
        scatter_dim = 1 if self.wise_dim == 'row' else 0 if self.wise_dim == 'column' else -1
        P_mask.scatter_(scatter_dim, self.sort_indices, probabilities)
        return P_mask

    def get_prune_mask(self):
        W_mask = self.get_weight_mask()
        P_mask = self.get_prob_mask()
        prune_mask = W_mask.detach() - P_mask.detach() + P_mask
        prune_mask = prune_mask.to(dtype=self.weight.dtype)

        return prune_mask
    
    def forward(self, input: torch.Tensor):
        weight = self.weight.detach()
        if self.learn_sparsity:
            self.update_sparsity()
            prune_mask = self.get_prune_mask()
            if self.use_lora:
                lora_weight = (self.lora_B @ self.lora_A) * self.lora_scaling
                weight += lora_weight.to(dtype=self.weight.dtype)
            weight = torch.mul(weight, prune_mask)
        out = self.linear_func(input, weight, self.bias)

        return out
