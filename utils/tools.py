import os
import random
import datetime
import builtins
import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import fnmatch
from lm_eval import tasks, evaluator
from accelerate import dispatch_model
from accelerate.utils import get_balanced_memory, infer_auto_device_map


def slurm_dist_init(seed=0, port=1999):
    mp.set_start_method('spawn', force=True)

    rank = int(os.environ['SLURM_PROCID'])
    world_size = os.environ['SLURM_NTASKS']
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')

    os.environ['MASTER_ADDR'] = addr
    # os.environ['MASTER_PORT'] = str(port)
    os.environ['WORLD_SIZE'] = world_size
    os.environ['RANK'] = str(rank)

    for _ in range(10):
        try:
            os.environ['MASTER_PORT'] = str(port)
            dist.init_process_group(backend='nccl')
            break
        except:
            port += 99
            continue   

    # dist.init_process_group(backend='nccl')
    torch.cuda.set_device(gpu_id)
    dist.barrier()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master_process():
    return get_rank() == 0


def setup_distributed_print(is_master_process):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        if is_master_process:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print


def use_old_forward(module: nn.Module, recurse=False):
    if hasattr(module, '_old_forward'):
        module._new_forward = module.forward
        module.forward = module._old_forward
    
    if recurse:
        for child in module.children():
            use_old_forward(child, recurse)


def use_new_forward(module: nn.Module, recurse=False):
    if hasattr(module, '_new_forward'):
        module.forward = module._new_forward
        delattr(module, "_new_forward")
    
    if recurse:
        for child in module.children():
            use_new_forward(child, recurse)


def auto_map_model(model):
    print(f"Check no split modules: {model.model._no_split_modules}")
    max_memory = get_balanced_memory(model.model, dtype=torch.float16, no_split_module_classes=model.model._no_split_modules)
    print(f"Check max memory: {max_memory}")
    model.model.tie_weights()
    print("Model weights tied")
    device_map = infer_auto_device_map(model.model, dtype=torch.float16, max_memory=max_memory, no_split_module_classes=model.model._no_split_modules)
    print(f"Check device map: {device_map}")
    dispatch_model(model.model, device_map)


class FakeScaler:
    def __call__(self, loss, optimizer, parameters=None, clip_grad=None, clip_mode=None):
        loss.backward()
        optimizer.step()


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def lm_eval_model(model, args):
    if args.test_datasets is None:
        test_datasets = args.dataset
    else:
        test_datasets = pattern_match(args.test_datasets.split(","), tasks.ALL_TASKS)
    if test_datasets == []:
        return "No test dataset specified"

    return evaluator.simple_evaluate(
        model=model,
        tasks=test_datasets,
        batch_size=args.batch_size,
        device=model.device,
        no_cache=True
    )


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def init_learn_sparsity(sparse_layers, sparsity_step=0.01, prune_n=0, prune_m=0, blocksize=-1, sigmoid_smooth=False, lora_rank=-1, lora_alpha=1):
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparse_layer.init_learn_sparsity(sparsity_step, prune_n, prune_m, blocksize, sigmoid_smooth, lora_rank, lora_alpha)


def finish_learn_sparsity(sparse_layers):
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparse_layer.finish_learn_sparsity()


def get_sparsity(sparse_layers):
    total_param = sum([sparse_layers[layer_name].param_num for layer_name in sparse_layers])
    sparsity = 0
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        sparsity += sparse_layer.sparsity * (sparse_layer.param_num / total_param)
    return sparsity


def get_sparsity_params(sparse_layers):
    params = []
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        if sparse_layer.sparsity_probabilities is not None:
            layer_sparsity_params = sparse_layer.sparsity_probabilities
            if type(layer_sparsity_params) is list:
                params.extend(layer_sparsity_params)
            else:
                params.append(layer_sparsity_params)
    return params


def get_lora_params(sparse_layers):
    params = []
    for layer_name in sparse_layers:
        sparse_layer = sparse_layers[layer_name]
        if sparse_layer.use_lora:
            params.append(sparse_layer.lora_A)
            params.append(sparse_layer.lora_B)
    return params


def eval_ppl(model, testenc, batch_size=1, device='cuda'):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    neg_log_likelihoods = []
    for i in range(0, nsamples, batch_size):
        j = min(i + batch_size, nsamples)
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model._model_call(inputs)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j - i)
        neg_log_likelihoods.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(neg_log_likelihoods).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()

    return ppl.item()
