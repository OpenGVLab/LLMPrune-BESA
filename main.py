import os
import gc
import time
import uuid
import pickle
import argparse
import contextlib

import torch
import torch.nn as nn

from timm.utils import NativeScaler

from optimizers import Prodigy
from models.llama import LLaMA
from models.llama_seq import LLaMA_Seq
from models.sparse_layers import SparseLinear
from utils.data import get_loaders
from utils.tools import slurm_dist_init, is_master_process, lm_eval_model, find_layers, init_learn_sparsity, finish_learn_sparsity, get_sparsity, get_sparsity_params, eval_ppl, FakeScaler, use_old_forward, use_new_forward, get_lora_params, auto_map_model

USE_WANDB = False
if USE_WANDB:
    import wandb
else:
    class Wandb:
        def __init__(self): pass

        def login(self): pass

        def init(self, **kwargs): pass

        def finish(self): pass

        def log(self, log_info):
            print(f"wandb: {log_info}")
    
    wandb = Wandb()


def get_model(model_name, batch_size=1):
    def skip(*args, **kwargs):
        pass
    nn.init.kaiming_uniform_ = skip
    nn.init.uniform_ = skip
    nn.init.normal_ = skip

    if 'llama' in model_name.lower():
        if USE_LLaMA_SEQ:
            model = LLaMA_Seq(model_name, batch_size=batch_size)
        else:
            model = LLaMA(model_name, batch_size=batch_size)
    else:
        raise NotImplementedError(f"Invalid model name: {model_name}")
    return model


def loss_func(l2_loss, sparsity):
    loss = args.l2_alpha * l2_loss + args.sparsity_beta * ((sparsity - args.sparsity) / args.sparsity) ** 2
    return loss


def val_epoch(layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs, refer_dense=False):
    refer_outs = dense_outs if refer_dense else outs
    with torch.no_grad():
        loss_list, l2_loss_list, dense_l2_loss_list = [], [], []
        sparsity = float(get_sparsity(sparse_layers))
        if args.norm_all:
            l2_scaler = torch.norm(refer_outs.type(torch.float32).reshape((-1, refer_outs.shape[-1])).t(), p=2, dim=1)

        for begin_idx in range(0, args.nsamples, args.prune_batch_size):
            end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
            with inference_context:
                pruned_outs[begin_idx: end_idx,] = layer(inps[begin_idx: end_idx,], attention_mask, position_ids, end_idx - begin_idx)[0]
                if not args.norm_all:
                    l2_scaler = torch.norm(refer_outs[begin_idx: end_idx,].type(torch.float32).reshape((-1, refer_outs[begin_idx: end_idx,].shape[-1])).t(), p=2, dim=1).detach()
                l2_loss = (((refer_outs[begin_idx: end_idx,] - pruned_outs[begin_idx: end_idx,]) / l2_scaler) ** 2).sum() / pruned_outs[begin_idx: end_idx,].shape[-1]
                loss = loss_func(l2_loss, sparsity)
                if not args.no_dense_loss:
                    dense_l2_loss = ((dense_outs[begin_idx: end_idx,] - pruned_outs[begin_idx: end_idx,]) ** 2).sum() / pruned_outs[begin_idx: end_idx,].numel()
                    dense_l2_loss_list.append(dense_l2_loss.item())
            loss_list.append(float(loss))
            l2_loss_list.append(l2_loss.item())
    val_loss = sum(loss_list) / len(loss_list)
    val_l2_loss = sum(l2_loss_list) / len(l2_loss_list)
    return sparsity, val_loss, val_l2_loss, dense_l2_loss_list


def train_epoch(layer, sparse_layers, attention_mask, position_ids, inps, refer_outs, optimizer, loss_scaler, train_params):
    l2_loss_list, loss_list = [], []
    if args.norm_all:
        l2_scaler = torch.norm(refer_outs.type(torch.float32).reshape((-1, refer_outs.shape[-1])).t(), p=2, dim=1).detach()

    for begin_idx in range(0, args.nsamples, args.prune_batch_size):
        end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
        with inference_context:
            pruned_out = layer(inps[begin_idx: end_idx,], attention_mask, position_ids, end_idx - begin_idx)[0]
            sparsity = get_sparsity(sparse_layers)
            if not args.norm_all:
                l2_scaler = torch.norm(refer_outs[begin_idx: end_idx,].type(torch.float32).reshape((-1, refer_outs[begin_idx: end_idx,].shape[-1])).t(), p=2, dim=1).detach()
            l2_loss = (((refer_outs[begin_idx: end_idx,] - pruned_out) / l2_scaler) ** 2).sum() / refer_outs[begin_idx: end_idx,].shape[-1]
            loss = loss_func(l2_loss, sparsity)
            loss_list.append(loss.item())
            l2_loss_list.append(l2_loss.item())
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, parameters=train_params, clip_grad=args.clip_grad, clip_mode=args.clip_mode)
        torch.cuda.empty_cache()
    train_loss = sum(loss_list) / len(loss_list)
    train_l2_loss = sum(l2_loss_list) / len(l2_loss_list)

    return train_loss, train_l2_loss


def grad_prune(layer_index, layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs):
    print(f"Grad prune layer {layer_index}")
    sparsity_params = get_sparsity_params(sparse_layers)
    lora_params = get_lora_params(sparse_layers)
    if len(lora_params) > 0:
        param_lr = args.prodigy_lr if not args.normal_opt else 1e-3 if args.normal_default else args.normal_opt_lr
        compress_params = [
            {'params': sparsity_params, 'lr': param_lr},
            {'params': lora_params, 'lr': param_lr},
        ]
        train_params = sparsity_params + lora_params
    else:
        compress_params = train_params = sparsity_params
    loss_scaler = FakeScaler() if args.no_scaler else NativeScaler()

    if args.normal_opt:
        if args.normal_default:
            optimizer = torch.optim.AdamW(compress_params)
        else:
            optimizer = torch.optim.AdamW(compress_params, lr=args.normal_opt_lr, weight_decay=0)
    else:
        optimizer = Prodigy(compress_params, args.prodigy_lr, 
                            weight_decay=args.weight_decay,
                            decouple=not args.no_decouple,
                            use_bias_correction=args.use_bias_correction,
                            safeguard_warmup=args.safeguard_warmup,
                            d_coef=args.d_coef
                        )

    if args.use_cos_sche:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   
    # learn sparsity epochs
    refer_outs = dense_outs if args.prune_dense else outs
    for epoch in range(args.epochs):
        # train epoch
        train_loss, train_l2_loss = train_epoch(layer, sparse_layers, attention_mask, position_ids, inps, refer_outs, optimizer, loss_scaler, train_params)
        if args.use_cos_sche:
            lr_scheduler.step(epoch)
        torch.cuda.empty_cache()

        # val epoch
        sparsity, val_loss, val_l2_loss, dense_l2_loss_list = val_epoch(layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs, args.prune_dense)

        wandb_log = {
            f'layer_{layer_index}-train_loss': train_loss,
            f'layer_{layer_index}-train_l2_loss': train_l2_loss,
            f'layer_{layer_index}-sparsity': sparsity,
            f'layer_{layer_index}-val_loss': val_loss,
            f'layer_{layer_index}-val_l2_loss': val_l2_loss,
        }
        if not args.no_dense_loss:
            dense_val_l2_loss = sum(dense_l2_loss_list) / len(dense_l2_loss_list)
            wandb_log[f'layer_{layer_index}-dense_val_l2_loss'] = dense_val_l2_loss
        for layer_name in sparse_layers:
            sparse_layer = sparse_layers[layer_name]
            wandb_log[f"layer_{layer_index}-{layer_name}_sparsity"] = float(sparse_layer.sparsity)
        wandb.log(wandb_log)

    return wandb_log, sparsity


def fixed_prune(layer_index, layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs):
    print(f"Fixed prune layer {layer_index}")
    sparsity, val_loss, val_l2_loss, dense_l2_loss_list = val_epoch(layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs, args.prune_dense)
    wandb_log = {
        f'layer_{layer_index}-val_loss': val_loss,
        f'layer_{layer_index}-val_l2_loss': val_l2_loss,
        f'layer_{layer_index}-sparsity': args.sparsity,
    }
    if not args.no_dense_loss:
        dense_val_l2_loss = val_l2_loss if args.prune_dense else sum(dense_l2_loss_list) / len(dense_l2_loss_list)
        wandb_log[f'layer_{layer_index}-dense_val_l2_loss'] = dense_val_l2_loss
    wandb.log(wandb_log)

    return wandb_log, sparsity


def compress_model(model, dataloader):
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    def add_batch(layer_name):
        def tmp(_, inp, out):
            sparse_layers[layer_name].add_batch(inp[0].data, out.data)
        return tmp

    print('Starting ...')
    prune_start = time.time()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.model.layers
    model.model.model.embed_tokens = model.model.model.embed_tokens.to(dev)

    dtype = next(iter(model.model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), device=dev, dtype=dtype
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    for i in range(args.nsamples):
        try:
            batch = dataloader[i]
            model.model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers = layers.cpu()
    model.model.model.embed_tokens = model.model.model.embed_tokens.cpu()

    position_ids = cache['position_ids']
    attention_mask = cache['attention_mask']
    if args.use_fp32:
        inps = inps.float()
        attention_mask = attention_mask.float()
        dtype = torch.float32
    torch.cuda.empty_cache()

    pruned_outs = torch.zeros_like(inps)
    if args.prune_dense or (not args.no_dense_loss):
        dense_inps = inps.clone()
        dense_outs = torch.zeros_like(inps)
    else:
        dense_outs = None
    outs = None if args.prune_dense else torch.zeros_like(inps)

    print('Ready.')
    model_prune_log, model_sparsity = [], []
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        use_old_forward(layer, recurse=True)
        if args.use_fp32:
            layer = layer.float()

        layer.self_attn.q_proj = SparseLinear(layer.self_attn.q_proj, args.metric_type, args.wise_dim)
        layer.self_attn.k_proj = SparseLinear(layer.self_attn.k_proj, args.metric_type, args.wise_dim)
        layer.self_attn.v_proj = SparseLinear(layer.self_attn.v_proj, args.metric_type, args.wise_dim)
        if 'llama' in args.model.lower():
            layer.self_attn.o_proj = SparseLinear(layer.self_attn.o_proj, args.metric_type, args.wise_dim)
            layer.mlp.gate_proj = SparseLinear(layer.mlp.gate_proj, args.metric_type, args.wise_dim)
            layer.mlp.up_proj = SparseLinear(layer.mlp.up_proj, args.metric_type, args.wise_dim)
            layer.mlp.down_proj = SparseLinear(layer.mlp.down_proj, args.metric_type, args.wise_dim)
        elif 'opt' in args.model.lower():
            layer.self_attn.out_proj = SparseLinear(layer.self_attn.out_proj, args.metric_type, args.wise_dim)
            layer.fc1 = SparseLinear(layer.fc1, args.metric_type, args.wise_dim)
            layer.fc2 = SparseLinear(layer.fc2, args.metric_type, args.wise_dim)

        handles = []
        sparse_layers = find_layers(layer, layers=[SparseLinear])
        for layer_name in sparse_layers:
            sparse_layer = sparse_layers[layer_name]
            handles.append(sparse_layer.register_forward_hook(add_batch(layer_name)))
        with inference_context:
            refer_outs = pruned_outs if outs is None else outs
            for begin_idx in range(0, args.nsamples, args.prune_batch_size):
                end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
                refer_outs[begin_idx: end_idx,] = layer(inps[begin_idx: end_idx,], attention_mask, position_ids, end_idx - begin_idx)[0]
                torch.cuda.empty_cache()
        for h in handles:
            h.remove()

        if args.prune_dense or (not args.no_dense_loss):
            with inference_context:
                for begin_idx in range(0, args.nsamples, args.prune_batch_size):
                    end_idx = min(args.nsamples, begin_idx + args.prune_batch_size)
                    dense_outs[begin_idx: end_idx,] = layer(dense_inps[begin_idx: end_idx,], attention_mask, position_ids, end_idx - begin_idx)[0]
                    torch.cuda.empty_cache()

        prune_func = grad_prune
        if args.fix_layers:
            fix_layers = list(sparse_layers.keys()) if args.fix_layers == 'all' else args.fix_layers.split(',')
            prune_func = fixed_prune if args.fix_layers == 'all' else grad_prune
            for layer_name in fix_layers:
                sparse_layers[layer_name].sparsity = args.sparsity

        torch.set_grad_enabled(True)
        init_learn_sparsity(sparse_layers, args.sparsity_step, blocksize=args.blocksize, sigmoid_smooth=not args.no_sigmoid_smooth, lora_rank=args.lora_rank)
        layer_prune_log, layer_sparsity = prune_func(i, layer, sparse_layers, attention_mask, position_ids, inps, outs, pruned_outs, dense_outs)
        torch.set_grad_enabled(False)
        finish_learn_sparsity(sparse_layers)
        model_prune_log.append(layer_prune_log)
        model_sparsity.append(layer_sparsity)

        layer.self_attn.q_proj = layer.self_attn.q_proj.layer
        layer.self_attn.k_proj = layer.self_attn.k_proj.layer
        layer.self_attn.v_proj = layer.self_attn.v_proj.layer
        layer.self_attn.o_proj = layer.self_attn.o_proj.layer
        layer.mlp.gate_proj = layer.mlp.gate_proj.layer
        layer.mlp.up_proj = layer.mlp.up_proj.layer
        layer.mlp.down_proj = layer.mlp.down_proj.layer

        layer = layer.cpu().to(dtype=dtype)
        use_new_forward(layer, recurse=True)
        layers[i] = layer
        del layer
        del sparse_layers
        gc.collect()
        torch.cuda.empty_cache()
        inps, pruned_outs = pruned_outs, inps
        if args.prune_dense or (not args.no_dense_loss):
            dense_inps, dense_outs = dense_outs, dense_inps

    model.config.use_cache = use_cache
    prune_time_cost = time.time() - prune_start
    print(f'Prune time cost: {prune_time_cost:.3f} seconds')
    model_sparsity = sum(model_sparsity) / len(model_sparsity)
    print(f"Model sparsity: {model_sparsity:.2f}")

    return model_prune_log


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='/mnt/lustre/share_data/xupeng/llama-7b-hf',
        help='model to load.'
    )
    parser.add_argument(
        '--test-datasets', type=str, default='piqa,boolq,hellaswag,winogrande,arc_easy,arc_challenge',
        help='Evaluate model on test datasets'
    )
    parser.add_argument(
        '--eval-dense', action='store_true',
        help='Whether to evaluate the dense model'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='batch size of model evaluation'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--port', type=int, default=1999,
        help='Port to init torch distributed.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0.5,
        help='Target sparsity'
    )

    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default='exp_0')
    parser.add_argument('--fix-layers', type=str, default=None)
    parser.add_argument('--no-dense-loss', action='store_true') 
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--prune-batch-size', type=int, default=1)
    parser.add_argument('--use-fp32', action='store_true')
    parser.add_argument('--metric-type', type=str, default='Wanda')
    parser.add_argument('--wise-dim', type=str, default='row')
    # Learning parameter settings
    parser.add_argument('--blocksize', type=int, default=-1)
    parser.add_argument('--sparsity-step', type=float, default=0.01)
    parser.add_argument('--lora-rank', type=int, default=-1)
    # Loss settings
    parser.add_argument('--norm-all', action='store_true')
    parser.add_argument('--prune-dense', action='store_true')
    parser.add_argument('--l2-alpha', type=float, default=1)
    parser.add_argument('--sparsity-beta', type=float, default=1)
    parser.add_argument('--no-sigmoid-smooth', action='store_true')
    # Scaler (norm, value) and Scheduler
    parser.add_argument('--clip-grad', type=float)
    parser.add_argument('--clip-mode', type=str, default='norm')
    parser.add_argument('--no-scaler', action='store_true')
    parser.add_argument('--use-cos-sche', action='store_true')
    # Normal Opt settings (AdamW)
    parser.add_argument('--normal-opt', action='store_true')
    parser.add_argument('--normal-opt-lr', type=float, default=1e-2)
    parser.add_argument('--normal-default', action='store_true')
    # Prodigy settings
    parser.add_argument('--prodigy-lr', type=float, default=1)
    parser.add_argument('--no-decouple', action='store_true')
    parser.add_argument('--use-bias-correction', action='store_true')
    parser.add_argument('--safeguard-warmup', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--d-coef', type=float, default=1)

    args = parser.parse_args()

    return args


def main(args):
    print('Getting model ...')
    model = get_model(args.model, args.batch_size)

    if args.sparsity:
        print('Loading dataset ...')
        dataloader, c4_testenc = get_loaders(
            "c4", nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        _, wikitext_testenc = get_loaders(
            "wikitext2", nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        _, ptb_testenc = get_loaders(
            "ptb", nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )

        ppl_test_sets = ['c4', 'wikitext', 'ptb']
        gc.collect()
        torch.cuda.empty_cache()

        if args.eval_dense:
            result = lm_eval_model(model, args)
            print(f"Dense model zero-shot evaluation result: {result}")
            for set_name in ppl_test_sets:
                ppl = eval_ppl(model, eval(f"{set_name}_testenc"), args.batch_size)
                print(f"Dense model {set_name} ppl: {ppl}")

        wandb.login()
        wandb.init(
            project="LLaMA",
            name=args.exp_name,
            config={
                "model": args.model,
                "sparsity-step": args.sparsity_step,
                "epochs": args.epochs,
                "prune-batch-size": args.prune_batch_size,
                'l2-alpha': args.l2_alpha,
                'sparsity-beta': args.sparsity_beta,
                'fix-layers': args.fix_layers,
                'prune-dense': args.prune_dense,
                'dense-loss': not args.no_dense_loss
        })
        model_prune_log = compress_model(model, dataloader)
        wandb.finish()

        del dataloader
        torch.cuda.empty_cache()

        if not USE_LLaMA_SEQ:
            auto_map_model(model)

        if args.save_path:
            model.model.save_pretrained(args.save_path)
            model.tokenizer.save_pretrained(args.save_path)

    eval_result = lm_eval_model(model, args)
    print(f"Evaluation result: {eval_result}")
    c4_ppl = eval_ppl(model, c4_testenc, args.batch_size)
    ptb_ppl = eval_ppl(model, ptb_testenc, args.batch_size)
    wikitext_ppl = eval_ppl(model, wikitext_testenc, args.batch_size)
    for set_name in ppl_test_sets:
        print(f"{set_name} ppl: {eval(f'{set_name}_ppl')}")

    exp_log = os.path.join('exp_logs', f"{args.model.split('/')[-1]}-{args.exp_name}-{str(uuid.uuid4())}.pkl")
    while os.path.exists(exp_log):
        exp_log = os.path.join('exp_logs', f"{args.model.split('/')[-1]}-{args.exp_name}-{str(uuid.uuid4())}.pkl")
    with open(exp_log, 'wb') as f:
        pickle.dump({
            'args': args,
            'c4_ppl': c4_ppl,
            'ptb_ppl': ptb_ppl,
            'wikitext_ppl': wikitext_ppl,
            'eval_result': eval_result,
            'model_prune_log': model_prune_log,
        }, f)


if __name__ == "__main__":
    args = get_args()
    if torch.cuda.device_count() > 1:
        slurm_dist_init(args.seed, args.port)
    USE_LLaMA_SEQ = torch.cuda.device_count() == 1
    inference_context = contextlib.nullcontext() if args.use_fp32 else torch.cuda.amp.autocast()
    if is_master_process():
        print(args)
        main(args)