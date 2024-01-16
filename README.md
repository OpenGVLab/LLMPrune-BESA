# BESA

This repository contains code to reproduce the key results of the paper "BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation", accepted at International Conference on Learning Representations (ICLR), 2024.

## Dependencies

* `torch`: tested on v2.0.1+cu118
* `transformers`: tested on v4.31.0
* `accelerate`: tested on v0.21.0
* `datasets`: tested on v2.14.4
* `timm`: tested on v0.9.5

**lm-evaluation-harness**
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

**Customized Cuda Operator**
```
cd models/ops
python setup.py install
```

## Usage

Here is the command to run baseline experiments followed by perplexity evaluations on WikiText2, PTB, C4 and zero-shot tasks.
See also the CMD-argument documentation.

```
bash main_exps.sh
```

## Hardware Simulation

We utilize the ViTCoD accelerator to achieve the speed-up that can be obtained through the sparse accelerator. Here is the command to simulate the average runtime of each module in the pruned model.

```
python main_hw.py \
--model-name MODEL_DIR or HF_MODEL_NAME
--func SIMULATE_CHOICE (q, k, v, o, gate, up, and down are available)
```

## Others

In the experiment section of our paper, we present the results of row-wise sparsity, which customize sparsity for each row of target layer's weight within in the block. Additionally, we provide an extension presenting the outcomes of layer-wise sparsity, where each row of the target layer is assigned uniform sparsity. You can find the commands to execute the layer-wise sparsity experiments in the **main_exps.sh** script. Below, we present the perplexity results for the Wikitext2 dataset.

|                   | 1-7B | 1-13B | 1-30B | 1-65B | 2-7B | 2-13B | 2-70B |
|------------------:|:-----|:------|:------|:------|:-----|:------|:------|
| Dense             | 5.68 | 5.09  | 4.10  | 3.53  | 5.47 | 4.88  | 3.31  |
| SparseGPT         | 7.22 | 6.21  | 5.33  | 4.60  | 6.99 | 6.02  | 4.25  |
| Wanda             | 7.26 | 6.15  | 5.25  | 4.60  | 6.92 | 5.97  | 4.22  |
| BESA (layer-wise) | 7.04 | 6.07  | 5.16  | 4.51  | 6.77 | 5.85  | 4.14  |
| BESA (row-wise)   | 6.86 | 5.92  | 5.00  | 4.33  | 6.60 | 5.75  | 4.09  |

## Acknowledgement

This repo benefits from [SparseGPT](https://github.com/IST-DASLab/sparsegpt), [Prodigy](https://github.com/konstmish/prodigy), and [ViTCoD](https://github.com/GATECH-EIC/ViTCoD). Thanks for their wonderful works.