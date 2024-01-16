# row-wise sparsity
python main.py --exp-name LLaMA-7B-r1-e1-df5e0-beta5e0 --epochs 1 --d-coef 5e0 --sparsity-beta 5e0 --blocksize 1 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-7b-hf
python main.py --exp-name LLaMA-13B-r1-e1-df5e0-beta5e0 --epochs 1 --d-coef 5e0 --sparsity-beta 5e0 --blocksize 1 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-13b-hf
python main.py --exp-name LLaMA-30B-r1-e1-df1e1-beta5e0 --epochs 1 --d-coef 1e1 --sparsity-beta 5e0 --blocksize 1 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-30b-hf
python main.py --exp-name LLaMA-65B-r1-e1-df5e1-beta5e0 --epochs 1 --d-coef 5e1 --sparsity-beta 5e0 --blocksize 1 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-65b-hf
python main.py --exp-name LLaMA2-7B-r1-e1-df5e0-beta5e0 --epochs 1 --d-coef 5e0 --sparsity-beta 5e0 --blocksize 1 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama2-7b-hf
python main.py --exp-name LLaMA2-13B-r1-e1-df5e0-beta5e0 --epochs 1 --d-coef 5e0 --sparsity-beta 5e0 --blocksize 1 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama2-13b-hf
python main.py --exp-name LLaMA2-70B-r1-e1-df5e1-beta5e0 --epochs 1 --d-coef 5e1 --sparsity-beta 5e0 --blocksize 1 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama2-70b-hf

# layer-wise sparsity
python main.py --exp-name Dense-LLaMA-7B-e1-df5e1-beta5e0 --prune-dense --epochs 1 --d-coef 5e1 --sparsity-beta 5e0 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-7b-hf
python main.py --exp-name Dense-LLaMA-13B-e1-df5e-1-beta5e0 --prune-dense --epochs 1 --d-coef 5e-1 --sparsity-beta 5e0 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-13b-hf
python main.py --exp-name Dense-LLaMA-30B-e1-df5e-2-beta5e0 --prune-dense --epochs 1 --d-coef 5e-2 --sparsity-beta 5e0 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-30b-hf
python main.py --exp-name Dense-LLaMA-65B-e1-df5e-1-beta5e0 --prune-dense --epochs 1 --d-coef 5e-1 --sparsity-beta 5e0 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama-65b-hf
python main.py --exp-name Dense-LLaMA2-7B-e1-df5e-2-beta5e0 --prune-dense --epochs 1 --d-coef 5e-2 --sparsity-beta 5e0 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama2-7b-hf
python main.py --exp-name Dense-LLaMA2-13B-e1-df5e-2-beta5e0 --prune-dense --epochs 1 --d-coef 5e-2 --sparsity-beta 5e0 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama2-13b-hf
python main.py --exp-name Dense-LLaMA2-70B-e1-df5e-1-beta5e0 --prune-dense --epochs 1 --d-coef 5e-1 --sparsity-beta 5e0 --batch-size 16 --model /mnt/lustre/share_data/xupeng/llama2-70b-hf