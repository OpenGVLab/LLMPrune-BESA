import argparse
from hw_sim.check_funcs import check_sparsity, check_dense, check_proj, check_attn, check_mlp, check_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, default='llama-7b-0.5')
    parser.add_argument('-f', '--func', type=str, default='model')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    args = parser.parse_args()

    model_name = args.model_name
    func_name = args.func
    if func_name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
        check_proj(model_name, func_name, args.threshold)
    elif func_name in ['attn', 'mlp', 'model']:
        eval(f"check_{func_name}")(model_name, args.threshold)
    elif func_name in ['dense', 'sparsity']:
        eval(f"check_{func_name}")(model_name)
    else:
        raise ValueError(f"Invalid func name: {func_name}")
