"""
Train a linear probe for each layer to predict verbalised confidence from embeddings.

This file only looks at one token position: TBG; train_multitoken_verbalised_confidence_probe.py looks at more token positions.
This script loads pickle files, extracts all layers from emb_tok_bef_gen,
trains one independent probe per layer with the same configuration (model type, hyperparameters,
same train/val data), and saves under results/all_layers/<run_id>/layer_1/, layer_2/, ...

Usage:
    python train_verbalised_confidence_probe_all_layers.py \
        --train_path semantic_uncertainty/out/1/train_linear_probe.pkl \
        --val_path semantic_uncertainty/out/1/validation_linear_probe.pkl \
        [--output_dir ./results] \
        [--model_type ridge] \
        [--alpha 1.0] \
        [--plot]
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from verbalised_confidence_probes.train_verbalised_confidence_probe import (
        plot_results,
        train_verbalised_confidence_probe,
        write_config_txt,
    )
except ImportError:
    from train_verbalised_confidence_probe import (
        plot_results,
        train_verbalised_confidence_probe,
        write_config_txt,
    )


def _load_pickle_batches_minimal(path):
    """
    Load a multi-dump pickle file (written as multiple pickle.dump batches) in a loop.
    Incrementally builds a dict with only verbalised_confidence and emb_tok_bef_gen per example.
    """
    data = {}
    with open(path, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
            except EOFError:
                break
            if not isinstance(batch, dict):
                print(f"ERROR: Batch is not a dict: {type(batch)}")
                continue
            for example_id, example_data in batch.items():
                responses = example_data.get('responses', [])
                if not responses:
                    continue
                r = responses[0]
                vc = r.get('verbalised_confidence')
                emb = r.get('emb_tok_bef_gen')
                if vc is None or emb is None:
                    print(f"ERROR: No verbalised confidence or embeddings for example {example_id}")
                    continue
                data[example_id] = {
                    'responses': [{'verbalised_confidence': vc, 'emb_tok_bef_gen': emb}]
                }
    return data


def load_verbalised_confidence_data_all_layers(train_path, val_path):
    """
    Load pickle files and extract verbalised confidence and all-layer embeddings from emb_tok_bef_gen.

    Returns:
        X_train: (n_examples, n_layers, hidden_dim)
        X_val: (n_examples, n_layers, hidden_dim)
        y_train, y_val: (n_examples,) float
        n_layers: int
    """
    train_data = _load_pickle_batches_minimal(train_path)
    val_data = _load_pickle_batches_minimal(val_path)

    def extract_all_layers_and_labels(data_dict):
        X_list = []
        y_list = []
        for example_id, example_data in data_dict.items():
            responses = example_data.get('responses', [])
            if len(responses) == 0:
                print(f"ERROR: No responses for example {example_id}")
                continue
            if len(responses) > 1:
                print(f"ERROR: Multiple responses for example {example_id}")
            response = responses[0]
            verbalised_confidence = response.get('verbalised_confidence')
            emb_tok_bef_gen = response.get('emb_tok_bef_gen')
            if verbalised_confidence is None or emb_tok_bef_gen is None:
                print(f'ERROR: No verbalised confidence or embeddings for example {example_id}')
                continue
            if isinstance(emb_tok_bef_gen, np.ndarray):
                emb_array = emb_tok_bef_gen
            elif hasattr(emb_tok_bef_gen, 'numpy'):
                emb_array = emb_tok_bef_gen.numpy()
            elif hasattr(emb_tok_bef_gen, 'cpu'):
                emb_array = emb_tok_bef_gen.cpu().numpy()
            else:
                emb_array = np.array(emb_tok_bef_gen)
            # emb_array shape: (num_layers, batch_size, hidden_dim)
            # Take batch index 0 -> (num_layers, hidden_dim)
            layer_embs = emb_array[:, 0, :]
            X_list.append(layer_embs)
            y_list.append(float(verbalised_confidence))
        return np.array(X_list), np.array(y_list)

    X_train, y_train = extract_all_layers_and_labels(train_data)
    X_val, y_val = extract_all_layers_and_labels(val_data)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("No valid examples in train or val data.")

    n_layers = X_train.shape[1]
    hidden_dim = X_train.shape[2]
    if X_val.shape[1] != n_layers or X_val.shape[2] != hidden_dim:
        raise ValueError(
            f"Train and val layer/hidden dim mismatch: train ({n_layers}, {hidden_dim}), "
            f"val ({X_val.shape[1]}, {X_val.shape[2]})"
        )

    print(f"Loaded {len(X_train)} training examples, {len(X_val)} validation examples")
    print(f"Layers: {n_layers}, embedding dim per layer: {hidden_dim}")
    print(f"Confidence range: [{y_train.min():.3f}, {y_train.max():.3f}]")

    return X_train, X_val, y_train, y_val, n_layers


def _plot_metrics_by_layer(layer_numbers, train_metrics, val_metrics, metric_name, metric_label, run_base: Path):
    """Plot one metric (train and val) vs layer number and save to run_base."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_numbers, train_metrics, 'o-', label='Train', markersize=4)
    ax.plot(layer_numbers, val_metrics, 's-', label='Validation', markersize=4)
    ax.set_xlabel('Layer number')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} by layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = run_base / f'{metric_name}_by_layer.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_all_metrics_by_layer(run_base: Path, layer_numbers, train_metrics_list, val_metrics_list):
    """Create 3 graphs: MSE, MAE, R² vs layer number."""
    metrics_config = [
        ('mse', 'MSE', train_metrics_list[0], val_metrics_list[0]),
        ('mae', 'MAE', train_metrics_list[1], val_metrics_list[1]),
        ('r2', 'R²', train_metrics_list[2], val_metrics_list[2]),
    ]
    for metric_name, metric_label, train_vals, val_vals in metrics_config:
        _plot_metrics_by_layer(layer_numbers, train_vals, val_vals, metric_name, metric_label, run_base)


def _get_run_base_dir(output_dir: Path) -> Path:
    """Return output_dir / all_layers / k where k is the first unused 1, 2, 3, ..."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_layers_dir = output_dir / "all_layers"
    all_layers_dir.mkdir(parents=True, exist_ok=True)
    k = 1
    while (all_layers_dir / str(k)).exists():
        k += 1
    run_base = all_layers_dir / str(k)
    run_base.mkdir(parents=True, exist_ok=True)
    return run_base


def main():
    parser = argparse.ArgumentParser(
        description="Train one verbalised-confidence probe per layer (same config, independent probes)"
    )
    parser.add_argument('--train_path', type=str, required=True, help='Path to train_linear_probe.pkl')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation_linear_probe.pkl')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory (default: ./results); run dir is output_dir/all_layers/<run_id>/',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='ridge',
        choices=['ridge', 'linear'],
        help='Type of regression model (default: ridge)',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Regularization strength for Ridge (default: 1.0)',
    )
    parser.add_argument('--plot', action='store_true', help='Save train/val regression plots per layer')
    parser.add_argument('--save_model', default=True, action='store_true', help='Save trained probes to pickle')
    args = parser.parse_args()

    print("Loading data...")
    X_train, X_val, y_train, y_val, n_layers = load_verbalised_confidence_data_all_layers(
        args.train_path, args.val_path
    )

    run_base = _get_run_base_dir(Path(args.output_dir))
    print(f"\nRun directory: {run_base}")

    layer_numbers = []
    train_mse, train_mae, train_r2 = [], [], []
    val_mse, val_mae, val_r2 = [], [], []

    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx + 1}"
        layer_dir = run_base / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        X_train_l = X_train[:, layer_idx, :]
        X_val_l = X_val[:, layer_idx, :]

        model, metrics = train_verbalised_confidence_probe(
            X_train_l, y_train, X_val_l, y_val,
            model_type=args.model_type,
            alpha=args.alpha,
            verbose=False,
        )

        layer_numbers.append(layer_idx + 1)
        train_mse.append(metrics['train']['mse'])
        train_mae.append(metrics['train']['mae'])
        train_r2.append(metrics['train']['r2'])
        val_mse.append(metrics['val']['mse'])
        val_mae.append(metrics['val']['mae'])
        val_r2.append(metrics['val']['r2'])

        if args.plot:
            plot_results(y_train, model.predict(X_train_l), 'train', str(layer_dir))
            plot_results(y_val, model.predict(X_val_l), 'val', str(layer_dir))

        if args.save_model:
            model_path = layer_dir / 'verbalised_confidence_probe.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'metrics': metrics,
                    'layer_idx': layer_idx,
                    'model_type': args.model_type,
                    'alpha': args.alpha if args.model_type == 'ridge' else None,
                }, f)
            print(f"Saved model to {model_path}")
            layer_args = argparse.Namespace(layer_idx=layer_idx, alpha=args.alpha)
            write_config_txt(
                layer_dir, layer_args, args.model_type,
                n_train=len(X_train_l), n_val=len(X_val_l),
                metrics=metrics,
                train_path=str(args.train_path), val_path=str(args.val_path),
            )
            print(f"Saved config to {layer_dir / 'config.txt'}")

    # Plot each metric vs layer number (3 graphs)
    print("\nPlotting metrics by layer...")
    _plot_all_metrics_by_layer(
        run_base,
        layer_numbers,
        [train_mse, train_mae, train_r2],
        [val_mse, val_mae, val_r2],
    )

    print(f"\nDone. Trained {n_layers} probes in {run_base}")


if __name__ == '__main__':
    main()
