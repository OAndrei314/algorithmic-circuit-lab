"""Command-line entry points: ``circuit-lab-train`` and ``circuit-lab-analyze``."""

import argparse
import json

import torch

from circuit_lab.data import make_modular_addition_dataset
from circuit_lab.interp import (
    ablation_accuracy,
    compute_reference_means,
    fourier_power_spectrum,
    fraction_of_power_in_top_k,
    rank_components_by_dla,
)
from circuit_lab.model import OneLayerTransformer, TransformerConfig
from circuit_lab.train import TrainConfig, save_run, train


def train_cli():
    parser = argparse.ArgumentParser(description="Train a 1-layer transformer on modular addition.")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/modadd")
    args = parser.parse_args()

    cfg = TrainConfig(
        p=args.p,
        train_fraction=args.train_fraction,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    model, result, _ = train(cfg)
    save_run(model, result, args.out)
    print(f"grokked={result.grokked} grok_step={result.grok_step} saved to {args.out}")


def analyze_cli():
    parser = argparse.ArgumentParser(description="Analyze a trained modular-addition transformer.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--top-k-heads", type=int, default=1)
    parser.add_argument("--top-k-freqs", type=int, default=6)
    args = parser.parse_args()

    with open(f"{args.run_dir}/config.json") as f:
        cfg_dict = json.load(f)
    cfg = TrainConfig(**cfg_dict)

    dataset = make_modular_addition_dataset(cfg.p, cfg.train_fraction, seed=cfg.seed)
    model_cfg = TransformerConfig(
        vocab_size=dataset.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        seed=cfg.seed,
    )
    model = OneLayerTransformer(model_cfg)
    model.load_state_dict(torch.load(f"{args.run_dir}/model.pt"))
    model.eval()

    train_inputs, train_labels = dataset.train_inputs(), dataset.train_labels()
    test_inputs, test_labels = dataset.test_inputs(), dataset.test_labels()

    baseline_acc = ablation_accuracy(model, test_inputs, test_labels)

    order, dla_scores = rank_components_by_dla(model, test_inputs, test_labels)
    mean_z, mean_mlp_post = compute_reference_means(model, train_inputs)

    top_heads = order[: args.top_k_heads]
    bottom_heads = order[-args.top_k_heads :]
    acc_ablate_top = ablation_accuracy(
        model, test_inputs, test_labels, ablate_heads=top_heads, mean_z=mean_z
    )
    acc_ablate_bottom = ablation_accuracy(
        model, test_inputs, test_labels, ablate_heads=bottom_heads, mean_z=mean_z
    )

    spectrum = fourier_power_spectrum(model, cfg.p)
    power_top_k = fraction_of_power_in_top_k(spectrum, args.top_k_freqs)

    report = {
        "baseline_test_accuracy": baseline_acc,
        "head_dla_scores": dla_scores,
        "top_heads_by_dla": top_heads,
        "accuracy_after_ablating_top_heads": acc_ablate_top,
        "accuracy_after_ablating_bottom_heads": acc_ablate_bottom,
        "fraction_embedding_power_in_top_k_freqs": power_top_k,
        "top_k_freqs": args.top_k_freqs,
    }
    print(json.dumps(report, indent=2))
    with open(f"{args.run_dir}/analysis.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    train_cli()
