"""Re-run circuit analysis on an existing checkpoint without retraining.

Used to extend a run's ``analysis.json`` with a new analysis technique (e.g.
the exact-DLA neuron ranking added alongside the magnitude-proxy ranking)
against an already-grokked model, without paying for another multi-minute
training run to get there again. The dataset is rebuilt deterministically
from the saved config (same ``p`` / ``train_fraction`` / ``seed``), so the
train/test split matches the one the checkpoint was actually trained on.

    python scripts/reanalyze.py --run-dir runs/modadd_p53
"""

import argparse
import json

import torch

from circuit_lab.data import make_modular_addition_dataset
from circuit_lab.interp import (
    ablation_accuracy,
    compute_reference_means,
    rank_neurons_by_dla,
)
from circuit_lab.model import OneLayerTransformer, TransformerConfig
from circuit_lab.train import TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--top-k-neurons", type=int, default=20)
    args = parser.parse_args()

    torch.set_num_threads(1)

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

    train_inputs = dataset.train_inputs()
    test_inputs, test_labels = dataset.test_inputs(), dataset.test_labels()

    with open(f"{args.run_dir}/analysis.json") as f:
        report = json.load(f)

    _, mean_mlp_post = compute_reference_means(model, train_inputs)

    # Magnitude-proxy ranking, recomputed here so it's directly comparable
    # to the exact-DLA ranking below on identical inputs (the value already
    # in the report was computed the same way inside run_experiment.py).
    with torch.no_grad():
        _, cache = model(train_inputs, return_cache=True)
    neuron_magnitude_proxy = cache["mlp_post"][:, -1, :].abs().mean(dim=0)
    top_neurons_by_magnitude = torch.argsort(neuron_magnitude_proxy, descending=True)
    top_neurons_by_magnitude = top_neurons_by_magnitude[: args.top_k_neurons].tolist()

    neuron_dla_order, _ = rank_neurons_by_dla(model, test_inputs, test_labels)
    top_neurons_by_dla = neuron_dla_order[: args.top_k_neurons]
    acc_ablate_top_neurons_by_dla = ablation_accuracy(
        model, test_inputs, test_labels, ablate_neurons=top_neurons_by_dla, mean_mlp_post=mean_mlp_post
    )
    overlap = len(set(top_neurons_by_magnitude) & set(top_neurons_by_dla))

    report["top_20_neurons_by_magnitude_proxy"] = sorted(top_neurons_by_magnitude)
    report["top_20_neurons_by_exact_dla"] = sorted(top_neurons_by_dla)
    report["accuracy_after_ablating_top_20_neurons_by_dla"] = acc_ablate_top_neurons_by_dla
    report["neuron_ranking_overlap_count"] = overlap

    with open(f"{args.run_dir}/analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
