"""Reproduces the results reported in the README: train a 1-layer
transformer on modular addition until it groks, then run the circuit
analysis (direct logit attribution, ablation, Fourier spectrum) on the
result. Takes roughly 10-20 minutes on a laptop CPU with the default
settings.

    python scripts/run_experiment.py --out runs/modadd_p53
"""

import argparse
import json

import torch

from circuit_lab.interp import (
    ablation_accuracy,
    compute_reference_means,
    fourier_power_spectrum,
    fraction_of_power_in_top_k,
    rank_components_by_dla,
)
from circuit_lab.train import TrainConfig, save_run, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=53)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/modadd_p53")
    args = parser.parse_args()

    torch.set_num_threads(1)

    cfg = TrainConfig(
        p=args.p,
        train_fraction=args.train_fraction,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        steps=args.steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        eval_every=100,
        log_every=1000,
    )
    model, result, dataset = train(cfg)
    save_run(model, result, args.out)
    print(f"\ngrokked={result.grokked} grok_step={result.grok_step}")

    train_inputs, train_labels = dataset.train_inputs(), dataset.train_labels()
    test_inputs, test_labels = dataset.test_inputs(), dataset.test_labels()

    baseline_acc = ablation_accuracy(model, test_inputs, test_labels)
    order, dla_scores = rank_components_by_dla(model, test_inputs, test_labels)
    mean_z, mean_mlp_post = compute_reference_means(model, train_inputs)

    top_heads = order[:1]
    bottom_heads = order[-1:]
    acc_ablate_top = ablation_accuracy(
        model, test_inputs, test_labels, ablate_heads=top_heads, mean_z=mean_z
    )
    acc_ablate_bottom = ablation_accuracy(
        model, test_inputs, test_labels, ablate_heads=bottom_heads, mean_z=mean_z
    )
    acc_ablate_all_but_top = ablation_accuracy(
        model,
        test_inputs,
        test_labels,
        ablate_heads=[h for h in order if h != top_heads[0]],
        mean_z=mean_z,
    )

    spectrum = fourier_power_spectrum(model, cfg.p)
    power_top6 = fraction_of_power_in_top_k(spectrum, 6)

    # Neuron-level ablation: top vs. random-equal-sized set, several seeds
    # for the random baseline so the comparison is honest.
    _, cache = model(train_inputs, return_cache=True)
    neuron_dla_proxy = cache["mlp_post"][:, -1, :].abs().mean(dim=0)
    top_neurons = torch.argsort(neuron_dla_proxy, descending=True)[:20].tolist()
    acc_ablate_top_neurons = ablation_accuracy(
        model, test_inputs, test_labels, ablate_neurons=top_neurons, mean_mlp_post=mean_mlp_post
    )
    random_neuron_accs = []
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        rand_neurons = torch.randperm(cfg.d_mlp, generator=g)[:20].tolist()
        random_neuron_accs.append(
            ablation_accuracy(
                model,
                test_inputs,
                test_labels,
                ablate_neurons=rand_neurons,
                mean_mlp_post=mean_mlp_post,
            )
        )

    report = {
        "config": vars(args),
        "grokked": result.grokked,
        "grok_step": result.grok_step,
        "final_train_acc": result.history[-1]["train_acc"],
        "final_test_acc": result.history[-1]["test_acc"],
        "baseline_test_accuracy": baseline_acc,
        "head_dla_scores": dla_scores,
        "top_heads_by_dla": top_heads,
        "accuracy_after_ablating_top_head": acc_ablate_top,
        "accuracy_after_ablating_bottom_head": acc_ablate_bottom,
        "accuracy_after_ablating_all_but_top_head": acc_ablate_all_but_top,
        "fraction_embedding_power_in_top_6_freqs": power_top6,
        "accuracy_after_ablating_top_20_neurons": acc_ablate_top_neurons,
        "accuracy_after_ablating_random_20_neurons_5_seeds": random_neuron_accs,
        "wall_clock_seconds": result.wall_clock_seconds,
    }
    with open(f"{args.out}/analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
