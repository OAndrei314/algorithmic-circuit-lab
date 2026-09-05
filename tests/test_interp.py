import torch

from circuit_lab.data import make_modular_addition_dataset
from circuit_lab.interp import (
    ablation_accuracy,
    compute_reference_means,
    direct_logit_attribution,
    fourier_power_spectrum,
    fraction_of_power_in_top_k,
    greedy_iterative_ablation,
    neuron_direct_logit_attribution,
    rank_components_by_dla,
    rank_neurons_by_dla,
)
from circuit_lab.model import OneLayerTransformer, TransformerConfig


def _tiny_model_and_data(p=7, seed=0):
    ds = make_modular_addition_dataset(p, train_fraction=0.5, seed=seed)
    cfg = TransformerConfig(vocab_size=ds.vocab_size, n_ctx=3, d_model=16, n_heads=2, d_mlp=32, seed=seed)
    model = OneLayerTransformer(cfg)
    return model, ds, cfg


def test_direct_logit_attribution_is_an_exact_decomposition():
    """direct + mlp + sum(per_head) must equal the actual correct-class
    logit exactly (up to float error), since there is no LayerNorm to break
    additivity. This is the load-bearing correctness property of the whole
    DLA analysis -- if it doesn't hold, the attribution numbers are lying."""
    model, ds, _ = _tiny_model_and_data()
    inputs, labels = ds.test_inputs(), ds.test_labels()
    dla = direct_logit_attribution(model, inputs, labels)

    reconstructed = dla["direct_correct"] + dla["mlp_correct"] + dla["per_head_correct"].sum(dim=-1)
    assert torch.allclose(reconstructed, dla["full_correct"], atol=1e-4)


def test_rank_components_by_dla_returns_all_heads_once():
    model, ds, cfg = _tiny_model_and_data()
    order, scores = rank_components_by_dla(model, ds.test_inputs(), ds.test_labels())
    assert sorted(order) == list(range(cfg.n_heads))
    assert len(scores) == cfg.n_heads


def test_ablation_accuracy_with_no_targets_matches_plain_accuracy():
    model, ds, _ = _tiny_model_and_data()
    inputs, labels = ds.test_inputs(), ds.test_labels()
    baseline = ablation_accuracy(model, inputs, labels)

    with torch.no_grad():
        preds = model(inputs)[:, -1, :].argmax(dim=-1)
    manual = (preds == labels).float().mean().item()
    assert baseline == manual


def test_mean_ablation_uses_supplied_means_not_zero():
    model, ds, cfg = _tiny_model_and_data()
    train_inputs = ds.train_inputs()
    mean_z, mean_mlp_post = compute_reference_means(model, train_inputs)

    # Mean-ablating a head should generally differ from zero-ablating it,
    # since the reference mean is (almost certainly) not exactly zero.
    zero_logits = model(train_inputs, ablate_heads=[0])
    mean_logits = model(train_inputs, ablate_heads=[0], mean_z=mean_z)
    assert not torch.allclose(zero_logits, mean_logits)
    assert torch.linalg.norm(mean_z[:, 0, :]) > 0


def test_fourier_spectrum_is_normalized_and_correctly_shaped():
    model, ds, cfg = _tiny_model_and_data(p=11)
    spectrum = fourier_power_spectrum(model, p=11)
    assert spectrum.power.shape == (11 // 2 + 1,)
    assert torch.isclose(spectrum.power.sum(), torch.tensor(1.0), atol=1e-5)
    assert sorted(spectrum.top_frequencies) == list(range(11 // 2 + 1))


def test_top_k_power_fraction_is_monotonic_in_k():
    model, ds, cfg = _tiny_model_and_data(p=11)
    spectrum = fourier_power_spectrum(model, p=11)
    frac_1 = fraction_of_power_in_top_k(spectrum, 1)
    frac_3 = fraction_of_power_in_top_k(spectrum, 3)
    frac_all = fraction_of_power_in_top_k(spectrum, 11 // 2 + 1)
    assert frac_1 <= frac_3 <= frac_all
    assert abs(frac_all - 1.0) < 1e-5


def test_neuron_dla_is_an_exact_decomposition_of_the_mlp_contribution():
    """per_neuron_correct.sum(-1) + bias_correct must reconstruct the whole
    MLP's contribution to the correct logit exactly -- the same correctness
    bar as the head-level decomposition, just one level deeper (down to
    individual post-GELU neuron activations instead of whole components)."""
    model, ds, _ = _tiny_model_and_data()
    inputs, labels = ds.test_inputs(), ds.test_labels()
    dla = direct_logit_attribution(model, inputs, labels)
    neuron_dla = neuron_direct_logit_attribution(model, inputs, labels)

    reconstructed_mlp = neuron_dla["per_neuron_correct"].sum(dim=-1) + neuron_dla["bias_correct"]
    assert torch.allclose(reconstructed_mlp, dla["mlp_correct"], atol=1e-4)


def test_rank_neurons_by_dla_returns_all_neurons_once():
    model, ds, cfg = _tiny_model_and_data()
    order, scores = rank_neurons_by_dla(model, ds.test_inputs(), ds.test_labels())
    assert sorted(order) == list(range(cfg.d_mlp))
    assert len(scores) == cfg.d_mlp


def test_ablating_top_dla_neuron_actually_changes_the_logits():
    """Sanity check that the neuron index returned by ``rank_neurons_by_dla``
    is actually wired into ``ablate_neurons`` correctly: mean-ablating it
    should change the model's logits (it is exceedingly unlikely, but not
    provable via accuracy alone, that a random-init neuron's activation
    already equals its own reference mean at every example). This catches an
    off-by-one/wrong-axis indexing bug that an accuracy-only check could miss
    by ties."""
    model, ds, _ = _tiny_model_and_data(p=11)
    inputs, labels = ds.test_inputs(), ds.test_labels()
    order, _ = rank_neurons_by_dla(model, inputs, labels)
    _, mean_mlp_post = compute_reference_means(model, ds.train_inputs())

    with torch.no_grad():
        baseline_logits = model(inputs)
    ablated_logits = model(inputs, ablate_neurons=[order[0]], mean_mlp_post=mean_mlp_post)
    assert not torch.allclose(baseline_logits, ablated_logits)


def test_greedy_iterative_ablation_rejects_unknown_kind():
    model, ds, _ = _tiny_model_and_data()
    inputs, labels = ds.test_inputs(), ds.test_labels()
    try:
        greedy_iterative_ablation(model, inputs, labels, [0, 1], ablate_kind="neuron")
        assert False, "expected ValueError for a non-plural ablate_kind"
    except ValueError:
        pass


def test_greedy_iterative_ablation_visits_every_candidate_exactly_once():
    model, ds, cfg = _tiny_model_and_data()
    inputs, labels = ds.test_inputs(), ds.test_labels()
    mean_z, _ = compute_reference_means(model, ds.train_inputs())

    trajectory = greedy_iterative_ablation(
        model, inputs, labels, list(range(cfg.n_heads)), ablate_kind="heads", mean_z=mean_z
    )
    visited = [component for component, _ in trajectory]
    assert sorted(visited) == list(range(cfg.n_heads))
    assert len(trajectory) == cfg.n_heads


def test_greedy_iterative_ablation_final_step_matches_ablating_everything_at_once():
    """The last entry's accuracy, after greedily removing every candidate
    one at a time, must equal the accuracy from ablating all of them
    together in a single call -- the cumulative ablated set is the same
    either way, so the two computations should agree exactly."""
    model, ds, cfg = _tiny_model_and_data()
    inputs, labels = ds.test_inputs(), ds.test_labels()
    mean_z, _ = compute_reference_means(model, ds.train_inputs())
    all_heads = list(range(cfg.n_heads))

    trajectory = greedy_iterative_ablation(
        model, inputs, labels, all_heads, ablate_kind="heads", mean_z=mean_z
    )
    expected_final_acc = ablation_accuracy(model, inputs, labels, ablate_heads=all_heads, mean_z=mean_z)
    assert trajectory[-1][1] == expected_final_acc


def test_greedy_iterative_ablation_first_pick_matches_brute_force_single_ablation():
    """The first component greedy removes must be whichever single
    candidate, ablated alone, does the most damage -- brute-forced here
    independently of the greedy loop as a correctness cross-check."""
    model, ds, cfg = _tiny_model_and_data(p=11)
    inputs, labels = ds.test_inputs(), ds.test_labels()
    _, mean_mlp_post = compute_reference_means(model, ds.train_inputs())
    candidates = list(range(cfg.d_mlp))

    brute_force_accs = {
        c: ablation_accuracy(model, inputs, labels, ablate_neurons=[c], mean_mlp_post=mean_mlp_post)
        for c in candidates
    }
    expected_first = min(brute_force_accs, key=brute_force_accs.get)

    trajectory = greedy_iterative_ablation(
        model, inputs, labels, candidates, ablate_kind="neurons", mean_mlp_post=mean_mlp_post
    )
    assert trajectory[0][0] == expected_first
    assert trajectory[0][1] == brute_force_accs[expected_first]


def test_greedy_iterative_ablation_with_no_candidates_returns_empty_trajectory():
    model, ds, _ = _tiny_model_and_data()
    inputs, labels = ds.test_inputs(), ds.test_labels()
    trajectory = greedy_iterative_ablation(model, inputs, labels, [], ablate_kind="heads")
    assert trajectory == []
