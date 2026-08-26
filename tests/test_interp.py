import torch

from circuit_lab.data import make_modular_addition_dataset
from circuit_lab.interp import (
    ablation_accuracy,
    compute_reference_means,
    direct_logit_attribution,
    fourier_power_spectrum,
    fraction_of_power_in_top_k,
    rank_components_by_dla,
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
