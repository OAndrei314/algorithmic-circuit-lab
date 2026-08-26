import torch

from circuit_lab.data import make_modular_addition_dataset
from circuit_lab.model import OneLayerTransformer, TransformerConfig


def _tiny_model(seed=0):
    cfg = TransformerConfig(vocab_size=8, n_ctx=3, d_model=16, n_heads=2, d_mlp=32, seed=seed)
    return OneLayerTransformer(cfg), cfg


def test_forward_output_shape():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (5, 3))
    logits = model(tokens)
    assert logits.shape == (5, 3, cfg.vocab_size)


def test_cache_shapes():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (5, 3))
    logits, cache = model(tokens, return_cache=True)
    assert cache["z"].shape == (5, 3, cfg.n_heads, cfg.d_head)
    assert cache["attn_pattern"].shape == (5, cfg.n_heads, 3, 3)
    assert cache["mlp_post"].shape == (5, 3, cfg.d_mlp)
    assert cache["resid_final"].shape == (5, 3, cfg.d_model)
    assert torch.equal(cache["logits"], logits)


def test_attention_pattern_sums_to_one():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (4, 3))
    _, cache = model(tokens, return_cache=True)
    row_sums = cache["attn_pattern"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_gradients_flow_to_every_parameter():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (5, 3))
    logits = model(tokens)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


def test_zero_ablating_a_head_changes_output():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (5, 3))
    baseline = model(tokens)
    ablated = model(tokens, ablate_heads=[0])
    assert not torch.allclose(baseline, ablated)


def test_ablating_all_heads_and_neurons_reduces_to_direct_path():
    """With every attention head and every MLP neuron zero-ablated, the only
    surviving path is the embedding + positional embedding skip connection
    straight to the unembedding -- this is an exact algebraic identity given
    the no-LayerNorm architecture, not an approximation."""
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (5, 3))
    _, cache = model(tokens, return_cache=True)

    all_heads = list(range(cfg.n_heads))
    all_neurons = list(range(cfg.d_mlp))
    ablated_logits = model(tokens, ablate_heads=all_heads, ablate_neurons=all_neurons)

    direct_logits = torch.einsum("bpd,dv->bpv", cache["resid_pre"], model.W_U) + (
        model.b_out @ model.W_U
    )
    assert torch.allclose(ablated_logits, direct_logits, atol=1e-4)


def test_tiny_model_can_overfit_a_few_examples():
    """A genuine (if small) end-to-end training signal check: loss on a
    handful of examples should drop substantially within a few hundred
    full-batch gradient steps. This is not testing for grokking/
    generalization (that needs a much bigger run, see scripts/run_experiment.py) --
    just that the forward/backward pass actually optimizes."""
    ds = make_modular_addition_dataset(p=5, train_fraction=0.5, seed=0)
    cfg = TransformerConfig(vocab_size=ds.vocab_size, n_ctx=3, d_model=32, n_heads=2, d_mlp=64, seed=0)
    model = OneLayerTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    inputs, labels = ds.train_inputs(), ds.train_labels()
    losses = []
    for _ in range(300):
        logits = model(inputs)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < 0.1 * losses[0]
