import torch

from circuit_lab.data import make_modular_addition_dataset


def test_covers_every_pair_exactly_once():
    p = 11
    ds = make_modular_addition_dataset(p, train_fraction=0.3, seed=0)
    assert ds.inputs.shape == (p * p, 3)
    pairs = {(int(a), int(b)) for a, b, _ in ds.inputs}
    assert len(pairs) == p * p


def test_labels_match_modular_addition():
    p = 11
    ds = make_modular_addition_dataset(p, train_fraction=0.3, seed=0)
    for (a, b, eq), label in zip(ds.inputs, ds.labels):
        assert int(eq) == p
        assert int(label) == (int(a) + int(b)) % p


def test_train_test_split_is_disjoint_and_sized_correctly():
    p = 20
    ds = make_modular_addition_dataset(p, train_fraction=0.3, seed=0)
    n = p * p
    assert ds.train_mask.sum().item() == round(0.3 * n)
    assert (ds.train_mask & ~ds.train_mask).sum().item() == 0  # sanity
    train_pairs = {tuple(x.tolist()) for x in ds.train_inputs()}
    test_pairs = {tuple(x.tolist()) for x in ds.test_inputs()}
    assert train_pairs.isdisjoint(test_pairs)
    assert len(train_pairs) + len(test_pairs) == n


def test_deterministic_given_seed():
    ds1 = make_modular_addition_dataset(13, 0.3, seed=42)
    ds2 = make_modular_addition_dataset(13, 0.3, seed=42)
    assert torch.equal(ds1.train_mask, ds2.train_mask)

    ds3 = make_modular_addition_dataset(13, 0.3, seed=1)
    assert not torch.equal(ds1.train_mask, ds3.train_mask)


def test_rejects_invalid_train_fraction():
    import pytest

    with pytest.raises(ValueError):
        make_modular_addition_dataset(11, train_fraction=0.0)
    with pytest.raises(ValueError):
        make_modular_addition_dataset(11, train_fraction=1.0)
