import pytest
import torch

from circuit_lab.data import make_modular_addition_dataset, make_modular_arithmetic_dataset


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
    with pytest.raises(ValueError):
        make_modular_addition_dataset(11, train_fraction=0.0)
    with pytest.raises(ValueError):
        make_modular_addition_dataset(11, train_fraction=1.0)


def test_addition_wrapper_matches_generic_function_with_op_add():
    ds_wrapper = make_modular_addition_dataset(13, train_fraction=0.3, seed=7)
    ds_generic = make_modular_arithmetic_dataset(13, op="add", train_fraction=0.3, seed=7)
    assert torch.equal(ds_wrapper.inputs, ds_generic.inputs)
    assert torch.equal(ds_wrapper.labels, ds_generic.labels)
    assert torch.equal(ds_wrapper.train_mask, ds_generic.train_mask)


def test_rejects_unknown_op():
    with pytest.raises(ValueError):
        make_modular_arithmetic_dataset(11, op="multiply")


def test_labels_match_modular_subtraction():
    p = 11
    ds = make_modular_arithmetic_dataset(p, op="subtract", train_fraction=0.3, seed=0)
    for (a, b, eq), label in zip(ds.inputs, ds.labels):
        assert int(eq) == p
        assert int(label) == (int(a) - int(b)) % p


def test_subtraction_is_not_commutative_unlike_addition():
    """Sanity check that subtraction actually exercises operand order: for
    most pairs (a, b) with a != b, a - b mod p differs from b - a mod p,
    unlike addition where the two orders always give the same label. If this
    ever failed it would mean the dataset silently degenerated into another
    commutative task, defeating the point of adding it."""
    p = 11
    ds = make_modular_arithmetic_dataset(p, op="subtract", train_fraction=0.3, seed=0)
    pair_to_label = {
        (int(a), int(b)): int(label) for (a, b, _), label in zip(ds.inputs, ds.labels)
    }
    order_sensitive = sum(
        1
        for a in range(p)
        for b in range(p)
        if a != b and pair_to_label[(a, b)] != pair_to_label[(b, a)]
    )
    total_asymmetric_pairs = sum(1 for a in range(p) for b in range(p) if a != b)
    assert order_sensitive == total_asymmetric_pairs


def test_subtraction_train_test_split_is_disjoint_and_sized_correctly():
    p = 20
    ds = make_modular_arithmetic_dataset(p, op="subtract", train_fraction=0.3, seed=0)
    n = p * p
    assert ds.train_mask.sum().item() == round(0.3 * n)
    train_pairs = {tuple(x.tolist()) for x in ds.train_inputs()}
    test_pairs = {tuple(x.tolist()) for x in ds.test_inputs()}
    assert train_pairs.isdisjoint(test_pairs)
    assert len(train_pairs) + len(test_pairs) == n
