"""Dataset generation for modular arithmetic tasks.

Each example is the sequence ``[a, b, EQUALS]`` and the model must predict
``a OP b (mod p)`` at the final position, where ``OP`` is either addition or
subtraction. ``EQUALS`` is an extra token with id ``p`` so the vocabulary has
``p + 1`` entries.

Subtraction is included alongside addition because it is *not* commutative:
``(a - b) mod p != (b - a) mod p`` in general, so a model that solves it
cannot get away with a representation that ignores which operand came first.
That makes it a natural stress test for whether the addition circuit's
"represent numbers as a few sinusoids, combine via a trig identity" strategy
is specific to commutative structure or generalizes to a task where operand
order carries information.
"""

from dataclasses import dataclass

import torch

SUPPORTED_OPS = ("add", "subtract")


def _apply_op(a: torch.Tensor, b: torch.Tensor, op: str, p: int) -> torch.Tensor:
    if op == "add":
        return (a + b) % p
    if op == "subtract":
        return (a - b) % p
    raise ValueError(f"op must be one of {SUPPORTED_OPS}, got {op!r}")


@dataclass
class ModularArithmeticDataset:
    p: int
    op: str
    train_fraction: float
    seed: int

    inputs: torch.Tensor  # (n, 3) int64
    labels: torch.Tensor  # (n,) int64
    train_mask: torch.Tensor  # (n,) bool

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def equals_token(self) -> int:
        return self.p

    def train_inputs(self) -> torch.Tensor:
        return self.inputs[self.train_mask]

    def train_labels(self) -> torch.Tensor:
        return self.labels[self.train_mask]

    def test_inputs(self) -> torch.Tensor:
        return self.inputs[~self.train_mask]

    def test_labels(self) -> torch.Tensor:
        return self.labels[~self.train_mask]


def make_modular_arithmetic_dataset(
    p: int, op: str = "add", train_fraction: float = 0.3, seed: int = 0
) -> ModularArithmeticDataset:
    """Build the full a, b in [0, p) x [0, p) grid for ``a OP b (mod p)``.

    A fixed fraction of the p^2 pairs is marked for training; the rest is
    held out as a test set. A low train fraction (Nanda et al. use 0.3) is
    what makes the task hard enough that memorizing the training set does
    not generalize -- the model has to find the actual modular-arithmetic
    algorithm to do well on the held-out pairs.
    """
    if op not in SUPPORTED_OPS:
        raise ValueError(f"op must be one of {SUPPORTED_OPS}, got {op!r}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    equals = torch.full_like(a, fill_value=p)
    inputs = torch.stack([a, b, equals], dim=1)
    labels = _apply_op(a, b, op, p)

    n = p * p
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_train = int(round(train_fraction * n))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True

    return ModularArithmeticDataset(
        p=p,
        op=op,
        train_fraction=train_fraction,
        seed=seed,
        inputs=inputs,
        labels=labels,
        train_mask=train_mask,
    )


def make_modular_addition_dataset(
    p: int, train_fraction: float = 0.3, seed: int = 0
) -> ModularArithmeticDataset:
    """``make_modular_arithmetic_dataset(p, op="add", ...)``, kept as a
    named entry point since addition is this repo's original, most-tested
    task."""
    return make_modular_arithmetic_dataset(p, op="add", train_fraction=train_fraction, seed=seed)
