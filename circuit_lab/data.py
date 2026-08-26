"""Dataset generation for the modular addition task.

Each example is the sequence ``[a, b, EQUALS]`` and the model must predict
``(a + b) mod p`` at the final position. ``EQUALS`` is an extra token with
id ``p`` so the vocabulary has ``p + 1`` entries.
"""

from dataclasses import dataclass

import torch


@dataclass
class ModularAdditionDataset:
    p: int
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


def make_modular_addition_dataset(
    p: int, train_fraction: float = 0.3, seed: int = 0
) -> ModularAdditionDataset:
    """Build the full a, b in [0, p) x [0, p) grid for (a + b) mod p.

    A fixed fraction of the p^2 pairs is marked for training; the rest is
    held out as a test set. A low train fraction (Nanda et al. use 0.3) is
    what makes the task hard enough that memorizing the training set does
    not generalize -- the model has to find the actual mod-p addition
    algorithm to do well on the held-out pairs.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    equals = torch.full_like(a, fill_value=p)
    inputs = torch.stack([a, b, equals], dim=1)
    labels = (a + b) % p

    n = p * p
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_train = int(round(train_fraction * n))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True

    return ModularAdditionDataset(
        p=p,
        train_fraction=train_fraction,
        seed=seed,
        inputs=inputs,
        labels=labels,
        train_mask=train_mask,
    )
