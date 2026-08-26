"""Full-batch training loop for the modular addition task.

Uses full-batch gradient descent (the whole train split every step) with
AdamW and non-trivial weight decay, matching the setup that produces
"grokking": the model first memorizes the training pairs (train accuracy
saturates fast) and only much later, under continued weight decay pressure,
snaps to the generalizing algorithm (test accuracy jumps from chance to
~100% well after train accuracy has already hit 100%).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from circuit_lab.data import ModularAdditionDataset, make_modular_addition_dataset
from circuit_lab.model import OneLayerTransformer, TransformerConfig


@dataclass
class TrainConfig:
    p: int = 113
    train_fraction: float = 0.3
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    steps: int = 30_000
    lr: float = 1e-3
    weight_decay: float = 1.0
    warmup_steps: int = 10
    seed: int = 0
    eval_every: int = 100
    log_every: int = 500


@dataclass
class TrainResult:
    config: TrainConfig
    history: list  # list of dicts: step, train_loss, train_acc, test_loss, test_acc
    wall_clock_seconds: float
    grokked: bool
    grok_step: Optional[int]


GROK_THRESHOLD = 0.95


def first_threshold_crossing(
    history_test_accs: list, threshold: float = GROK_THRESHOLD
) -> Optional[int]:
    """Index of the first entry in ``history_test_accs`` at or above
    ``threshold``, or ``None`` if it's never crossed. Pulled out of the
    training loop as a pure function so the grok-detection logic can be unit
    tested without depending on an actual model reaching that accuracy."""
    for i, acc in enumerate(history_test_accs):
        if acc >= threshold:
            return i
    return None


def _loss_and_acc(logits: torch.Tensor, labels: torch.Tensor):
    final_logits = logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(final_logits, labels)
    acc = (final_logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def train(
    cfg: TrainConfig,
    dataset: Optional[ModularAdditionDataset] = None,
    init_state_dict: Optional[dict] = None,
    step_offset: int = 0,
) -> tuple:
    """Train a OneLayerTransformer on modular addition. Returns (model, TrainResult).

    ``init_state_dict`` / ``step_offset`` let a run continue training an
    existing checkpoint for more steps (e.g. because it was still visibly
    trending toward grokking when an earlier run ended) without re-deriving
    the first ``step_offset`` steps -- the logged ``step`` numbers in the
    returned history are offset so they stay comparable to the earlier run's.
    """
    if dataset is None:
        dataset = make_modular_addition_dataset(cfg.p, cfg.train_fraction, seed=cfg.seed)

    model_cfg = TransformerConfig(
        vocab_size=dataset.vocab_size,
        n_ctx=3,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        seed=cfg.seed,
    )
    model = OneLayerTransformer(model_cfg)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, cfg.warmup_steps))
    )

    train_inputs, train_labels = dataset.train_inputs(), dataset.train_labels()
    test_inputs, test_labels = dataset.test_inputs(), dataset.test_labels()

    history = []
    grok_step = None
    start = time.time()

    for step in range(cfg.steps):
        model.train()
        logits = model(train_inputs)
        loss, train_acc = _loss_and_acc(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            global_step = step + step_offset
            model.eval()
            with torch.no_grad():
                test_logits = model(test_inputs)
                test_loss, test_acc = _loss_and_acc(test_logits, test_labels)
            history.append(
                {
                    "step": global_step,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "test_loss": test_loss.item(),
                    "test_acc": test_acc,
                }
            )
            if grok_step is None and test_acc >= GROK_THRESHOLD:
                grok_step = global_step
            if cfg.log_every and step % cfg.log_every == 0:
                print(
                    f"step {global_step:6d}  train_loss {loss.item():.4f} train_acc {train_acc:.3f}  "
                    f"test_loss {test_loss.item():.4f} test_acc {test_acc:.3f}"
                )

    result = TrainResult(
        config=cfg,
        history=history,
        wall_clock_seconds=time.time() - start,
        grokked=grok_step is not None,
        grok_step=grok_step,
    )
    return model, result, dataset


def save_run(model: OneLayerTransformer, result: TrainResult, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "config.json", "w") as f:
        json.dump(asdict(result.config), f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(
            {
                "history": result.history,
                "wall_clock_seconds": result.wall_clock_seconds,
                "grokked": result.grokked,
                "grok_step": result.grok_step,
            },
            f,
            indent=2,
        )
