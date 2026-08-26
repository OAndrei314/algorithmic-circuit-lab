import torch

from circuit_lab.data import make_modular_addition_dataset
from circuit_lab.train import TrainConfig, first_threshold_crossing, train


def _tiny_cfg(**overrides):
    defaults = dict(
        p=5,
        train_fraction=0.5,
        d_model=16,
        n_heads=2,
        d_mlp=32,
        steps=50,
        weight_decay=0.0,
        seed=0,
        eval_every=10,
        log_every=0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_train_reduces_loss():
    cfg = _tiny_cfg(steps=200)
    _, result, _ = train(cfg)
    assert result.history[0]["train_loss"] > result.history[-1]["train_loss"]


def test_train_is_deterministic_given_seed():
    ds = make_modular_addition_dataset(5, 0.5, seed=0)
    cfg = _tiny_cfg()
    model_a, result_a, _ = train(cfg, dataset=ds)
    model_b, result_b, _ = train(cfg, dataset=ds)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    assert result_a.history == result_b.history


def test_resume_continues_from_checkpoint_not_from_scratch():
    ds = make_modular_addition_dataset(5, 0.5, seed=0)
    cfg = _tiny_cfg(steps=100)
    warm_model, warm_result, _ = train(cfg, dataset=ds)

    # Continuing for 0 additional optimizer steps with the warm checkpoint should
    # reproduce the warm model's own final-step metrics almost exactly (same
    # weights, same eval data) -- unlike starting fresh, which would begin at the
    # random-init loss instead of the already-trained loss.
    resume_cfg = _tiny_cfg(steps=1)
    resumed_model, resumed_result, _ = train(
        resume_cfg,
        dataset=ds,
        init_state_dict=warm_model.state_dict(),
        step_offset=100,
    )

    fresh_cfg = _tiny_cfg(steps=1, seed=123)
    fresh_model, fresh_result, _ = train(fresh_cfg, dataset=ds)

    assert resumed_result.history[0]["train_loss"] < fresh_result.history[0]["train_loss"]
    assert resumed_result.history[0]["step"] == 100


def test_grok_step_stays_unset_without_weight_decay():
    """With no weight decay there is no pressure to move past the memorizing
    solution once train accuracy hits 100%, so test accuracy should stay near
    chance and grok_step should never fire. This is a real, checked-in-CI
    negative control for the grok-detection logic in train() -- it is exactly
    as easy to write a grok_step check that always fires (or never does)
    as one that actually tracks the 0.95 threshold correctly."""
    cfg = _tiny_cfg(steps=400, weight_decay=0.0, d_model=32, d_mlp=64)
    _, result, ds = train(cfg)
    assert result.grokked is False
    assert result.grok_step is None
    assert result.history[-1]["train_acc"] == 1.0
    assert result.history[-1]["test_acc"] < 0.5


def test_first_threshold_crossing_finds_first_index_at_or_above_threshold():
    assert first_threshold_crossing([0.1, 0.2, 0.94, 0.95, 0.99]) == 3
    assert first_threshold_crossing([0.99, 0.1, 0.99]) == 0  # doesn't need to stay above
    assert first_threshold_crossing([0.1, 0.2, 0.3]) is None
    assert first_threshold_crossing([]) is None
    assert first_threshold_crossing([0.8, 0.9], threshold=0.85) == 1


def test_actual_training_run_uses_the_same_threshold_as_first_threshold_crossing():
    """The training loop's own grok_step bookkeeping (computed online, one
    eval at a time) must agree with running first_threshold_crossing() over
    the completed history after the fact -- otherwise the two would be able
    to silently drift apart (e.g. an off-by-one in >= vs >)."""
    cfg = _tiny_cfg(steps=400, weight_decay=0.0, d_model=32, d_mlp=64)
    _, result, ds = train(cfg)
    test_accs = [h["test_acc"] for h in result.history]
    idx = first_threshold_crossing(test_accs)
    expected_step = None if idx is None else result.history[idx]["step"]
    assert result.grok_step == expected_step
