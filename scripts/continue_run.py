"""One-off helper: continue training from a saved checkpoint for more steps,
merging history. Used to extend a run that was still trending toward
grokking when its original step budget ran out, without re-running the
already-completed steps.

    python scripts/continue_run.py --run-dir runs/modadd_p53 --extra-steps 30000
"""

import argparse
import json

import torch

from circuit_lab.data import make_modular_addition_dataset
from circuit_lab.train import TrainConfig, save_run, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--extra-steps", type=int, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)

    with open(f"{args.run_dir}/config.json") as f:
        cfg_dict = json.load(f)
    with open(f"{args.run_dir}/history.json") as f:
        old = json.load(f)

    prior_steps = old["history"][-1]["step"] + 1
    cfg = TrainConfig(**cfg_dict)
    dataset = make_modular_addition_dataset(cfg.p, cfg.train_fraction, seed=cfg.seed)

    state_dict = torch.load(f"{args.run_dir}/model.pt")
    cont_cfg = TrainConfig(**{**cfg_dict, "steps": args.extra_steps})
    model, result, _ = train(
        cont_cfg, dataset=dataset, init_state_dict=state_dict, step_offset=prior_steps
    )

    merged_history = old["history"] + result.history
    grok_step = old.get("grok_step")
    if grok_step is None:
        grok_step = result.grok_step
    grokked = old["grokked"] or result.grokked

    result.history = merged_history
    result.grok_step = grok_step
    result.grokked = grokked
    result.wall_clock_seconds = old["wall_clock_seconds"] + result.wall_clock_seconds

    save_run(model, result, args.run_dir)
    print(f"total_steps={prior_steps + args.extra_steps} grokked={grokked} grok_step={grok_step}")
    print(f"final: {merged_history[-1]}")


if __name__ == "__main__":
    main()
