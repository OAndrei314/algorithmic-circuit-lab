# algorithmic-circuit-lab

*Maintained by: claude-actions-daily-routine · Status: Active*

A from-scratch mechanistic-interpretability lab: train a tiny transformer on a synthetic
algorithmic task (modular addition) until it "groks" — generalizes well past the point where
it could just be memorizing — then reverse-engineer *what algorithm it actually learned*, and
check that reverse-engineering with causal experiments rather than trusting it on vibes.

This is an independent, from-scratch reimplementation of a well-known experimental paradigm
(Neel Nanda et al., ["Progress measures for grokking via mechanistic
interpretability"](https://arxiv.org/abs/2301.05217), 2023) — no code, weights, or data from
that work are used here. Everything in this repo — the model, the training loop, the
attribution/ablation/Fourier tooling — is written from scratch against that paradigm as a
target, not copied from any existing interpretability library (TransformerLens, etc).

## Why this problem

Every large-scale interpretability result rests on the same unverified assumption: that a
technique which looks reasonable on paper (patch this activation, attribute this logit to
that head, project onto that direction) actually reveals something true about what the
network computes, rather than a plausible-looking artifact. Modular addition is small enough
that "what the network computes" can be checked directly — the ground-truth algorithm is
just arithmetic — which makes it one of the few settings where you can hold an
interpretability technique to a real correctness bar instead of "the story is compelling."

Concretely, this repo asks and answers three separable questions on that testbed:

1. **Does the model actually generalize**, or is >99% test accuracy an accident of a lenient
   eval — i.e. does training show the delayed-generalization ("grokking") signature at all?
2. **Which parts of the network does direct logit attribution (DLA) say matter** — a cheap,
   purely correlational analysis (it just asks "how much does each component's output,
   projected through the unembedding, point at the correct answer")?
3. **Do those parts actually matter causally** — if you ablate the head DLA ranks as most
   important, does accuracy actually collapse, more than ablating a DLA-unimportant head or a
   random one? This is the step that separates "a good story" from "a verified mechanism,"
   and it's the step a lot of casual interpretability writeups skip.

## How it works

**Task.** Given `a, b ∈ [0, p)`, predict `(a + b) mod p`. Input is the 3-token sequence
`[a, b, EQUALS]`; the model reads off its answer from the logits at the final position. Only
a fixed fraction of the `p²` possible pairs (30% by default, following Nanda et al.) is used
for training — the rest is held out. That low train fraction is what makes the task hard: a
model that just memorizes its training pairs gets ~30% accuracy including train, but only
chance-level (~1/p) accuracy on the 70% of pairs it never saw. Only a model that discovers
the actual `mod p` addition algorithm does well on both.

**Model** (`circuit_lab/model.py`): a 1-layer, multi-head-attention + MLP transformer,
**with no LayerNorm**. That's a deliberate simplification, not an oversight — with no
LayerNorm, the residual stream is a literal sum of independent components (the embedding
skip-connection, the attention output, the MLP output), so the final logit is an *exact*
linear decomposition into each component's contribution. Every intermediate tensor needed
for interpretability (per-head attention output, per-neuron MLP activation) is directly
addressable, rather than hidden inside `nn.MultiheadAttention`.

**Training** (`circuit_lab/train.py`): full-batch AdamW with weight decay (1.0 by default).
Weight decay pressure after the model has already hit 100% train accuracy is the actual
mechanism behind grokking — it keeps nudging the parameters after memorization "works," and
at some point that pressure finds a lower-norm solution that happens to be the general
algorithm.

**Interpretability** (`circuit_lab/interp.py`):
- `direct_logit_attribution` — the exact decomposition described above, evaluated at the
  correct answer's logit specifically. Tested in `tests/test_interp.py` to actually sum back
  to the real logit (not just plausible-looking).
- `ablation_accuracy` — causal intervention: zero- or mean-ablate specific attention heads
  or MLP neurons (replacing their output with either 0 or their mean over a reference
  distribution — mean-ablation is the more careful choice, since zero-ablation also destroys
  a component's average contribution and can make an unimportant-but-nonzero-mean component
  look load-bearing) and re-measure test accuracy.
- `fourier_power_spectrum` — the known fingerprint of a generalizing modular-addition
  solution: each token's embedding, viewed as a function of the token's integer value 0..p-1,
  concentrates its power in a handful of low frequencies (the model represents numbers via
  roughly `sin`/`cos` features and does addition via trig angle-sum identities) rather than
  looking like noise.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Fast smoke test: tiny model, tiny p, a few hundred steps (~10s, does not grok --
# just confirms the pipeline runs end to end).
python scripts/run_experiment.py --p 11 --steps 500 --out runs/smoke

# Full experiment: trains until grokking, then runs the DLA / ablation / Fourier
# analysis and writes runs/modadd_p53/analysis.json. Takes ~15 minutes on a laptop CPU.
python scripts/run_experiment.py --out runs/modadd_p53
```

`circuit-lab-train` / `circuit-lab-analyze` (installed as console scripts) expose the same
training loop and analysis as separate steps if you want to inspect a checkpoint interactively
between them.

## Honest results

Setup: `p=53` (2,809 total pairs, 843 train / 1,966 test), `d_model=64`, 4 heads
(`d_head=16`), `d_mlp=256`, AdamW, `lr=1e-3`, `weight_decay=1.0`, seed 0, single CPU
thread. Full config and raw per-eval history are checked in at
[`runs/modadd_p53/`](runs/modadd_p53/) — `config.json`, `history.json`,
`analysis.json`, `model.pt` — so every number below is independently checkable, not
just asserted. Reproduce with `python scripts/run_experiment.py` followed by
`python scripts/continue_run.py --run-dir runs/modadd_p53 --extra-steps 30000`
(see "the training run took longer than expected" below for why it's two commands).
`python scripts/reanalyze.py --run-dir runs/modadd_p53` re-runs just the analysis step
(DLA, ablation, Fourier) against the existing checkpoint without retraining — used here
to add the exact-DLA neuron ranking to `analysis.json` after the fact.

### The training run took longer than expected, and that's worth reporting as-is

The first run trained for the originally-planned 30,000 steps. Train accuracy hit
100% almost immediately and stayed there; test accuracy crept up far more slowly than
expected and had only reached 33.0% by step 30,000 — visibly still trending upward,
but not grokked by the initial budget:

| step | train acc | test acc |
| ---: | ---: | ---: |
| 0 | 2.5% | 1.4% |
| 4,000 | 100% | 4.6% |
| 14,000 | 100% | 6.7% |
| 24,000 | 100% | 15.0% |
| 29,000 | 100% | 28.3% |
| 30,000 (end of first run) | 100% | 33.0% |

Rather than report that partial trajectory as "grokking is slow/unreliable at this
scale" — which would have been a plausible-sounding but wrong conclusion — the run
was resumed (`scripts/continue_run.py`, loading the checkpoint and continuing the
same optimizer trajectory) for another 30,000 steps. It kept climbing and completed a
full, clean phase transition:

| step | train acc | test acc |
| ---: | ---: | ---: |
| 34,000 | 100% | 60.2% |
| 38,000 | 100% | 74.9% |
| 40,000 | 100% | 82.3% |
| 42,000 | 100% | 89.9% |
| **42,600** | 100% | **95.0%** ← first crossing (`grok_step`) |
| 44,000 | 100% | 99.1% |
| 60,000 (final) | 100% | **99.24%** |

Total training wall-clock across both runs: 1,377s (~23 minutes) on a single CPU
thread. The honest takeaway: at this `p` and model size, the delayed-generalization
window is real and reproduces the literature's qualitative shape (train accuracy
saturates almost instantly; test accuracy stays near chance for tens of thousands of
steps, then transitions sharply once weight decay pressure finds the lower-norm
generalizing solution) — but the *specific* number of steps it takes is sensitive
enough to hyperparameters that a fixed budget picked in advance (the original 30,000)
genuinely wasn't enough, and extending it was the right call rather than reporting a
weaker result out of convenience.

### Does the embedding show the predicted Fourier structure?

Yes, clearly. The Fourier power spectrum of the numeric-token embeddings (`W_E[:53]`,
DFT along the token-value axis, power summed across all 64 embedding dimensions)
concentrates almost entirely in a handful of frequencies:

- Top 2 frequencies: **92.5%** of total embedding power.
- Top 6 frequencies: **95.7%** of total embedding power.

For comparison, an *untrained* model of identical shape (10 random seeds, measured the
same way) puts 8.6%-9.9% of power in its top 2 frequencies (mean 9.3%) — random
embeddings do have some concentration just from `d_model=64` being finite, but nowhere
near what training produces. The trained model's 92.5% is the expected fingerprint of
the "represent numbers as a few sinusoids, do addition via the trig angle-sum identity"
algorithm the literature describes for this task, not an artifact of the metric.

### Does direct logit attribution actually predict what ablation finds?

Partially — and the mismatch is itself the more interesting, honest result. Ranking
the 4 attention heads by direct logit attribution (DLA) to the correct answer gives
scores `[0.137, 0.467, 0.267, 0.368]` (heads 0-3) — head 1 is the clear DLA leader at
roughly 3.4x head 0's score. If DLA cleanly predicted causal importance, ablating head
1 alone should hurt far more than ablating head 0 alone. It doesn't, much:

| intervention (mean-ablated) | test accuracy | vs. baseline 99.24% |
| --- | ---: | ---: |
| none (baseline) | 99.24% | — |
| head 1 only (top DLA, score 0.467) | 36.37% | -62.9 pts |
| head 0 only (bottom DLA, score 0.137) | 31.43% | -67.8 pts |
| **all heads except head 1** | 3.76% | -95.5 pts |

Every single head, ablated alone, is catastrophic — there is no "safe to remove"
head, DLA ranking notwithstanding — but the *size* of the damage doesn't track the
DLA score ordering (head 0 hurts slightly *more* than head 1 despite a 3.4x lower
DLA score). Ablating everything except the DLA-favorite head is even more damaging
than ablating that head alone (3.76%, barely above the ~1.9% chance rate for 53
classes), confirming no single head is sufficient either. The likely mechanism,
consistent with the Fourier result above: the circuit sums several heads' separate
frequency-specific contributions to complete a trig identity, so removing *any one*
of them breaks the sum, largely independent of that head's average contribution
magnitude to any specific logit. **The practical lesson: DLA is a cheap, useful
correlational signal for finding candidate components, but it is not a substitute for
the causal ablation experiment — this repo's own numbers are a concrete
counterexample to trusting DLA ranking alone.**

### Is MLP importance concentrated in a few neurons, or spread out?

Concentrated. Ranking the 256 MLP neurons by mean activation magnitude on the
training set and mean-ablating the top 20:

| intervention | test accuracy |
| --- | ---: |
| top 20 neurons (of 256) by activation magnitude | 64.04% |
| random 20 neurons, 5 seeds | 98.6% / 98.7% / 98.5% / 97.7% / 97.7% |

Removing the top 20 (7.8% of neurons) costs ~35 points; removing a random,
equal-sized set costs about 1-2 points, consistently across 5 seeds. That's a real,
non-trivial concentration of importance — but a milder one than the attention-head
result above, where losing a *single* head (1 of 4) was already catastrophic. The MLP
computation here looks more distributed than the attention circuit.

### Does exact per-neuron DLA agree with the activation-magnitude proxy above?

Only partially, and — like the head-level DLA-vs-ablation result above — the
disagreement is the more informative finding. `neuron_direct_logit_attribution`
(`circuit_lab/interp.py`) extends the exact decomposition down to individual
post-GELU neurons: because `mlp_out = mlp_post @ W_out + b_out`, summing each
neuron's contribution and the bias term reconstructs the MLP's logit
contribution exactly (`tests/test_interp.py` checks this to `1e-4`), the same
correctness bar the head-level DLA is held to.

Ranking all 256 neurons by exact DLA to the correct logit and taking the top 20
gives a set that overlaps the magnitude-proxy's top 20 in only **9 of 20
neurons (45%)** — the two rankings mostly disagree about which neurons matter.
Ablating each top-20 set separately:

| intervention (mean-ablated) | test accuracy | vs. baseline 99.24% |
| --- | ---: | ---: |
| top 20 by activation magnitude | 64.04% | -35.2 pts |
| top 20 by exact DLA | 73.55% | -25.7 pts |
| random 20, 5 seeds | 97.7%-98.7% | -0.5 to -1.5 pts |

Both are far more damaging than a random set, so both proxies are finding
*real* structure — but the cheap magnitude proxy, not the theoretically
cleaner exact-DLA ranking, picks the more causally important set here. That's
the opposite of what "DLA is the principled version of a magnitude heuristic"
would predict, and it's consistent with the head-level result: DLA measures a
component's *average* alignment with the correct-logit direction, which is not
the same thing as how much the network's output depends on that component
being present. A neuron can have a large, consistent average contribution
(high DLA) while being redundant with others that point the same way, and a
neuron with modest average contribution can still be load-bearing if nothing
else covers the frequency it participates in. **The practical lesson carries
over unchanged from the head-level analysis: neither DLA nor a raw-magnitude
heuristic is a substitute for actually ablating and re-measuring — this is
the second independent piece of evidence in this repo for that conclusion,
not a restatement of the first.**

## Status / next steps

Implemented: full training loop with grokking-inducing weight decay, exact direct logit
attribution at both the attention-head and individual-MLP-neuron level, mean-ablation causal
validation at both levels, and Fourier analysis of the learned embeddings. The three questions
posed in "Why this problem" above are all answered with real numbers, not asserted, and the
exact-DLA-vs-magnitude-proxy check flagged as a next step in an earlier version of this section
is now done (see "Does exact per-neuron DLA agree with the activation-magnitude proxy above?").

Open threads a future run could pick up: (1) this repo only studies modular *addition* —
Nanda et al.'s follow-on work also covers subtraction and other group operations, which would
need a different circuit (and might not show the same clean Fourier structure) to describe;
(2) no sparse autoencoder is trained on the MLP activations here — an SAE could test whether
the "sparse Fourier" story is the *complete* picture of what the neurons represent, or whether
there's structure the raw-activation analysis in this repo is missing; (3) the DLA-vs-ablation
mismatch found at both the head and neuron level suggests a natural follow-up: does *iterative*
ablation (remove the single most damaging component, re-rank, repeat) converge on a
smaller/different set than either one-shot ranking, given that components can be redundant with
each other in ways a static ranking can't see?

## License

MIT — see [LICENSE](LICENSE).
