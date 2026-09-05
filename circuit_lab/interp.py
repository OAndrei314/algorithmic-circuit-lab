"""Interpretability tools for :class:`circuit_lab.model.OneLayerTransformer`.

Three techniques, all standard in the mechanistic-interpretability literature:

* **Direct logit attribution (DLA)** -- because the model has no LayerNorm,
  the final-position logits are an *exact* sum of independent per-component
  contributions (embedding skip connection, each attention head, the MLP).
  That additivity is what makes DLA exact here rather than a rough
  first-order approximation.
* **Activation patching / ablation** -- the causal ground truth. DLA tells
  you what *correlates* with the correct logit; ablating a component and
  re-measuring accuracy tells you what the model *actually depends on*.
* **Fourier analysis of the embedding matrix** -- the known signature of a
  generalizing modular-addition solution (Nanda et al. 2023) is that each
  token's embedding, viewed as a function of the token's integer value, is
  well-approximated by a handful of low-frequency sinusoids rather than
  being a dense, unstructured vector.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from circuit_lab.data import ModularAdditionDataset
from circuit_lab.model import OneLayerTransformer


@torch.no_grad()
def compute_reference_means(model: OneLayerTransformer, inputs: torch.Tensor):
    """Mean activations over a reference distribution (typically the train
    set), used for mean-ablation."""
    _, cache = model(inputs, return_cache=True)
    mean_z = cache["z"].mean(dim=0)  # (n_ctx, n_heads, d_head)
    mean_mlp_post = cache["mlp_post"].mean(dim=0)  # (n_ctx, d_mlp)
    return mean_z, mean_mlp_post


@torch.no_grad()
def direct_logit_attribution(
    model: OneLayerTransformer, inputs: torch.Tensor, labels: torch.Tensor
):
    """Decompose the correct-answer logit at the final position into the
    direct-path (embedding skip connection), per-head attention, and MLP
    contributions. Returns a dict of per-example contributions (batch,) and
    per-head contributions (batch, n_heads); these are exact, i.e. direct +
    attn.sum(-1) + mlp == the model's actual correct-class logit.
    """
    _, cache = model(inputs, return_cache=True)
    batch = torch.arange(inputs.shape[0])

    direct = cache["resid_pre"][:, -1, :] @ model.W_U  # (batch, vocab)
    mlp = cache["mlp_out"][:, -1, :] @ model.W_U  # (batch, vocab)

    z_final = cache["z"][:, -1, :, :]  # (batch, n_heads, d_head)
    per_head = torch.einsum("bhd,hdm,mv->bhv", z_final, model.W_O, model.W_U)

    full_logits = cache["logits"][:, -1, :]

    return {
        "direct_correct": direct[batch, labels],
        "mlp_correct": mlp[batch, labels],
        "per_head_correct": per_head[batch, :, labels],  # (batch, n_heads)
        "full_correct": full_logits[batch, labels],
    }


@torch.no_grad()
def ablation_accuracy(
    model: OneLayerTransformer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    ablate_heads: Optional[list] = None,
    ablate_neurons: Optional[list] = None,
    mean_z: Optional[torch.Tensor] = None,
    mean_mlp_post: Optional[torch.Tensor] = None,
) -> float:
    """Accuracy after mean-ablating the given heads/neurons. With no
    ablation targets this is just ordinary accuracy."""
    logits = model(
        inputs,
        ablate_heads=ablate_heads,
        ablate_neurons=ablate_neurons,
        mean_z=mean_z,
        mean_mlp_post=mean_mlp_post,
    )
    preds = logits[:, -1, :].argmax(dim=-1)
    return (preds == labels).float().mean().item()


@dataclass
class FourierSpectrum:
    power: torch.Tensor  # (p//2 + 1,) power per frequency, normalized to sum to 1
    top_frequencies: list  # frequency indices sorted by power, descending


def fourier_power_spectrum(model: OneLayerTransformer, p: int) -> FourierSpectrum:
    """Power spectrum of the numeric token embeddings (tokens 0..p-1, i.e.
    excluding the '=' token) as a function of token value, summed across
    embedding dimensions.
    """
    embed = model.W_E[:p, :].detach()  # (p, d_model)
    coeffs = torch.fft.rfft(embed, dim=0)  # (p//2+1, d_model)
    power = (coeffs.abs() ** 2).sum(dim=-1)
    power = power / power.sum()
    order = torch.argsort(power, descending=True).tolist()
    return FourierSpectrum(power=power, top_frequencies=order)


def fraction_of_power_in_top_k(spectrum: FourierSpectrum, k: int) -> float:
    return spectrum.power[spectrum.top_frequencies[:k]].sum().item()


def rank_components_by_dla(
    model: OneLayerTransformer, inputs: torch.Tensor, labels: torch.Tensor
):
    """Rank attention heads by mean direct logit attribution to the correct
    class, as a cheap correlational proxy for importance (to be checked
    against the causal ablation results, not trusted on its own)."""
    dla = direct_logit_attribution(model, inputs, labels)
    per_head_mean = dla["per_head_correct"].mean(dim=0)  # (n_heads,)
    order = torch.argsort(per_head_mean, descending=True).tolist()
    return order, per_head_mean.tolist()


@torch.no_grad()
def neuron_direct_logit_attribution(
    model: OneLayerTransformer, inputs: torch.Tensor, labels: torch.Tensor
):
    """Decompose the MLP's contribution to the correct-answer logit into an
    exact per-neuron term plus a constant bias term.

    ``mlp_out = mlp_post @ W_out + b_out``, i.e. the MLP's output is itself
    an exact linear combination of the *post-GELU* neuron activations --
    summing per-neuron contributions is still exact here, unlike attributing
    importance from the pre-activation, because the nonlinearity has already
    been applied by this point. So ``per_neuron_correct.sum(-1) +
    bias_correct`` reconstructs ``direct_logit_attribution(...)["mlp_correct"]``
    exactly, the same way the head-level decomposition reconstructs the full
    logit.
    """
    _, cache = model(inputs, return_cache=True)
    batch = torch.arange(inputs.shape[0])

    mlp_post_final = cache["mlp_post"][:, -1, :]  # (batch, d_mlp)
    neuron_to_logit = model.W_out @ model.W_U  # (d_mlp, vocab)
    per_neuron = mlp_post_final * neuron_to_logit[:, labels].T  # (batch, d_mlp)

    bias_to_logit = model.b_out @ model.W_U  # (vocab,)
    bias_correct = bias_to_logit[labels]

    return {
        "per_neuron_correct": per_neuron,
        "bias_correct": bias_correct,
    }


def rank_neurons_by_dla(
    model: OneLayerTransformer, inputs: torch.Tensor, labels: torch.Tensor
):
    """Rank MLP neurons by mean direct logit attribution to the correct
    class -- the exact-decomposition analogue of the cheap activation-
    magnitude proxy used for neuron importance elsewhere, so the two can be
    checked against each other."""
    dla = neuron_direct_logit_attribution(model, inputs, labels)
    per_neuron_mean = dla["per_neuron_correct"].mean(dim=0)  # (d_mlp,)
    order = torch.argsort(per_neuron_mean, descending=True).tolist()
    return order, per_neuron_mean.tolist()


@torch.no_grad()
def greedy_iterative_ablation(
    model: OneLayerTransformer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    candidates: list,
    ablate_kind: str,
    mean_z: Optional[torch.Tensor] = None,
    mean_mlp_post: Optional[torch.Tensor] = None,
):
    """Greedily and cumulatively ablate whichever remaining candidate does
    the most damage *given what has already been removed*, one at a time.

    This is a different question from ranking components once (by DLA or
    activation magnitude) and ablating that whole set together: a one-shot
    ranking asks "how important does each component look in isolation, on
    top of the intact model?", which can't see redundancy -- two components
    that each look load-bearing alone might become interchangeable once one
    of them is already gone. Greedy re-evaluates every remaining candidate
    against the *current* (partially-ablated) model at each step, so it can
    surface that kind of redundancy a static ranking cannot.

    Returns a list of ``(component_index, accuracy_after_removing_it)`` in
    removal order -- a permutation of ``candidates`` paired with the
    cumulative accuracy trajectory as each one is removed.
    """
    if ablate_kind not in ("heads", "neurons"):
        raise ValueError(f"ablate_kind must be 'heads' or 'neurons', got {ablate_kind!r}")

    remaining = list(candidates)
    already_ablated = []
    trajectory = []
    while remaining:
        best_component, best_acc = None, None
        for c in remaining:
            trial = already_ablated + [c]
            if ablate_kind == "heads":
                acc = ablation_accuracy(model, inputs, labels, ablate_heads=trial, mean_z=mean_z)
            else:
                acc = ablation_accuracy(
                    model, inputs, labels, ablate_neurons=trial, mean_mlp_post=mean_mlp_post
                )
            if best_acc is None or acc < best_acc:
                best_component, best_acc = c, acc
        already_ablated.append(best_component)
        remaining.remove(best_component)
        trajectory.append((best_component, best_acc))
    return trajectory
