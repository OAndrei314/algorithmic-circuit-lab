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
