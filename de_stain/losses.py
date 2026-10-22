"""Morphology consistency and conditional staining objectives."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .stain import dab_mask, decompose_dab


def unwrap_module(module):
    return module.module if hasattr(module, "module") else module


@torch.no_grad()
def extract_morphology_descriptor(reference_encoder, images, layer_ids):
    """Concatenate pooled shallow/deep features of the frozen source encoder."""
    features = unwrap_module(reference_encoder)(images, list(layer_ids), encode_only=True)
    return F.normalize(torch.cat([f.mean((2, 3)) for f in features], dim=1), dim=1)


def compute_l_mpi(netG, view_1, view_2, mpi_layers: Sequence[int]):
    """Compute mean cosine distance over pooled ResNet block features."""
    if not mpi_layers:
        raise ValueError("MPI requires at least one ResNet block.")
    features_1 = netG(view_1, list(mpi_layers), encode_only=True)
    features_2 = netG(view_2, list(mpi_layers), encode_only=True)
    distances = [
        1.0 - F.cosine_similarity(f1.mean((2, 3)), f2.mean((2, 3)), dim=1)
        for f1, f2 in zip(features_1, features_2)
    ]
    return torch.stack(distances).mean()


def soft_histogram(values, bins=64, sigma=0.02, chunk_size=8192):
    """Gaussian soft assignments to equal-width bins on normalized [0, 1]."""
    if bins < 2 or sigma <= 0 or chunk_size < 1:
        raise ValueError("bins >= 2, sigma > 0 and chunk_size > 0 are required.")
    values = values.reshape(-1).clamp(0.0, 1.0)
    if values.numel() == 0:
        raise ValueError("Cannot form a tissue histogram from zero pixels.")
    centers = (torch.arange(bins, device=values.device, dtype=values.dtype) + 0.5) / bins
    counts = values.new_zeros(bins)
    for chunk in values.split(chunk_size):
        logits = -0.5 * ((chunk[:, None] - centers[None, :]) / sigma).square()
        counts = counts + logits.softmax(dim=1).sum(dim=0)
    return counts / counts.sum().clamp_min(torch.finfo(counts.dtype).eps)


def histogram_wasserstein_1(histogram, prior, bin_width=None):
    """Compute the one-dimensional Wasserstein distance between histograms."""
    if histogram.shape != prior.shape or histogram.ndim < 1:
        raise ValueError("Histograms must have equal non-empty shapes.")
    width = 1.0 / histogram.shape[-1] if bin_width is None else bin_width
    return ((histogram.cumsum(-1) - prior.cumsum(-1)).abs().sum(-1) * width).mean()


def compute_l_csp(fake_b, target_feat, memory_bank, sample_indices=None):
    """Equal-average DAB histograms of Top-K neighbours, excluding self."""
    prior, neighbours = memory_bank.query_histograms(target_feat, sample_indices)
    dab = decompose_dab(fake_b)
    masks = dab_mask(dab, memory_bank.dab_threshold, memory_bank.dab_dilation_radius)
    normalized = ((dab - memory_bank.dab_v_min) / memory_bank.dab_range).clamp(0.0, 1.0)
    losses = []
    for i in range(fake_b.shape[0]):
        pixels = normalized[i][masks[i]]
        if pixels.numel() == 0:
            pixels = normalized[i].reshape(-1)
        histogram = soft_histogram(pixels, memory_bank.hist_bins, memory_bank.hist_sigma)
        losses.append(histogram_wasserstein_1(histogram, prior[i]))
    return torch.stack(losses).mean(), prior, neighbours


def backward_decoupled_losses(shared_loss, decoder_loss, decoder_parameters):
    """Backpropagate ASP+MPI normally; accumulate CSP only on decoder params.

    Both terms use the same generated image. Explicit parameter gradients
    do not accumulate any CSP contribution into encoder parameters.
    """
    shared_loss.backward(retain_graph=decoder_loss is not None)
    if decoder_loss is None:
        return
    parameters = tuple(p for p in decoder_parameters if p.requires_grad)
    if not parameters:
        raise ValueError("CSP requires trainable decoder parameters.")
    gradients = torch.autograd.grad(decoder_loss, parameters)
    for parameter, gradient in zip(parameters, gradients):
        if parameter.grad is None:
            parameter.grad = gradient.detach()
        else:
            parameter.grad.add_(gradient.detach())
