"""Image-specific Macenko H/E perturbations and H-DAB deconvolution."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6
# H/E fallback for blank or rank-deficient images; separate from H-DAB.
HE_BASIS = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]], dtype=torch.float32)
HDAB_BASIS = torch.tensor([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776]], dtype=torch.float32)


def _rgb(image):
    return ((image + 1.0) * 0.5).clamp(EPS, 1.0)


def dab_mask(
    dab,
    threshold=0.0,  # Set a dataset-specific DAB threshold before use.
    radius=2,
):
    """Threshold a (B,H,W) DAB map and dilate with a circular neighbourhood."""
    if dab.ndim != 3 or radius < 0:
        raise ValueError("Expected a (B,H,W) DAB map and a non-negative radius.")
    mask = dab > threshold
    if radius == 0:
        return mask
    coordinates = torch.arange(-radius, radius + 1, device=dab.device)
    disk = (coordinates[:, None].square() + coordinates[None, :].square()) <= radius ** 2
    expanded = F.conv2d(mask[:, None].float(), disk[None, None].float(), padding=radius)
    return expanded[:, 0] > 0


def estimate_stain_matrix(image, alpha=1.0, beta=0.15):
    """Estimate image-specific H/E extinction vectors in Macenko's OD plane."""
    if torch.is_tensor(image):
        image = image.detach().cpu().float().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    od = -np.log(np.clip(rgb, EPS, 1.0)).reshape(-1, 3)
    tissue = od[np.max(od, axis=1) > beta]
    fallback = F.normalize(HE_BASIS, dim=1)
    if tissue.shape[0] < 10 or np.linalg.matrix_rank(tissue) < 2:
        return fallback.clone()
    _, vectors = np.linalg.eigh(np.cov(tissue, rowvar=False))
    plane = vectors[:, -2:]
    for column in range(2):
        if plane[:, column].sum() < 0:
            plane[:, column] *= -1
    projected = tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [alpha, 100.0 - alpha])
    basis = np.stack([plane @ [np.cos(low), np.sin(low)],
                      plane @ [np.cos(high), np.sin(high)]])
    basis = np.where(basis.sum(axis=1, keepdims=True) < 0, -basis, basis)
    basis = basis[np.argsort(-basis[:, 0])].copy()
    if not np.isfinite(basis).all() or np.linalg.matrix_rank(basis) < 2:
        return fallback.clone()
    return F.normalize(torch.tensor(basis, dtype=torch.float32), dim=1)


@torch.no_grad()
def decompose_he_batch(images):
    rgb = _rgb(images)
    matrices = torch.stack([estimate_stain_matrix(image) for image in rgb]).to(rgb)
    od = -rgb.log().permute(0, 2, 3, 1)
    concentrations = torch.einsum(
        "bhwc,bcd->bhwd", od, torch.linalg.pinv(matrices)
    ).clamp_min(0.0)
    return concentrations.permute(0, 3, 1, 2), matrices


def _reconstruct(concentrations, matrices):
    od = torch.einsum("bdhw,bdc->bchw", concentrations, matrices)
    return (-od).exp().clamp(0.0, 1.0) * 2.0 - 1.0


@torch.no_grad()
def augment_two_stain_views(images, strength=0.2):
    """Apply independent H/E concentration scaling with the original stain basis."""
    if not 0.0 <= strength < 1.0:
        raise ValueError("OD perturbation strength must be in [0, 1).")
    concentrations, matrices = decompose_he_batch(images)
    views = []
    for _ in range(2):
        factors = 1.0 - strength + 2.0 * strength * torch.rand(
            images.shape[0], 2, 1, 1, device=images.device, dtype=images.dtype
        )
        views.append(_reconstruct(concentrations * factors, matrices))
    return views[0], views[1]


def decompose_dab(images):
    """Differentiable H-DAB color deconvolution, returning (B,H,W)."""
    rgb = _rgb(images)
    basis = F.normalize(HDAB_BASIS.to(rgb), dim=1)
    concentrations = torch.einsum(
        "bhwc,cd->bhwd", -rgb.log().permute(0, 2, 3, 1), torch.linalg.pinv(basis)
    )
    return concentrations[..., 1]
