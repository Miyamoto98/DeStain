"""Frozen morphology-to-staining memory bank."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .losses import extract_morphology_descriptor, soft_histogram
from .stain import dab_mask, decompose_dab


@dataclass
class TargetMemoryBank:
    morphology: torch.Tensor
    histograms: torch.Tensor
    indices: torch.Tensor
    dab_v_min: float
    dab_v_max: float
    hist_bins: int = 64
    hist_sigma: float = 0.02
    top_k: int = 10
    dab_threshold: float = 0.0
    dab_dilation_radius: int = 2

    @property
    def dab_range(self):
        return max(self.dab_v_max - self.dab_v_min, 1e-8)

    @property
    def num_entries(self):
        return self.indices.numel()

    @classmethod
    @torch.no_grad()
    def build(cls, reference_encoder, loader, descriptor_layers, device,
              hist_bins=64, hist_sigma=0.02, top_k=10,
              dab_threshold=0.0, dab_dilation_radius=2):
        reference_encoder.eval()
        keys, pixels, indices = [], [], []
        for batch in loader:
            a, b = batch["A"].to(device), batch["B"].to(device)
            keys.append(extract_morphology_descriptor(reference_encoder, a, descriptor_layers).cpu())
            dab = decompose_dab(b)
            masks = dab_mask(dab, dab_threshold, dab_dilation_radius)
            for i in range(b.shape[0]):
                values = dab[i][masks[i]].detach().cpu()
                if values.numel() == 0:
                    # Use the full DAB map when no pixels pass the threshold.
                    values = dab[i].detach().cpu().reshape(-1)
                pixels.append(values)
            indices.append(torch.as_tensor(batch["index"]).view(-1).cpu())
        if len(pixels) < 2:
            raise ValueError("CSP memory must contain at least two training pairs.")
        # Use only target training values to fix a shared DAB scale.
        pooled = torch.cat(pixels).numpy()
        low, high = np.percentile(pooled, [0.1, 99.9])
        scale = max(float(high - low), 1e-8)
        histograms = torch.stack([
            soft_histogram(((values.to(device) - float(low)) / scale).clamp(0.0, 1.0),
                           bins=hist_bins, sigma=hist_sigma)
            for values in pixels
        ])
        return cls(torch.cat(keys).to(device), histograms.detach(),
                   torch.cat(indices).long().to(device), float(low), float(high),
                   hist_bins, hist_sigma, top_k, dab_threshold, dab_dilation_radius)

    @torch.no_grad()
    def query_histograms(self, morphology, sample_indices=None):
        query = F.normalize(morphology, dim=1)
        similarities = query @ self.morphology.to(query).t()
        bank_ids = self.indices.to(query.device)
        if sample_indices is not None:
            sample_indices = torch.as_tensor(sample_indices, device=query.device).reshape(-1)
            if sample_indices.numel() != query.shape[0]:
                raise ValueError("One sample index is required for each query.")
            similarities.masked_fill_(sample_indices[:, None] == bank_ids[None, :], -torch.inf)
        priors, neighbours = [], []
        for row in similarities:
            # Filter before topk: N=10,K=10 must still exclude the query itself.
            valid = torch.nonzero(torch.isfinite(row), as_tuple=False).flatten()
            if valid.numel() == 0:
                raise ValueError("No non-self memory entries are available.")
            count = min(self.top_k, valid.numel())
            selected = valid[torch.topk(row[valid], k=count).indices]
            # Mean of separately normalized neighbour histograms, not pooled pixels.
            priors.append(self.histograms.to(query.device)[selected].mean(dim=0))
            neighbours.append(bank_ids[selected])
        return torch.stack(priors), neighbours

    def state_dict(self):
        return {name: value.detach().cpu() if torch.is_tensor(value) else value
                for name, value in vars(self).items()}
