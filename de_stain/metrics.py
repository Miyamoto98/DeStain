from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "asp"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from util.fid import get_activations
from util.inception import InceptionV3
from util.kid_score import polynomial_mmd_averages


def safe_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    cov = sigma1 @ sigma2
    covmean = linalg.sqrtm(cov)

    if not np.isfinite(covmean).all():
        off = np.eye(sigma1.shape[0]) * eps
        cov = (sigma1 + off) @ (sigma2 + off)
        covmean = linalg.sqrtm(cov)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0.0, atol=1e-3):
            raise ValueError("FID covariance square root has a material imaginary component.")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_fid_kid(real_dir, fake_dir, batch_size=10, kid_subsets=100):
    real_ps = sorted(glob.glob(os.path.join(real_dir, "*.png")) + glob.glob(os.path.join(real_dir, "*.jpg")))
    fake_ps = sorted(glob.glob(os.path.join(fake_dir, "*.png")) + glob.glob(os.path.join(fake_dir, "*.jpg")))

    if len(real_ps) == 0 or len(fake_ps) == 0:
        return None, None, None
    if len(real_ps) < 2 or len(fake_ps) < 2:
        return None, None, None

    try:
        n = min(len(real_ps), len(fake_ps))
        real_ps = real_ps[:n]
        fake_ps = fake_ps[:n]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dims = 2048
        block = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        fid_model = InceptionV3([block], use_fid_inception=True).to(device)
        fid_model.eval()
        act_r = get_activations(real_ps, fid_model, batch_size=batch_size, dims=dims, device=device, num_workers=0)
        act_f = get_activations(fake_ps, fid_model, batch_size=batch_size, dims=dims, device=device, num_workers=0)
        mu_r, s_r = np.mean(act_r, axis=0), np.cov(act_r, rowvar=False)
        mu_f, s_f = np.mean(act_f, axis=0), np.cov(act_f, rowvar=False)
        fid = safe_fid(mu_r, s_r, mu_f, s_f)
        del fid_model

        # Reuse the same FID Inception features for both distribution metrics.
        act_r_k, act_f_k = act_r, act_f
        kv = polynomial_mmd_averages(
            act_r_k,
            act_f_k,
            n_subsets=kid_subsets,
            subset_size=min(100, len(act_r_k), len(act_f_k)),
        )
        kid_mean = float(kv[0].mean())
        kid_std = float(kv[0].std())
        torch.cuda.empty_cache()

        return fid, kid_mean, kid_std
    except Exception as exc:  # pragma: no cover - evaluation fallback
        print(f"[WARN] Skipping FID/KID evaluation: {exc}")
        return None, None, None


# DAB-channel distribution metrics.

# Standard H-DAB stain vectors (Ruifrok & Johnston, Anal Quant Cytol Histol 2001)
_H_DAB_STAIN = torch.tensor(
    [[0.650, 0.704, 0.286],   # Hematoxylin
     [0.268, 0.570, 0.776]],  # DAB
    dtype=torch.float32,
)

_EPS = 1e-8
_N_BINS = 256


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / (matrix.norm(dim=1, keepdim=True) + _EPS)


def extract_dab_channel(rgb: np.ndarray) -> np.ndarray:
    """Extract DAB concentration map from an IHC RGB image via colour deconvolution.

    Parameters
    ----------
    rgb : np.ndarray  (H, W, 3) float32 in [0, 1]

    Returns
    -------
    dab : np.ndarray  (H, W) float32 — DAB optical density (concentration) map
    """
    stain = _normalize_rows(_H_DAB_STAIN.clone().float())
    pinv = torch.linalg.pinv(stain).cpu().numpy()  # (3, 2)

    od = -np.log(np.clip(rgb, _EPS, 1.0))          # (H, W, 3)
    flat = od.reshape(-1, 3)                        # (N, 3)
    conc = flat @ pinv                               # (N, 2)
    dab = conc[:, 1].reshape(rgb.shape[0], rgb.shape[1])  # DAB channel (column 1)
    return dab


def _dab_to_gray_levels(
    dab: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Quantize a continuous DAB map to 256 discrete gray levels [0, 255].

    All 256 levels are guaranteed to be used (no empty bins), since the
    normalization maps the data range exactly onto [0, 255].
    """
    if hi <= lo:
        return np.zeros(dab.shape, dtype=np.uint8)
    # Clip to [lo, hi] then linearly map to [0, 255]
    clipped = np.clip(dab, lo, hi)
    gray = ((clipped - lo) / (hi - lo) * 255.0)
    return gray.round().clip(0, 255).astype(np.uint8)


def _image_to_dab_gray_histogram(
    img_path: str,
    global_lo: float,
    global_hi: float,
) -> np.ndarray:
    """Load an IHC image, extract DAB, quantize to 256 gray levels, histogram.

    Uses a fixed global [lo, hi] range for all images so histograms are
    directly comparable.

    Returns
    -------
    hist : np.ndarray  (256,) float64 — normalised histogram (sum = 1)
    """
    rgb = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    dab = extract_dab_channel(rgb)
    gray = _dab_to_gray_levels(dab, global_lo, global_hi)

    # bincount over 256 gray levels — every pixel contributes to exactly one bin
    counts = np.bincount(gray.ravel(), minlength=_N_BINS)
    hist = counts.astype(np.float64)
    hist = hist / (hist.sum() + _EPS)
    return hist


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = _EPS) -> float:
    """Kullback–Leibler divergence KL(P || Q) with Laplace smoothing."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def dabkl_single(
    real_path: str,
    fake_path: str,
    global_lo: float,
    global_hi: float,
) -> dict:
    """Compute DABKL metrics for a single paired (real, fake) IHC image.

    Parameters
    ----------
    real_path  : str   — path to real IHC (ground truth) image
    fake_path  : str   — path to generated IHC image
    global_lo  : float — global lower bound for gray-level quantization
    global_hi  : float — global upper bound for gray-level quantization

    Returns
    -------
    dict with keys:
        kl_real_fake  : KL(P_real || P_fake)  — primary metric
        kl_fake_real  : KL(P_fake || P_real)  — reverse direction
        js_distance   : Jensen–Shannon distance ∈ [0, 1]  — symmetric
        hist_real     : (256,) normalised gray-level histogram of real DAB
        hist_fake     : (256,) normalised gray-level histogram of fake DAB
    """
    hist_r = _image_to_dab_gray_histogram(real_path, global_lo, global_hi)
    hist_f = _image_to_dab_gray_histogram(fake_path, global_lo, global_hi)

    kl_rf = _kl_divergence(hist_r, hist_f)
    kl_fr = _kl_divergence(hist_f, hist_r)
    js = float(jensenshannon(hist_r, hist_f))

    return {
        "kl_real_fake": kl_rf,
        "kl_fake_real": kl_fr,
        "js_distance": js,
        "hist_real": hist_r,
        "hist_fake": hist_f,
    }


def _compute_global_dab_range(
    img_paths: list[str],
    lo_percentile: float = 0.5,
    hi_percentile: float = 99.5,
    max_samples: int = 200,
    random_seed: int | None = None,
) -> tuple[float, float]:
    """Estimate global DAB [lo, hi] range from a sample of real IHC images.

    Uses percentiles of DAB values pooled across randomly sampled images
    to get a stable, outlier-robust global range for gray-level quantization.
    """
    import random

    sample_paths = img_paths[:]
    if len(sample_paths) > max_samples:
        rng = random.Random(random_seed)
        sample_paths = rng.sample(sample_paths, max_samples)

    all_dab_vals = []
    for p in sample_paths:
        rgb = np.array(Image.open(p).convert("RGB")).astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        dab = extract_dab_channel(rgb)
        all_dab_vals.append(dab.ravel())

    pooled = np.concatenate(all_dab_vals)
    global_lo = float(np.percentile(pooled, lo_percentile))
    global_hi = float(np.percentile(pooled, hi_percentile))
    return global_lo, global_hi


def compute_dabkl(
    real_dir: str,
    fake_dir: str,
    num_images: int | None = None,
    verbose: bool = True,
    lo_percentile: float = 0.5,
    hi_percentile: float = 99.5,
    random_seed: int | None = None,
) -> dict:
    """Compute DABKL metrics over a directory of paired real/fake IHC images.

    Quantizes DAB channel to 256 discrete gray levels using a global
    percentile range computed from all real images. Every image pair
    uses the same [lo, hi] → [0, 255] mapping, so KL values are
    directly comparable across pairs and experiments.

    Parameters
    ----------
    real_dir       : str  — directory of real IHC images (ground truth)
    fake_dir       : str  — directory of generated IHC images
    num_images     : int | None — limit to first N paired images (None = all)
    verbose        : bool — print progress
    lo_percentile  : float — low percentile for global DAB range (default 0.5)
    hi_percentile  : float — high percentile for global DAB range (default 99.5)
    random_seed    : int | None — runtime state for real-image subsampling

    Returns
    -------
    dict with keys:
        dabkl_mean        : mean KL(P_real || P_fake) across all pairs
        dabkl_std         : std  KL(P_real || P_fake) across all pairs
        dabkl_reverse_mean: mean KL(P_fake || P_real)
        dabkl_reverse_std : std  KL(P_fake || P_real)
        js_distance_mean  : mean Jensen–Shannon distance
        js_distance_std   : std  Jensen–Shannon distance
        n_pairs           : number of paired images evaluated
        global_dab_lo     : global DAB lower bound used
        global_dab_hi     : global DAB upper bound used
    """
    real_ps = sorted(
        glob.glob(os.path.join(real_dir, "*.png"))
        + glob.glob(os.path.join(real_dir, "*.jpg"))
        + glob.glob(os.path.join(real_dir, "*.jpeg"))
    )
    fake_ps = sorted(
        glob.glob(os.path.join(fake_dir, "*.png"))
        + glob.glob(os.path.join(fake_dir, "*.jpg"))
        + glob.glob(os.path.join(fake_dir, "*.jpeg"))
    )

    if len(real_ps) == 0:
        raise FileNotFoundError(f"No images found in real_dir: {real_dir}")
    if len(fake_ps) == 0:
        raise FileNotFoundError(f"No images found in fake_dir: {fake_dir}")

    # Pair by filename
    real_names = {os.path.basename(p): p for p in real_ps}
    fake_names = {os.path.basename(p): p for p in fake_ps}
    common = sorted(set(real_names.keys()) & set(fake_names.keys()))

    if len(common) == 0:
        raise ValueError(
            f"No matching filenames between real_dir ({len(real_ps)} images) "
            f"and fake_dir ({len(fake_ps)} images)"
        )

    if num_images is not None:
        common = common[:num_images]

    n = len(common)

    # --- Pre-compute global DAB range from real images -----------------------
    if verbose:
        print(f"[DABKL] Computing global DAB range from real images "
              f"(P_{lo_percentile}–P_{hi_percentile})...")
    all_real_paths = [real_names[name] for name in common]
    global_lo, global_hi = _compute_global_dab_range(
        all_real_paths,
        lo_percentile=lo_percentile,
        hi_percentile=hi_percentile,
        random_seed=random_seed,
    )
    if verbose:
        print(f"[DABKL] Global DAB range: [{global_lo:.4f}, {global_hi:.4f}]")

    if verbose:
        print(f"[DABKL] Evaluating {n} paired images (256 gray-level quantisation)...")

    kl_rf_vals = []
    kl_fr_vals = []
    js_vals = []

    for i, name in enumerate(common):
        try:
            res = dabkl_single(real_names[name], fake_names[name], global_lo, global_hi)
            kl_rf_vals.append(res["kl_real_fake"])
            kl_fr_vals.append(res["kl_fake_real"])
            js_vals.append(res["js_distance"])
        except Exception as exc:
            if verbose:
                print(f"[WARN] DABKL failed for {name}: {exc}")
            continue

        if verbose and (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{n}")

    if len(kl_rf_vals) == 0:
        raise RuntimeError("DABKL computation failed for all image pairs.")

    kl_arr = np.array(kl_rf_vals)
    kl_rev_arr = np.array(kl_fr_vals)
    js_arr = np.array(js_vals)

    result = {
        "dabkl_mean": float(kl_arr.mean()),
        "dabkl_std": float(kl_arr.std()),
        "dabkl_reverse_mean": float(kl_rev_arr.mean()),
        "dabkl_reverse_std": float(kl_rev_arr.std()),
        "js_distance_mean": float(js_arr.mean()),
        "js_distance_std": float(js_arr.std()),
        "n_pairs": len(kl_rf_vals),
        "global_dab_lo": global_lo,
        "global_dab_hi": global_hi,
    }

    if verbose:
        print(f"  DABKL (KL real→fake): {result['dabkl_mean']:.4f} ± {result['dabkl_std']:.4f}")
        print(f"  DABKL (KL fake→real): {result['dabkl_reverse_mean']:.4f} ± {result['dabkl_reverse_std']:.4f}")
        print(f"  JS distance:           {result['js_distance_mean']:.4f} ± {result['js_distance_std']:.4f}")

    return result
