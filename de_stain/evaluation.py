"""One evaluation path shared by training and standalone checkpoint evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

def _image(tensor):
    return Image.fromarray(
        ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .round().byte().permute(1, 2, 0).numpy()
    )


@torch.no_grad()
def save_eval_outputs(model, dataset, out_dir, num_workers=0):
    out_dir = Path(out_dir)
    real_dir, fake_dir = out_dir / "real_B", out_dir / "fake_B"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    if any(real_dir.iterdir()) or any(fake_dir.iterdir()):
        raise FileExistsError(f"Evaluation image folders must be empty: {out_dir}")
    modes = {}
    for name in ["netG", "netD", "netF"]:
        network = getattr(model, name, None)
        if network is not None:
            modes[name] = network.training
            network.eval()
    pairs = []
    try:
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False,
                                                num_workers=num_workers)):
            model.set_input(batch)
            model.test()
            filename = f"{index:05d}.png"
            _image(model.real_B[0]).save(real_dir / filename)
            _image(model.fake_B[0]).save(fake_dir / filename)
            pairs.append({"file": filename, "A": batch["A_paths"][0], "B": batch["B_paths"][0]})
    finally:
        for name, training in modes.items():
            getattr(model, name).train(training)
    (out_dir / "image_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return real_dir, fake_dir


def evaluate(model, dataset, out_dir, args):
    from .metrics import compute_dabkl, compute_fid_kid
    real_dir, fake_dir = save_eval_outputs(model, dataset, out_dir, args.num_workers)
    numpy_state = np.random.get_state()
    np.random.seed(args.eval_seed)
    try:
        fid, kid_mean, kid_std = compute_fid_kid(
            str(real_dir), str(fake_dir), args.metric_batch_size, args.kid_subsets
        )
    finally:
        np.random.set_state(numpy_state)
    if fid is None or kid_mean is None:
        raise RuntimeError("FID/KID evaluation failed; no complete metric record was produced.")
    dabkl = compute_dabkl(str(real_dir), str(fake_dir), verbose=False,
                         random_seed=args.eval_seed)
    record = {
        "FID": float(fid), "KID": float(kid_mean), "KID_x100": float(kid_mean * 100),
        "KID_subset_std": float(kid_std), "DABKL": dabkl["dabkl_mean"],
        "n_pairs": len(dataset),
        "DABKL_global_lo": dabkl["global_dab_lo"], "DABKL_global_hi": dabkl["global_dab_hi"],
    }
    if not all(np.isfinite(record[key]) for key in ["FID", "KID", "DABKL"]):
        raise RuntimeError("Non-finite evaluation metric.")
    Path(out_dir, "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return record
