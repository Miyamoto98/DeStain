"""Paired target data with reproducible draws, crops and portable manifests."""
from __future__ import annotations

import json
import random
import secrets
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def brightness_normalize(image, percentile=95.0):
    """LAB luminosity standardization; preserves white background."""
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(np.float32)
    scale = max(float(np.percentile(lab[..., 0], percentile)), 1.0)
    lab[..., 0] = np.clip(lab[..., 0] * 255.0 / scale, 0.0, 255.0)
    return Image.fromarray(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))


def find_aligned_pairs(root, split):
    root = Path(root)
    layouts = [(root / f"{split}A", root / f"{split}B"),
               (root / "HE" / split, root / "IHC" / split)]
    for directory_a, directory_b in layouts:
        if directory_a.is_dir() and directory_b.is_dir():
            break
    else:
        raise FileNotFoundError(f"Missing {split}A/{split}B or HE/{split} + IHC/{split} under {root}")
    def collect(directory):
        return {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS}
    a, b = collect(directory_a), collect(directory_b)
    if not a or a.keys() != b.keys():
        raise ValueError(f"Expected nonempty, identically named HE/IHC pairs in {directory_a} and {directory_b}.")
    return [(a[key], b[key]) for key in sorted(a)]


class PairedTargetDataset(Dataset):
    def __init__(self, root, split, opt, max_samples=None, seed=None, phase="train",
                 pairs=None, manifest=None, brightness_norm=True,
                 brightness_percentile=95.0):
        self.root = Path(root).resolve()
        self.opt, self.phase = opt, phase
        self.seed = secrets.randbits(32) if seed is None else seed
        self.epoch = 0
        self.brightness_norm = brightness_norm
        self.brightness_percentile = brightness_percentile
        available = find_aligned_pairs(self.root, split) if pairs is None else list(pairs)
        manifest_path = Path(manifest) if manifest else None
        if manifest_path and manifest_path.is_file():
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            if record.get("split") != split:
                raise ValueError("Manifest split differs from the requested data split.")
            lookup = {(a.relative_to(self.root).as_posix(), b.relative_to(self.root).as_posix()): (a, b)
                      for a, b in available}
            try:
                self.pairs = [lookup[(item["A"], item["B"])] for item in record["pairs"]]
            except KeyError as exc:
                raise ValueError("Manifest references a pair absent from the requested training split.") from exc
            if len({str(a) for a, _ in self.pairs}) != len(self.pairs):
                raise ValueError("Manifest contains duplicate training samples.")
            if max_samples and len(self.pairs) != max_samples:
                raise ValueError("Manifest size differs from target_train_samples.")
        else:
            if max_samples and max_samples > len(available):
                raise ValueError(f"Requested {max_samples} pairs; only {len(available)} are available.")
            selected = random.Random(self.seed).sample(available, max_samples) if max_samples else available
            self.pairs = sorted(selected, key=lambda pair: str(pair[0]))
            if manifest_path:
                self.write_manifest(manifest_path, split)

    def write_manifest(self, path, split):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"split": split, "count": len(self.pairs),
                  "pairs": [{"A": a.relative_to(self.root).as_posix(),
                             "B": b.relative_to(self.root).as_posix()} for a, b in self.pairs]}
        text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace an existing different split manifest: {path}")
        path.write_text(text, encoding="utf-8")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_path, b_path = self.pairs[index]
        with Image.open(a_path) as image:
            a = image.convert("RGB")
        with Image.open(b_path) as image:
            b = image.convert("RGB")
        if a.size != b.size:
            raise ValueError(f"Paired image sizes differ: {a_path} and {b_path}")
        size = self.opt.crop_size
        if min(a.size) < size:
            raise ValueError(f"crop_size={size} exceeds image dimensions for {a_path}")
        if self.brightness_norm:
            a = brightness_normalize(a, self.brightness_percentile)
            b = brightness_normalize(b, self.brightness_percentile)
        key = a_path.relative_to(self.root).as_posix()
        epoch = self.epoch if self.phase == "train" else 0
        rng = random.Random(f"{self.seed}:{epoch}:{key}")
        x, y = rng.randrange(a.width - size + 1), rng.randrange(a.height - size + 1)
        box = (x, y, x + size, y + size)
        a, b = a.crop(box), b.crop(box)
        if self.phase == "train" and rng.random() < 0.5:
            a, b = a.transpose(Image.Transpose.FLIP_LEFT_RIGHT), b.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        def tensor(image):
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        return {"A": tensor(a), "B": tensor(b), "A_paths": str(a_path),
                "B_paths": str(b_path), "index": index}
