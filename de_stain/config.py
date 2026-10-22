"""Shared configuration for DE_STAIN training and inference."""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.json"


def parse_args(argv=None, evaluation=False):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=str(DEFAULT_CONFIG))
    probe.add_argument("--output_root")
    probe.add_argument("--exp_name")
    probe.add_argument("--target_train_samples", type=int)
    probe.add_argument("--train_mode")
    initial, _ = probe.parse_known_args(argv)
    config_path = Path(initial.config)
    with config_path.open(encoding="utf-8-sig") as stream:
        defaults = json.load(stream)
    if evaluation:
        name_settings = dict(defaults)
        for key in ["target_train_samples", "train_mode"]:
            if getattr(initial, key) is not None:
                name_settings[key] = getattr(initial, key)
        name = initial.exp_name or defaults.get("exp_name") or experiment_name(name_settings)
        snapshot = Path(initial.output_root or defaults["output_root"]) / name / "config.json"
        if snapshot.is_file():
            defaults.update(json.loads(snapshot.read_text(encoding="utf-8"))["settings"])
        defaults["config"] = str(config_path)

    parser = argparse.ArgumentParser(
        description="DE_STAIN" + (" inference" if evaluation else " training")
    )
    parser.add_argument("--config", default=str(config_path))
    strings = ["target_dataroot", "target_train_split", "target_test_split",
               "pretrained_checkpoints_dir", "pretrained_name", "pretrained_epoch",
               "output_root", "exp_name", "gpu_ids", "netG", "netD", "weight_norm",
               "nce_layers", "asp_loss_mode"]
    integers = ["target_train_samples", "batch_size", "num_workers",
                "epochs", "crop_size", "load_size", "n_layers_D", "ngf", "ndf",
                "num_patches", "csp_top_k", "csp_hist_bins", "metric_batch_size",
                "kid_subsets", "eval_every", "save_epoch_freq", "dab_dilation_radius"]
    floats = ["lr", "nce_T", "lambda_NCE", "lambda_GAN", "lambda_gp", "lambda_asp",
              "lambda_mpi", "lambda_csp", "csp_hist_sigma", "stain_strength",
              "brightness_percentile", "dab_threshold"]
    for key in strings:
        parser.add_argument("--" + key, type=str)
    for key in integers:
        parser.add_argument("--" + key, type=int)
    for key in floats:
        parser.add_argument("--" + key, type=float)
    parser.add_argument("--train_mode", choices=["ours", "full_ft", "mpi_only", "csp_only"])
    parser.add_argument("--brightness_norm", action=argparse.BooleanOptionalAction)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip test-set evaluation during training.")
    parser.add_argument("--verbose", action="store_true")
    if evaluation:
        parser.add_argument("--epoch", default="latest")
        parser.add_argument("--netG_path", default=None)
        parser.add_argument("--results_dir", default=None)
    valid = {action.dest for action in parser._actions}
    unexpected = set(defaults) - valid
    if unexpected:
        parser.error("Unknown configuration fields: " + ", ".join(sorted(unexpected)))
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.seed = secrets.randbits(32)
    args.eval_seed = args.seed
    for key in ["target_train_samples", "epochs", "batch_size", "crop_size",
                "csp_top_k", "metric_batch_size", "kid_subsets", "save_epoch_freq"]:
        if getattr(args, key) < 1:
            parser.error(key + " must be positive.")
    if args.target_train_samples < 2 and args.train_mode in {"ours", "csp_only"}:
        parser.error("CSP requires at least two distinct training pairs.")
    if args.batch_size != 1:
        parser.error("batch_size must be 1.")
    if args.csp_hist_bins < 2 or args.csp_hist_sigma <= 0 or args.lr <= 0:
        parser.error("Histogram bins >= 2, sigma > 0 and lr > 0 are required.")
    if args.dab_threshold < 0 or args.dab_dilation_radius < 0:
        parser.error("DAB threshold and dilation radius must be non-negative.")
    if not 0 <= args.stain_strength < 1:
        parser.error("stain_strength must be in [0, 1).")
    if min(args.lambda_mpi, args.lambda_csp) < 0:
        parser.error("Loss weights must be non-negative.")
    if args.train_mode == "full_ft":
        args.lambda_mpi = args.lambda_csp = 0.0
    elif args.train_mode == "mpi_only":
        args.lambda_csp = 0.0
    elif args.train_mode == "csp_only":
        args.lambda_mpi = 0.0
    if args.exp_name is None:
        args.exp_name = experiment_name(vars(args))
    return args


def experiment_name(settings):
    mode = settings.get("train_mode", "ours")
    return f"DE_STAIN_{mode}"
