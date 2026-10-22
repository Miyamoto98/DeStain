"""Evaluate the trained DE_STAIN generator using its saved configuration."""
from __future__ import annotations

from pathlib import Path

from de_stain.backbone import build_backbone_options, create_model, load_tensor_state
from de_stain.config import parse_args
from de_stain.data import PairedTargetDataset
from de_stain.evaluation import evaluate
from de_stain.optim import unwrap_module


def main(argv=None):
    args = parse_args(argv, evaluation=True)
    experiment = Path(args.output_root) / args.exp_name
    checkpoint = Path(args.netG_path) if args.netG_path else experiment / f"{args.epoch}_net_G.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained generator: {checkpoint}")
    opt = build_backbone_options(args, is_train=False)
    model = create_model(opt)
    unwrap_module(model.netG).load_state_dict(load_tensor_state(checkpoint, model.device), strict=True)
    test_ds = PairedTargetDataset(
        args.target_dataroot, args.target_test_split, opt, seed=args.eval_seed, phase="test",
        brightness_norm=args.brightness_norm, brightness_percentile=args.brightness_percentile
    )
    results = Path(args.results_dir) if args.results_dir else experiment / f"eval_checkpoint_{args.epoch}"
    evaluate(model, test_ds, results, args)


if __name__ == "__main__":
    main()
