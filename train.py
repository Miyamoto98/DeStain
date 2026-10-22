"""DE_STAIN training: ASP + direction-only MPI + decoder-only W1 CSP."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from de_stain.backbone import (build_backbone_options, create_model, load_pretrained,
                               make_reference_generator, pretrained_paths, save_networks)
from de_stain.config import parse_args
from de_stain.data import PairedTargetDataset
from de_stain.evaluation import evaluate
from de_stain.losses import (backward_decoupled_losses, compute_l_csp, compute_l_mpi,
                            extract_morphology_descriptor)
from de_stain.memory_bank import TargetMemoryBank
from de_stain.optim import (build_optimizer_D, build_optimizer_F, build_optimizer_G,
                           decoder_parameters, generator_layout)
from de_stain.stain import augment_two_stain_views


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset(args, opt, phase, pairs=None, manifest=None):
    return PairedTargetDataset(
        args.target_dataroot,
        args.target_train_split if phase != "test" else args.target_test_split,
        opt, max_samples=args.target_train_samples if phase == "train" else None,
        seed=args.seed if phase != "test" else args.eval_seed, phase=phase,
        pairs=pairs, manifest=manifest, brightness_norm=args.brightness_norm,
        brightness_percentile=args.brightness_percentile,
    )


def train_step(model, batch, reference, bank, mpi_layers, key_layers, args, epoch, iteration):
    batch = dict(batch, current_epoch=epoch, current_iter=iteration)
    model.set_input(batch)
    model.forward()

    model.set_requires_grad(model.netD, True)
    model.optimizer_D.zero_grad(set_to_none=True)
    loss_d = model.compute_D_loss()
    loss_d.backward()
    model.optimizer_D.step()
    model.set_requires_grad(model.netD, False)

    model.optimizer_G.zero_grad(set_to_none=True)
    model.optimizer_F.zero_grad(set_to_none=True)
    base = model.compute_G_loss()
    zero = model.fake_B.new_zeros(())
    mpi = zero
    if args.lambda_mpi > 0:
        view_1, view_2 = augment_two_stain_views(model.real_A, args.stain_strength)
        mpi = compute_l_mpi(model.netG, view_1, view_2, mpi_layers)
    csp = zero
    if args.lambda_csp > 0:
        query = extract_morphology_descriptor(reference, model.real_A, key_layers)
        csp, _, _ = compute_l_csp(model.fake_B, query, bank, batch["index"])
    shared = base + args.lambda_mpi * mpi
    decoder_loss = args.lambda_csp * csp if args.lambda_csp > 0 else None
    backward_decoupled_losses(shared, decoder_loss, decoder_parameters(model.netG))
    model.optimizer_G.step()
    model.optimizer_F.step()
    return {
        "loss_d": loss_d.item(), "loss_asp_base": base.item(),
        "loss_mpi": mpi.item(), "loss_csp": csp.item(),
        "loss_total_g": (shared.detach() + args.lambda_csp * csp.detach()).item(),
    }


def main(argv=None):
    args = parse_args(argv)
    paths = pretrained_paths(args)  # fail before allocating a model if weights are absent
    exp_dir = Path(args.output_root) / args.exp_name
    if (exp_dir / "latest_net_G.pth").exists():
        raise FileExistsError(f"An experiment already exists at {exp_dir}; choose another exp_name.")
    seed_everything(args.seed)
    opt = build_backbone_options(args)
    manifest = args.split_manifest or exp_dir / "train_pairs.json"
    train_ds = dataset(args, opt, "train", manifest=manifest)
    train_ds.write_manifest(exp_dir / "train_pairs.json", args.target_train_split)
    bank_ds = dataset(args, opt, "bank", pairs=train_ds.pairs)
    test_ds = None if args.skip_evaluation else dataset(args, opt, "test")
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=loader_generator)
    bank_loader = DataLoader(bank_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = create_model(opt)
    first = dict(next(iter(bank_loader)), current_epoch=1, current_iter=0)
    model.data_dependent_initialize(first)
    sources = load_pretrained(model, paths)
    reference = make_reference_generator(model, opt) if args.lambda_csp > 0 else None
    if opt.gpu_ids:
        model.parallelize()
    model.optimizer_G = build_optimizer_G(model.netG, opt)
    model.optimizer_D = build_optimizer_D(model.netD, opt)
    model.optimizer_F = build_optimizer_F(model.netF, opt)
    model.optimizers = [model.optimizer_G, model.optimizer_D, model.optimizer_F]
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(0.0, 1.0 - epoch / args.epochs))
        for optimizer in model.optimizers
    ]
    mpi_layers, shallow_key, _ = generator_layout(model.netG)
    key_layers = [shallow_key, mpi_layers[-1]]
    bank = None
    if args.lambda_csp > 0:
        bank = TargetMemoryBank.build(
            reference, bank_loader, key_layers, model.device,
            hist_bins=args.csp_hist_bins, hist_sigma=args.csp_hist_sigma,
            top_k=args.csp_top_k, dab_threshold=args.dab_threshold,
            dab_dilation_radius=args.dab_dilation_radius,
        )
        torch.save(bank.state_dict(), exp_dir / "memory_bank.pth")
    snapshot = {
        "method": "DE_STAIN",
        "settings": {key: value for key, value in vars(args).items()
                     if key not in {"seed", "eval_seed"}},
        "mpi_layers": mpi_layers, "key_layers": key_layers,
        "source_checkpoints": sources,
        "evaluation_crop": "fixed per-image crop",
        "brightness_rule": "LAB L-channel percentile standardization on HE and IHC",
    }
    (exp_dir / "config.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DE_STAIN] Training on {len(train_ds)} paired images.")
    metrics_file = exp_dir / "epoch_losses.csv"
    fields = ["epoch", "lr", "loss_d", "loss_asp_base", "loss_mpi", "loss_csp", "loss_total_g"]
    iteration = 0
    with metrics_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_ds.set_epoch(epoch)
            totals = {key: 0.0 for key in fields[2:]}
            bar = tqdm(train_loader, desc=f"{args.exp_name} epoch {epoch}/{args.epochs}")
            for batch in bar:
                losses = train_step(model, batch, reference, bank, mpi_layers, key_layers,
                                    args, epoch, iteration)
                iteration += batch["A"].shape[0]
                for key, value in losses.items():
                    totals[key] += value
                bar.set_postfix(mpi=f"{losses['loss_mpi']:.4f}", csp=f"{losses['loss_csp']:.4f}")
            writer.writerow(dict(epoch=epoch, lr=model.optimizer_G.param_groups[0]["lr"],
                                 **{key: value / len(train_loader) for key, value in totals.items()}))
            stream.flush()
            if epoch % args.save_epoch_freq == 0 or epoch == args.epochs:
                save_networks(model, exp_dir, "latest")
                save_networks(model, exp_dir, epoch)
            if test_ds is not None and (epoch == args.epochs or
                                       (args.eval_every > 0 and epoch % args.eval_every == 0)):
                evaluate(model, test_ds, exp_dir / "eval" / f"epoch_{epoch:03d}", args)
            for scheduler in schedulers:
                scheduler.step()
    print(f"[DONE] DE_STAIN checkpoints and records: {exp_dir.resolve()}")


if __name__ == "__main__":
    main()
