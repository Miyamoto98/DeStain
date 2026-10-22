"""ASP integration, explicit pretrained loading and checkpoint utilities."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from .config import ROOT
from .optim import unwrap_module

ASP_ROOT = ROOT / "asp"
if str(ASP_ROOT) not in sys.path:
    sys.path.insert(0, str(ASP_ROOT))

from models import create_model, networks
from options.train_options import TrainOptions


def build_backbone_options(args, is_train=True):
    fields = {
        "dataroot": args.target_dataroot, "name": args.exp_name,
        "checkpoints_dir": args.output_root, "model": "cpt", "dataset_mode": "aligned",
        "direction": "AtoB", "gpu_ids": args.gpu_ids, "batch_size": args.batch_size,
        "num_threads": args.num_workers, "preprocess": "crop",
        "load_size": args.load_size, "crop_size": args.crop_size,
        "n_epochs": 0, "n_epochs_decay": args.epochs, "lr": args.lr,
        "netG": args.netG, "netD": args.netD, "ngf": args.ngf, "ndf": args.ndf,
        "n_layers_D": args.n_layers_D, "weight_norm": args.weight_norm,
        "netF": "mlp_sample", "nce_layers": args.nce_layers,
        "num_patches": args.num_patches, "nce_T": args.nce_T,
        "lambda_NCE": args.lambda_NCE, "lambda_GAN": args.lambda_GAN,
        "lambda_gp": args.lambda_gp, "lambda_asp": args.lambda_asp,
        "asp_loss_mode": args.asp_loss_mode, "nce_idt": "False",
        "display_id": -1, "phase": "train",
        "save_epoch_freq": args.save_epoch_freq,
    }
    command = [item for key, value in fields.items() for item in ("--" + key, str(value))]
    command.append("--no_html")
    if args.brightness_norm:
        command.append("--brightness_norm")
    # Do not let evaluation overwrite the vendor's training option file.
    opt = TrainOptions(cmd_line=command).gather_options()
    opt.isTrain = is_train
    opt.gpu_ids = [int(item) for item in args.gpu_ids.split(",") if int(item) >= 0]
    if opt.gpu_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --gpu_ids -1 for CPU execution.")
        torch.cuda.set_device(opt.gpu_ids[0])
    return opt


def load_tensor_state(path, device="cpu"):
    path = Path(path)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def pretrained_paths(args):
    directory = Path(args.pretrained_checkpoints_dir) / args.pretrained_name
    paths = {name: directory / f"{args.pretrained_epoch}_net_{name}.pth" for name in ("G", "F", "D")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Supply the MIST ASP Lambda-Linear source checkpoints:\n" + "\n".join(missing))
    return paths


def load_pretrained(model, paths):
    result = {}
    for name, path in paths.items():
        unwrap_module(getattr(model, "net" + name)).load_state_dict(load_tensor_state(path, model.device), strict=True)
        result[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def make_reference_generator(model, opt):
    reference = networks.define_G(
        opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
        opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, [], opt
    ).to(model.device)
    reference.load_state_dict(unwrap_module(model.netG).state_dict(), strict=True)
    reference.eval()
    reference.requires_grad_(False)
    return reference


def save_networks(model, directory, epoch):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in model.model_names:
        network = unwrap_module(getattr(model, "net" + name))
        state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        path = directory / f"{epoch}_net_{name}.pth"
        temporary = path.with_suffix(".pth.tmp")
        torch.save(state, temporary)
        temporary.replace(path)
