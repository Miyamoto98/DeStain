"""Uniform full-model Adam and architecture-derived module boundaries."""
from __future__ import annotations

import torch


def unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def generator_layout(netG):
    """Return ResNet block indices, shallow retrieval layer and decoder start."""
    model = unwrap_module(netG).model
    blocks = [i for i, layer in enumerate(model) if layer.__class__.__name__ == "ResnetBlock"]
    if not blocks or blocks != list(range(blocks[0], blocks[-1] + 1)):
        raise ValueError("DE_STAIN requires a sequential ASP ResNet generator.")
    convolutions = [i for i, layer in enumerate(model)
                    if isinstance(layer, torch.nn.Conv2d) and i < blocks[0]]
    return blocks, convolutions[-1], blocks[-1] + 1


def decoder_parameters(netG):
    _, _, start = generator_layout(netG)
    return tuple(unwrap_module(netG).model[start:].parameters())


def build_optimizer_G(netG, opt):
    for parameter in netG.parameters():
        parameter.requires_grad_(True)
    return torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


def build_optimizer_D(netD, opt):
    return torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


def build_optimizer_F(netF, opt):
    return torch.optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
