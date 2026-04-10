#!/usr/bin/env python3
"""
Mini Predictive-Coding Demo (PyTorch)
-------------------------------------
A tiny, self-contained predictive-coding network that:
  - learns a simple generative hierarchy x <- W1*z1 + b1, z1 <- W2*z2 + b2
  - performs latent inference by minimizing prediction errors
  - updates parameters with local Hebbian-like rules
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def whiten(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    return (x - mean) / std


@dataclass
class SynthConfig:
    n_samples: int = 4096
    dim_x: int = 20
    dim_z1: int = 12
    dim_z2: int = 4
    noise_x: float = 0.05
    noise_z1: float = 0.05


def make_synth(cfg: SynthConfig, device: torch.device) -> Tuple[torch.Tensor, dict]:
    W2_true = torch.randn(cfg.dim_z1, cfg.dim_z2, device=device) / math.sqrt(cfg.dim_z2)
    b2_true = torch.randn(cfg.dim_z1, device=device) * 0.1
    W1_true = torch.randn(cfg.dim_x, cfg.dim_z1, device=device) / math.sqrt(cfg.dim_z1)
    b1_true = torch.randn(cfg.dim_x, device=device) * 0.1

    z2 = torch.randn(cfg.n_samples, cfg.dim_z2, device=device)
    z1 = (z2 @ W2_true.T) + b2_true + cfg.noise_z1 * torch.randn(cfg.n_samples, cfg.dim_z1, device=device)
    x = (z1 @ W1_true.T) + b1_true + cfg.noise_x * torch.randn(cfg.n_samples, cfg.dim_x, device=device)
    x = whiten(x)

    data = {
        'W1_true': W1_true,
        'b1_true': b1_true,
        'W2_true': W2_true,
        'b2_true': b2_true,
        'z2_true': z2,
        'z1_true': z1,
    }
    return x, data


class PCLinearLayer(nn.Module):
    def __init__(self, dim_low: int, dim_high: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim_low, dim_high) / math.sqrt(dim_high))
        self.b = nn.Parameter(torch.zeros(dim_low))

    def predict(self, mu_high: torch.Tensor) -> torch.Tensor:
        return mu_high @ self.W.T + self.b


class PCNetwork(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        assert len(dims) >= 2
        self.dims = dims
        self.layers = nn.ModuleList([PCLinearLayer(dims[l - 1], dims[l]) for l in range(1, len(dims))])

    @torch.no_grad()
    def infer(self, x: torch.Tensor, steps: int = 20, lr_mu: float = 0.2) -> List[torch.Tensor]:
        B = x.size(0)
        mus = [x.clone()]
        for l in range(1, len(self.dims)):
            mus.append(torch.zeros(B, self.dims[l], device=x.device))

        for _ in range(steps):
            errors = []
            for l, layer in enumerate(self.layers, start=1):
                pred = layer.predict(mus[l])
                e_low = mus[l - 1] - pred
                errors.append(e_low)
            errors.append(mus[-1])

            for l in range(1, len(mus)):
                e_l = errors[l]
                e_low = errors[l - 1]
                W_l = self.layers[l - 1].W
                delta = (e_low @ W_l) - e_l
                mus[l] = mus[l] + lr_mu * delta
        return mus

    @torch.no_grad()
    def learn(self, x: torch.Tensor, steps: int = 20, lr_mu: float = 0.2, lr_w: float = 0.02, lr_b: float = 0.02) -> dict:
        mus = self.infer(x, steps=steps, lr_mu=lr_mu)
        errors = []
        recon = None
        for l, layer in enumerate(self.layers, start=1):
            pred = layer.predict(mus[l])
            e_low = mus[l - 1] - pred
            errors.append(e_low)
            if l == 1:
                recon = pred
        errors.append(mus[-1])

        for l, layer in enumerate(self.layers, start=1):
            e_low = errors[l - 1]
            mu_high = mus[l]
            dW = (e_low.T @ mu_high) / x.size(0)
            db = e_low.mean(dim=0)
            layer.W.data += lr_w * dW
            layer.b.data += lr_b * db

        mse_recon = (errors[0] ** 2).mean().item()
        mse_all = torch.mean(torch.cat([e.view(x.size(0), -1) ** 2 for e in errors], dim=1)).mean().item()
        return {'mse_recon': mse_recon, 'mse_all': mse_all, 'recon': recon, 'mus': mus}


def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_seed(args.seed)

    synth = SynthConfig(
        n_samples=args.n_samples,
        dim_x=args.input,
        dim_z1=args.hidden,
        dim_z2=args.top,
        noise_x=args.noise_x,
        noise_z1=args.noise_z1,
    )
    X, _truth = make_synth(synth, device)

    net = PCNetwork([args.input, args.hidden, args.top]).to(device)
    N = X.size(0)
    idx = torch.randperm(N, device=device)
    X = X[idx]

    losses_recon = []
    for epoch in range(1, args.epochs + 1):
        start = ((epoch - 1) * args.batch) % N
        end = start + args.batch
        xb = X[start:end]
        out = net.learn(xb, steps=args.infer_steps, lr_mu=args.lr_mu, lr_w=args.lr_w, lr_b=args.lr_b)
        losses_recon.append(out['mse_recon'])
        if epoch % args.log_every == 0:
            print(f"epoch {epoch:04d}  recon_mse={out['mse_recon']:.6f}  all_mse={out['mse_all']:.6f}")

    with torch.no_grad():
        xb = X[:args.batch]
        mus = net.infer(xb, steps=args.infer_steps, lr_mu=args.lr_mu)
        recon = net.layers[0].predict(mus[1])
        final_mse = torch.mean((xb - recon) ** 2).item()

    print('\nFinal reconstruction MSE (mini-batch):', final_mse)

    if HAS_MPL and args.plot:
        plt.figure()
        plt.plot(losses_recon)
        plt.xlabel('Training step')
        plt.ylabel('Recon MSE')
        plt.title('Predictive-Coding Training (Recon MSE)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Mini predictive-coding (linear) in PyTorch')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true', help='force CPU')
    p.add_argument('--n-samples', type=int, default=4096)
    p.add_argument('--input', type=int, default=20, help='dim of observed x')
    p.add_argument('--hidden', type=int, default=16, help='dim of z1')
    p.add_argument('--top', type=int, default=4, help='dim of z2')
    p.add_argument('--noise-x', type=float, default=0.05)
    p.add_argument('--noise-z1', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--infer-steps', type=int, default=30)
    p.add_argument('--lr-mu', type=float, default=0.2)
    p.add_argument('--lr-w', type=float, default=0.02)
    p.add_argument('--lr-b', type=float, default=0.02)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--plot', action='store_true')
    return p


if __name__ == '__main__':
    train(build_argparser().parse_args())
