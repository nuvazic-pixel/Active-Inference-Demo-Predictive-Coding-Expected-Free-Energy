#!/usr/bin/env python3
"""
Mini Predictive Coding + Active Inference (PyTorch)
---------------------------------------------------
A compact demo with:
  - nonlinear generative hierarchy (tanh)
  - precision-weighted predictive-coding inference
  - simple EFE-style policy scoring
  - ablations: baseline / random_policy / risk_only / no_precision
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# NOTE: In this toy, policy ablations mainly affect policy diagnostics.
# A stronger closed-loop effect appears in the foveated-observation version,
# where actions actually change what is observed.


@dataclass
class Config:
    # Data / truth
    n_samples: int = 1024
    dim_x: int = 8
    dim_z1: int = 4
    dim_z2: int = 2
    noise_x: float = 0.10
    noise_z1: float = 0.10

    # Training
    epochs: int = 30
    batch_size: int = 32
    seed: int = 42
    log_every: int = 5

    # Inference (inner loop)
    infer_steps: int = 8
    lr_mu: float = 0.10

    # Learning (outer loop)
    lr_w: float = 0.010
    lr_b: float = 0.010

    # Precisions
    precision_x: float = 10.0
    precision_z1: float = 5.0
    precision_z2: float = 1.0

    # Ablations: baseline | random_policy | risk_only | no_precision
    ablation: str = 'baseline'

    # Policy proposals in z2
    n_policies: int = 8
    saccade_std: float = 0.30

    # Misc
    plot: bool = False
    device: str = 'cpu'


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def whiten(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    return (x - mean) / std


def make_synthetic_data(cfg: Config, device: torch.device) -> Tuple[torch.utils.data.DataLoader, Dict[str, torch.Tensor]]:
    set_seed(cfg.seed)
    n = cfg.n_samples

    W2_true = torch.randn(cfg.dim_z2, cfg.dim_z1, device=device) / math.sqrt(cfg.dim_z2)
    b2_true = torch.randn(cfg.dim_z1, device=device) * 0.1
    W1_true = torch.randn(cfg.dim_z1, cfg.dim_x, device=device) / math.sqrt(cfg.dim_z1)
    b1_true = torch.randn(cfg.dim_x, device=device) * 0.1

    z2 = torch.randn(n, cfg.dim_z2, device=device)
    pre_z1 = z2 @ W2_true + b2_true
    z1 = torch.tanh(pre_z1) + cfg.noise_z1 * torch.randn(n, cfg.dim_z1, device=device)
    pre_x = z1 @ W1_true + b1_true
    x = torch.tanh(pre_x) + cfg.noise_x * torch.randn(n, cfg.dim_x, device=device)

    x = whiten(x)
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    truth = {'W1': W1_true, 'b1': b1_true, 'W2': W2_true, 'b2': b2_true, 'z2': z2, 'z1': z1, 'x': x}
    return loader, truth


class PCNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d_x, d1, d2 = cfg.dim_x, cfg.dim_z1, cfg.dim_z2
        self.W2 = nn.Parameter(torch.randn(d2, d1) / math.sqrt(d2))
        self.b2 = nn.Parameter(torch.zeros(d1))
        self.W1 = nn.Parameter(torch.randn(d1, d_x) / math.sqrt(d1))
        self.b1 = nn.Parameter(torch.zeros(d_x))
        self.log_prec_x = nn.Parameter(torch.log(torch.tensor(cfg.precision_x)))
        self.log_prec_z1 = nn.Parameter(torch.log(torch.tensor(cfg.precision_z1)))
        self.log_prec_z2 = nn.Parameter(torch.log(torch.tensor(cfg.precision_z2)))

    def predict_z1(self, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = z2 @ self.W2 + self.b2
        z1_hat = torch.tanh(pre)
        return z1_hat, pre

    def predict_x(self, z1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = z1 @ self.W1 + self.b1
        x_hat = torch.tanh(pre)
        return x_hat, pre


def get_effective_precisions(net: PCNet):
    cfg = net.cfg
    if cfg.ablation == 'no_precision':
        one = torch.tensor(1.0, device=net.W1.device)
        return one, one, one
    return torch.exp(net.log_prec_x), torch.exp(net.log_prec_z1), torch.exp(net.log_prec_z2)


def inference_step(net: PCNet, x: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor):
    cfg = net.cfg
    lx, lz1, lz2 = get_effective_precisions(net)

    x_hat, pre_x = net.predict_x(z1)
    z1_hat, pre_z1 = net.predict_z1(z2)

    fxp = 1.0 - x_hat ** 2
    fzp = 1.0 - z1_hat ** 2

    eps_x = lx * (x - x_hat)
    eps_z1 = lz1 * (z1 - z1_hat)
    eps_z2 = lz2 * z2

    dz1 = (eps_x * fxp) @ net.W1.T - eps_z1
    dz2 = (eps_z1 * fzp) @ net.W2.T - eps_z2

    z1 = z1 + cfg.lr_mu * dz1
    z2 = z2 + cfg.lr_mu * dz2
    return z1, z2, eps_x, eps_z1, eps_z2, pre_x, pre_z1


def learning_step(net: PCNet, x: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor,
                  eps_x: torch.Tensor, eps_z1: torch.Tensor,
                  pre_x: torch.Tensor, pre_z1: torch.Tensor) -> None:
    cfg = net.cfg
    fxp = 1.0 - torch.tanh(pre_x) ** 2
    fzp = 1.0 - torch.tanh(pre_z1) ** 2

    net.W1.data += cfg.lr_w * (z1.T @ (eps_x * fxp)) / x.size(0)
    net.b1.data += cfg.lr_b * (eps_x * fxp).mean(dim=0)
    net.W2.data += cfg.lr_w * (z2.T @ (eps_z1 * fzp)) / x.size(0)
    net.b2.data += cfg.lr_b * (eps_z1 * fzp).mean(dim=0)


def compute_policy_EFE(net: PCNet, x: torch.Tensor, z2: torch.Tensor, policies: torch.Tensor) -> torch.Tensor:
    """
    Toy EFE surrogate:
      baseline     -> risk - info_gain
      risk_only    -> risk only
      no_precision -> same objective but unit precisions
    """
    device = x.device
    lx, _lz1, lz2 = get_effective_precisions(net)
    G = torch.zeros(policies.size(0), device=device)

    with torch.no_grad():
        for i, d in enumerate(policies):
            z2_pi = z2 + d.unsqueeze(0)
            z1_hat, pre_z1 = net.predict_z1(z2_pi)
            x_hat, pre_x = net.predict_x(z1_hat)

            risk = 0.5 * lx * ((x - x_hat) ** 2).sum()
            fxp = (1.0 - torch.tanh(pre_x) ** 2).squeeze(0)
            fzp = (1.0 - torch.tanh(pre_z1) ** 2).squeeze(0)
            J_norm2 = torch.sum((net.W2 * fzp.unsqueeze(0)).pow(2)) * torch.sum((net.W1 * fxp.unsqueeze(0)).pow(2))
            post_prec = lz2 + lx * J_norm2.clamp_min(1e-6)
            prior_var = 1.0 / (lz2 + 1e-6)
            post_var = 1.0 / (post_prec + 1e-6)
            info_gain = 0.5 * torch.log(prior_var / post_var)

            if net.cfg.ablation == 'risk_only':
                G[i] = risk
            else:
                G[i] = risk - info_gain
    return G


def train(cfg: Config):
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    loader, truth = make_synthetic_data(cfg, device)
    net = PCNet(cfg).to(device)

    losses, policies_log, metrics_history = [], [], []
    print(f"Training PC + Active Inference on {len(loader.dataset)} samples…")
    print(f"seed={cfg.seed} | ablation={cfg.ablation} | epochs={cfg.epochs} | infer_steps={cfg.infer_steps} | hidden={cfg.dim_z1} | top={cfg.dim_z2}")

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_z1_norms, epoch_z2_norms, epoch_best_efes = [], [], []

        for (x_batch,) in loader:
            x = x_batch.to(device)
            B = x.size(0)
            z2 = torch.zeros(B, cfg.dim_z2, device=device)
            z1 = torch.zeros(B, cfg.dim_z1, device=device)

            for _ in range(cfg.infer_steps):
                z1, z2, eps_x, eps_z1, eps_z2, pre_x, pre_z1 = inference_step(net, x, z1, z2)

            learning_step(net, x, z1, z2, eps_x, eps_z1, pre_x, pre_z1)

            deltas = cfg.saccade_std * torch.randn(cfg.n_policies, cfg.dim_z2, device=device)
            G = compute_policy_EFE(net, x[:1], z2[:1], deltas)
            if cfg.ablation == 'random_policy':
                best_idx = torch.randint(0, deltas.size(0), (1,), device=device).item()
            else:
                best_idx = torch.argmin(G).item()
            best_pi = deltas[best_idx]
            best_efe = G[best_idx].item()
            policies_log.append(best_pi.detach().cpu())
            epoch_best_efes.append(best_efe)

            epoch_z1_norms.append(z1.norm(dim=1).mean().item())
            epoch_z2_norms.append(z2.norm(dim=1).mean().item())

            x_hat, _ = net.predict_x(z1)
            loss = F.mse_loss(x_hat, x)
            epoch_loss += loss.item()

        recon_mse = epoch_loss / len(loader)
        z1_norm_mean = sum(epoch_z1_norms) / max(len(epoch_z1_norms), 1)
        z2_norm_mean = sum(epoch_z2_norms) / max(len(epoch_z2_norms), 1)
        best_efe_mean = sum(epoch_best_efes) / max(len(epoch_best_efes), 1)
        best_efe_std = float(torch.tensor(epoch_best_efes).std().item()) if len(epoch_best_efes) > 1 else 0.0

        losses.append(recon_mse)
        metrics_history.append({
            'epoch': epoch,
            'recon_mse': recon_mse,
            'z1_norm_mean': z1_norm_mean,
            'z2_norm_mean': z2_norm_mean,
            'best_efe_mean': best_efe_mean,
            'best_efe_std': best_efe_std,
        })

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            print(
                f"Epoch {epoch:3d} | recon_mse={recon_mse:.6f} | "
                f"z1_norm={z1_norm_mean:.4f} | z2_norm={z2_norm_mean:.4f} | "
                f"best_efe_mean={best_efe_mean:.6f} | best_efe_std={best_efe_std:.6f}"
            )

    if cfg.plot and HAS_MPL:
        plot_results(net, truth, losses, policies_log, cfg)

    return net, losses, metrics_history


def plot_results(net: PCNet, truth: Dict[str, torch.Tensor], losses: List[float], policies_log: List[torch.Tensor], cfg: Config) -> None:
    device = next(net.parameters()).device
    x_true = truth['x'][:128].to(device)
    z2 = torch.zeros(x_true.size(0), cfg.dim_z2, device=device)
    z1 = torch.zeros(x_true.size(0), cfg.dim_z1, device=device)
    for _ in range(cfg.infer_steps):
        z1, z2, *_ = inference_step(net, x_true, z1, z2)
    x_rec, _ = net.predict_x(z1)

    fig = plt.figure(figsize=(14, 9))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(losses)
    ax1.set_title('Reconstruction MSE')
    ax1.set_xlabel('Epoch')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    if cfg.dim_z2 == 2:
        z2_true = truth['z2'][:128].cpu()
        ax2.scatter(z2_true[:, 0], z2_true[:, 1], s=12, alpha=0.5, label='True z2')
        ax2.scatter(z2[:, 0].cpu(), z2[:, 1].cpu(), s=12, alpha=0.5, label='Inferred z2')
        ax2.set_title('Latent Space (z2)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.1, 0.5, 'z2 dim != 2')
        ax2.axis('off')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title('EFE-Minimizing Policies (Δz2)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    for p in policies_log[-50:]:
        p = p.numpy()
        ax3.arrow(0, 0, p[0], p[1] if p.shape[0] > 1 else 0.0, head_width=0.05, alpha=0.5)
    ax3.scatter([0], [0], c='k', s=25)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(x_true[0].cpu(), label='True')
    ax4.plot(x_rec[0].cpu(), '--', label='Recon')
    ax4.legend()
    ax4.set_title('Input vs Reconstruction')

    ax5 = fig.add_subplot(2, 3, 5)
    im1 = ax5.imshow(net.W1.data.cpu(), aspect='auto', cmap='coolwarm')
    fig.colorbar(im1, ax=ax5)
    ax5.set_title('W1 (z1→x)')

    ax6 = fig.add_subplot(2, 3, 6)
    im2 = ax6.imshow(net.W2.data.cpu(), aspect='auto', cmap='coolwarm')
    fig.colorbar(im2, ax=ax6)
    ax6.set_title('W2 (z2→z1)')

    plt.tight_layout()
    plt.show()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Mini predictive-coding + active inference (PyTorch)')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--infer-steps', type=int, default=8)
    p.add_argument('--hidden', type=int, default=4, help='dim of z1')
    p.add_argument('--top', type=int, default=2, help='dim of z2')
    p.add_argument('--lr-mu', type=float, default=0.10)
    p.add_argument('--lr-w', type=float, default=0.010)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--ablation', type=str, default='baseline', choices=['baseline', 'random_policy', 'risk_only', 'no_precision'])
    p.add_argument('--plot', action='store_true')
    p.add_argument('--device', type=str, default='cpu')
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        epochs=args.epochs,
        infer_steps=args.infer_steps,
        dim_z1=args.hidden,
        dim_z2=args.top,
        lr_mu=args.lr_mu,
        lr_w=args.lr_w,
        lr_b=args.lr_w,
        seed=args.seed,
        log_every=args.log_every,
        ablation=args.ablation,
        plot=args.plot,
        device=args.device,
    )
    train(cfg)


if __name__ == '__main__':
    main()
