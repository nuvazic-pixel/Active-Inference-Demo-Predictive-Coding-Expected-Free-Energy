#!/usr/bin/env python3
"""
Foveated MNIST — Active Inference Lite (PyTorch)
================================================
A small active-sensing benchmark:
  - MNIST 28x28
  - fixed 8x8 glimpse patches
  - 3x3 discrete glimpse grid
  - recurrent hidden state
  - outputs class logits + full-image reconstruction
  - compares random glimpses vs EFE-lite uncertainty-guided glimpses
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MPL = True
except Exception:
    HAS_MPL = False


@dataclass
class Config:
    data_dir: str = './data'
    train_size: int = 5000
    test_size: int = 1000
    batch_size: int = 128
    epochs: int = 5
    seed: int = 42
    device: str = 'cpu'

    patch_size: int = 8
    n_glimpses: int = 5
    hidden_dim: int = 128
    glimpse_dim: int = 128
    dropout_p: float = 0.15
    recon_weight: float = 0.25

    lr: float = 1e-3
    mc_samples: int = 8
    log_every: int = 100
    plot: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GRID_COORDS: List[Tuple[int, int]] = [
    (0, 0), (0, 10), (0, 20),
    (10, 0), (10, 10), (10, 20),
    (20, 0), (20, 10), (20, 20),
]
CENTER_INDEX = 4
N_LOCS = len(GRID_COORDS)


def extract_patches(images: torch.Tensor, loc_idx: torch.Tensor, patch_size: int) -> torch.Tensor:
    out = []
    for i in range(images.size(0)):
        y, x = GRID_COORDS[int(loc_idx[i].item())]
        out.append(images[i:i+1, :, y:y+patch_size, x:x+patch_size])
    return torch.cat(out, dim=0)


def patch_scores_from_uncertainty(var_map: torch.Tensor, patch_size: int) -> torch.Tensor:
    scores = []
    for y, x in GRID_COORDS:
        patch_var = var_map[:, :, y:y+patch_size, x:x+patch_size].mean(dim=(1, 2, 3))
        scores.append(patch_var)
    return torch.stack(scores, dim=1)


def make_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=tfm)
    train_sub = Subset(train_ds, list(range(min(cfg.train_size, len(train_ds)))))
    test_sub = Subset(test_ds, list(range(min(cfg.test_size, len(test_ds)))))
    train_loader = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_sub, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


class FoveatedGlimpseModel(nn.Module):
    def __init__(self, patch_size: int, hidden_dim: int, glimpse_dim: int, dropout_p: float):
        super().__init__()
        patch_dim = patch_size * patch_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        self.glimpse_encoder = nn.Sequential(
            nn.Linear(patch_dim + N_LOCS, glimpse_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(glimpse_dim, glimpse_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(glimpse_dim, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 10)
        self.recon_head = nn.Linear(hidden_dim, 28 * 28)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def encode_glimpse(self, patch: torch.Tensor, loc_idx: torch.Tensor) -> torch.Tensor:
        B = patch.size(0)
        patch_flat = patch.view(B, -1)
        loc_onehot = F.one_hot(loc_idx, num_classes=N_LOCS).float()
        return self.glimpse_encoder(torch.cat([patch_flat, loc_onehot], dim=1))

    def step(self, patch: torch.Tensor, loc_idx: torch.Tensor, h: torch.Tensor):
        g = self.encode_glimpse(patch, loc_idx)
        h = self.rnn(g, h)
        logits = self.class_head(h)
        recon = torch.sigmoid(self.recon_head(h)).view(-1, 1, 28, 28)
        return h, logits, recon

    @torch.no_grad()
    def predictive_stats(self, h: torch.Tensor, mc_samples: int = 8):
        recons = []
        probs = []
        for _ in range(mc_samples):
            h_drop = F.dropout(h, p=self.dropout_p, training=True)
            logits = self.class_head(h_drop)
            recon = torch.sigmoid(self.recon_head(h_drop)).view(-1, 1, 28, 28)
            recons.append(recon)
            probs.append(F.softmax(logits, dim=1))
        recon_stack = torch.stack(recons, dim=0)
        prob_stack = torch.stack(probs, dim=0)
        mean_recon = recon_stack.mean(dim=0)
        var_recon = recon_stack.var(dim=0, unbiased=False)
        mean_prob = prob_stack.mean(dim=0)
        class_entropy = -(mean_prob * torch.log(mean_prob.clamp_min(1e-8))).sum(dim=1)
        return mean_recon, var_recon, class_entropy


def sample_random_glimpse_orders(batch_size: int, n_glimpses: int, device: torch.device) -> torch.Tensor:
    orders = []
    for _ in range(batch_size):
        candidates = list(range(N_LOCS))
        candidates.remove(CENTER_INDEX)
        random.shuffle(candidates)
        seq = [CENTER_INDEX] + candidates[:max(0, n_glimpses - 1)]
        orders.append(seq)
    return torch.tensor(orders, device=device, dtype=torch.long)


def train_one_epoch(model: FoveatedGlimpseModel, loader: DataLoader, opt: torch.optim.Optimizer,
                    cfg: Config, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_recon = 0.0
    total_correct = 0
    total_seen = 0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        B = images.size(0)
        opt.zero_grad()
        h = model.init_hidden(B, device)
        orders = sample_random_glimpse_orders(B, cfg.n_glimpses, device)

        loss = 0.0
        ce_acc = 0.0
        rec_acc = 0.0
        for t in range(cfg.n_glimpses):
            loc_idx = orders[:, t]
            patch = extract_patches(images, loc_idx, cfg.patch_size)
            h, logits, recon = model.step(patch, loc_idx, h)
            ce = F.cross_entropy(logits, labels)
            rec = F.mse_loss(recon, images)
            loss = loss + ce + cfg.recon_weight * rec
            ce_acc += ce.item()
            rec_acc += rec.item()

        loss = loss / cfg.n_glimpses
        loss.backward()
        opt.step()

        total_loss += float(loss.item())
        total_ce += ce_acc / cfg.n_glimpses
        total_recon += rec_acc / cfg.n_glimpses
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_seen += B

        if (step + 1) % cfg.log_every == 0:
            print(f'train step {step+1:04d} | loss={loss.item():.4f}')

    return {
        'loss': total_loss / max(len(loader), 1),
        'ce': total_ce / max(len(loader), 1),
        'recon_mse': total_recon / max(len(loader), 1),
        'acc': total_correct / max(total_seen, 1),
    }


@torch.no_grad()
def choose_next_random(unvisited: torch.Tensor) -> torch.Tensor:
    next_idx = []
    for i in range(unvisited.size(0)):
        choices = torch.where(unvisited[i])[0].tolist()
        next_idx.append(random.choice(choices))
    return torch.tensor(next_idx, device=unvisited.device, dtype=torch.long)


@torch.no_grad()
def choose_next_efe_lite(model: FoveatedGlimpseModel, h: torch.Tensor, unvisited: torch.Tensor, cfg: Config) -> torch.Tensor:
    _, var_map, _ = model.predictive_stats(h, mc_samples=cfg.mc_samples)
    scores = patch_scores_from_uncertainty(var_map, cfg.patch_size)
    scores = scores.masked_fill(~unvisited, float('-inf'))
    return scores.argmax(dim=1)


@torch.no_grad()
def evaluate_policy(model: FoveatedGlimpseModel, loader: DataLoader, cfg: Config,
                    device: torch.device, policy: str = 'random') -> Dict[str, object]:
    assert policy in {'random', 'efe_lite'}
    model.eval()

    total_correct = 0
    total_seen = 0
    total_recon = 0.0

    correct_by_glimpse = [0 for _ in range(cfg.n_glimpses)]
    recon_sum_by_glimpse = [0.0 for _ in range(cfg.n_glimpses)]
    seen_by_glimpse = [0 for _ in range(cfg.n_glimpses)]

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        B = images.size(0)
        h = model.init_hidden(B, device)
        unvisited = torch.ones(B, N_LOCS, dtype=torch.bool, device=device)

        loc_idx = torch.full((B,), CENTER_INDEX, device=device, dtype=torch.long)
        unvisited[torch.arange(B), loc_idx] = False

        last_logits = None
        last_recon = None
        for t in range(cfg.n_glimpses):
            patch = extract_patches(images, loc_idx, cfg.patch_size)
            h, logits, recon = model.step(patch, loc_idx, h)
            preds = logits.argmax(dim=1)

            correct_by_glimpse[t] += int((preds == labels).sum().item())
            recon_sum_by_glimpse[t] += float(F.mse_loss(recon, images, reduction='mean').item())
            seen_by_glimpse[t] += B

            last_logits = logits
            last_recon = recon

            if t < cfg.n_glimpses - 1:
                if policy == 'random':
                    loc_idx = choose_next_random(unvisited)
                else:
                    loc_idx = choose_next_efe_lite(model, h, unvisited, cfg)
                unvisited[torch.arange(B), loc_idx] = False

        total_correct += int((last_logits.argmax(dim=1) == labels).sum().item())
        total_seen += B
        total_recon += float(F.mse_loss(last_recon, images, reduction='mean').item())

    acc_by_glimpse = [correct_by_glimpse[t] / max(seen_by_glimpse[t], 1) for t in range(cfg.n_glimpses)]
    recon_by_glimpse = [recon_sum_by_glimpse[t] / max(len(loader), 1) for t in range(cfg.n_glimpses)]

    final_acc = total_correct / max(total_seen, 1)
    final_recon = total_recon / max(len(loader), 1)

    tol = 1e-9
    assert abs(final_acc - acc_by_glimpse[-1]) < tol, (final_acc, acc_by_glimpse[-1], policy)
    assert abs(final_recon - recon_by_glimpse[-1]) < tol, (final_recon, recon_by_glimpse[-1], policy)

    return {
        'acc': final_acc,
        'recon_mse': final_recon,
        'acc_by_glimpse': acc_by_glimpse,
        'recon_by_glimpse': recon_by_glimpse,
    }


def print_policy_report(name: str, stats: Dict[str, object]) -> None:
    print(f'{name:8s} | acc={stats["acc"]:.4f} | recon_mse={stats["recon_mse"]:.4f}')
    print(f'  acc_by_glimpse   = {[round(x, 4) for x in stats["acc_by_glimpse"]]}')
    print(f'  recon_by_glimpse = {[round(x, 4) for x in stats["recon_by_glimpse"]]}')


@torch.no_grad()
def rollout_one(model: FoveatedGlimpseModel, image: torch.Tensor, label: int,
                cfg: Config, device: torch.device, policy: str) -> Dict[str, object]:
    model.eval()
    image = image.unsqueeze(0).to(device)
    h = model.init_hidden(1, device)
    unvisited = torch.ones(1, N_LOCS, dtype=torch.bool, device=device)

    locs = [CENTER_INDEX]
    loc_idx = torch.tensor([CENTER_INDEX], device=device)
    unvisited[0, CENTER_INDEX] = False

    recons = []
    logits_hist = []
    for t in range(cfg.n_glimpses):
        patch = extract_patches(image, loc_idx, cfg.patch_size)
        h, logits, recon = model.step(patch, loc_idx, h)
        recons.append(recon.squeeze(0).cpu())
        logits_hist.append(logits.squeeze(0).cpu())

        if t < cfg.n_glimpses - 1:
            if policy == 'random':
                loc_idx = choose_next_random(unvisited)
            else:
                loc_idx = choose_next_efe_lite(model, h, unvisited, cfg)
            locs.append(int(loc_idx.item()))
            unvisited[0, loc_idx] = False

    pred = int(logits.argmax(dim=1).item())
    return {'label': label, 'pred': pred, 'locs': locs, 'recons': recons, 'final_logits': logits_hist[-1]}


@torch.no_grad()
def plot_demo(model: FoveatedGlimpseModel, test_loader: DataLoader, cfg: Config, device: torch.device) -> None:
    if not HAS_MPL:
        print('matplotlib not available; skipping plots')
        return

    images, labels = next(iter(test_loader))
    image = images[0]
    label = int(labels[0].item())

    out_rand = rollout_one(model, image, label, cfg, device, policy='random')
    out_efe = rollout_one(model, image, label, cfg, device, policy='efe_lite')

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    ax = axes[0, 0]
    ax.imshow(image.squeeze(0), cmap='gray')
    ax.set_title(f'Original | label={label}')
    for i, idx in enumerate(out_rand['locs']):
        y, x = GRID_COORDS[idx]
        rect = patches.Rectangle((x, y), cfg.patch_size, cfg.patch_size, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 0.5, str(i + 1), color='red', fontsize=8)
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(out_rand['recons'][-1].squeeze(0), cmap='gray')
    ax.set_title(f'Random recon | pred={out_rand["pred"]}')
    ax.axis('off')

    ax = axes[0, 2]
    probs = F.softmax(out_rand['final_logits'], dim=0).numpy()
    ax.bar(range(10), probs)
    ax.set_title('Random class probs')
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    ax.imshow(image.squeeze(0), cmap='gray')
    ax.set_title(f'Original | label={label}')
    for i, idx in enumerate(out_efe['locs']):
        y, x = GRID_COORDS[idx]
        rect = patches.Rectangle((x, y), cfg.patch_size, cfg.patch_size, linewidth=1.5, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 0.5, str(i + 1), color='lime', fontsize=8)
    ax.axis('off')

    ax = axes[1, 1]
    ax.imshow(out_efe['recons'][-1].squeeze(0), cmap='gray')
    ax.set_title(f'EFE-lite recon | pred={out_efe["pred"]}')
    ax.axis('off')

    ax = axes[1, 2]
    probs = F.softmax(out_efe['final_logits'], dim=0).numpy()
    ax.bar(range(10), probs)
    ax.set_title('EFE-lite class probs')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Foveated MNIST active inference lite')
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--train-size', type=int, default=5000)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--patch-size', type=int, default=8)
    p.add_argument('--glimpses', type=int, default=5)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--glimpse-dim', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--recon-weight', type=float, default=0.25)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--mc-samples', type=int, default=8)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--plot', action='store_true')
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        patch_size=args.patch_size,
        n_glimpses=args.glimpses,
        hidden_dim=args.hidden,
        glimpse_dim=args.glimpse_dim,
        dropout_p=args.dropout,
        recon_weight=args.recon_weight,
        lr=args.lr,
        mc_samples=args.mc_samples,
        log_every=args.log_every,
        plot=args.plot,
    )

    if cfg.patch_size != 8:
        print('Note: the discrete 3x3 grid is hard-coded for 8x8 patches on 28x28 MNIST. Use patch_size=8 for now.')

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_loader, test_loader = make_loaders(cfg)
    model = FoveatedGlimpseModel(cfg.patch_size, cfg.hidden_dim, cfg.glimpse_dim, cfg.dropout_p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f'Training on device={device} | train_size={cfg.train_size} | test_size={cfg.test_size}')
    print(f'glimpses={cfg.n_glimpses} | hidden={cfg.hidden_dim} | patch={cfg.patch_size} | seed={cfg.seed}')

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, opt, cfg, device)
        print(
            f'Epoch {epoch:02d} | '
            f'loss={train_stats["loss"]:.4f} | '
            f'train_acc={train_stats["acc"]:.4f} | '
            f'train_recon={train_stats["recon_mse"]:.4f}'
        )

    rand_stats = evaluate_policy(model, test_loader, cfg, device, policy='random')
    efe_stats = evaluate_policy(model, test_loader, cfg, device, policy='efe_lite')

    print('\nFinal evaluation (same glimpse budget):')
    print_policy_report('Random', rand_stats)
    print_policy_report('EFE-lite', efe_stats)

    acc_gain = efe_stats['acc'] - rand_stats['acc']
    recon_gain = rand_stats['recon_mse'] - efe_stats['recon_mse']
    print(f'  Δacc (EFE-lite - Random)   = {acc_gain:+.4f}')
    print(f'  Δrecon (Random - EFE-lite) = {recon_gain:+.4f}')

    tol = 1e-9
    assert abs(rand_stats['acc_by_glimpse'][0] - efe_stats['acc_by_glimpse'][0]) < tol
    assert abs(rand_stats['recon_by_glimpse'][0] - efe_stats['recon_by_glimpse'][0]) < tol

    print('\nPer-glimpse deltas:')
    for i, (ra, ea, rr, er) in enumerate(zip(
        rand_stats['acc_by_glimpse'],
        efe_stats['acc_by_glimpse'],
        rand_stats['recon_by_glimpse'],
        efe_stats['recon_by_glimpse'],
    ), start=1):
        print(f'  glimpse {i}: Δacc={ea - ra:+.4f} | Δrecon={rr - er:+.4f}')

    if cfg.plot:
        plot_demo(model, test_loader, cfg, device)


if __name__ == '__main__':
    main()
