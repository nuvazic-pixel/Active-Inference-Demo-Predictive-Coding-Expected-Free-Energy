<<<<<<< HEAD
# Predictive Processing & Active Inference Demos

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![Status](https://img.shields.io/badge/status-research--toy-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.2.0-informational)

A GitHub-ready repo with three progressively stronger demos:

1. **Linear predictive coding** — the smallest possible PyTorch teaching example.
2. **Predictive coding + active inference** — nonlinear hierarchy, precision weighting, and EFE-style policy scoring.
3. **Foveated MNIST active sensing** — a glimpse-based benchmark where guided sampling is compared against random sampling under the same observation budget.

> This repo is designed as a **teaching + experimentation** scaffold. It is not a paper-ready implementation of formal active inference.

---

## Why this repo exists

This project helps you move from:
- abstract predictive-processing ideas,

- to a runnable predictive-coding toy,

- to a minimal active-inference toy,
- to a real **perception–action benchmark**.


Core benchmark:

> **Under the same glimpse budget, guided sampling should beat random sampling.**

---

## Repo layout

```text
pp_active_inference_repo_v2/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .gitignore
├── docs/
│   ├── notes.md
│   └── ROADMAP.md
├── scripts/
│   ├── run_linear_demo.py
│   ├── run_active_demo.py
│   └── run_mnist_demo.py
└── src/
    └── pp_active_inference/
        ├── __init__.py
        ├── cli.py
        ├── mini_pc_pytorch.py
        ├── mini_pc_active_inference.py
        └── mnist_foveated_active_inference_lite.py
```

---

## Quick start

### 1) Create a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install

```bash
pip install -e .
```

Or, if you prefer: Test2

```bash
pip install -r requirements.txt
```

---

## CLI commands

After `pip install -e .`, these commands are available:

```bash
pc-linear-demo
pc-active-demo
mnist-foveated-demo
```

You can still run the wrapper scripts directly if you prefer.

---

## Demo 1 — Linear predictive coding

### What it does
- learns a tiny linear generative hierarchy,
- infers latent states with local predictive-coding updates,
- updates weights with Hebbian-like rules.

### Run
```bash
pc-linear-demo --epochs 200 --infer-steps 30 --hidden 16 --top 4 --plot
```

### Why it matters
This is the cleanest stepping stone into predictive coding before adding active inference.

---

## Demo 2 — Predictive coding + active inference

### What it does
- nonlinear generative path (`tanh`),
- precision-weighted inference,
- EFE-style policy scoring,
- ablation modes:
  - `baseline`
  - `random_policy`
  - `risk_only`
  - `no_precision`
- metrics for reconstruction MSE, latent norms, and policy quality.

### Lite demo
```bash
pc-active-demo --plot
```

### Stronger run
```bash
pc-active-demo --epochs 60 --infer-steps 12 --hidden 6 --seed 42 --log-every 5 --plot
```

### Ablation examples
```bash
pc-active-demo --ablation baseline --epochs 60 --infer-steps 12 --hidden 6 --seed 42
pc-active-demo --ablation random_policy --epochs 60 --infer-steps 12 --hidden 6 --seed 42
pc-active-demo --ablation risk_only --epochs 60 --infer-steps 12 --hidden 6 --seed 42
pc-active-demo --ablation no_precision --epochs 60 --infer-steps 12 --hidden 6 --seed 42
```

---

## Demo 3 — Foveated MNIST active inference lite

### What it does
- trains a recurrent glimpse model on MNIST,
- observes only fixed `8x8` patches,
- uses a discrete `3x3` glimpse grid,
- compares:
  - **random glimpses**
  - **EFE-lite uncertainty-guided glimpses**
- predicts:
  - digit class,
  - full-image reconstruction.

### Run
```bash
mnist-foveated-demo --epochs 5 --seed 42 --plot
```

### Success criterion
You want the final report to show:
- positive `Δacc = EFE-lite acc - Random acc`
- positive `Δrecon = Random recon - EFE-lite recon`

That means guided glimpses beat random glimpses under the same observation budget.

---

## Recommended workflow

1. **Prove the toy works**
   ```bash
   pc-active-demo --plot
   ```
2. **Check reproducibility**
   Change only the seed.
3. **Run ablations**
   See which component is carrying behavior.
4. **Move to MNIST**
   ```bash
   mnist-foveated-demo --epochs 5 --seed 42 --plot
   ```
5. **Test robustness**
   Compare across multiple seeds and glimpse budgets.

---

## Suggested experiment matrix

### Foveated MNIST reproducibility
```bash
mnist-foveated-demo --epochs 5 --glimpses 3 --seed 42
mnist-foveated-demo --epochs 5 --glimpses 3 --seed 123
mnist-foveated-demo --epochs 5 --glimpses 3 --seed 999

mnist-foveated-demo --epochs 5 --glimpses 5 --seed 42
mnist-foveated-demo --epochs 5 --glimpses 5 --seed 123
mnist-foveated-demo --epochs 5 --glimpses 5 --seed 999

mnist-foveated-demo --epochs 5 --glimpses 7 --seed 42
mnist-foveated-demo --epochs 5 --glimpses 7 --seed 123
mnist-foveated-demo --epochs 5 --glimpses 7 --seed 999
```

What you want to see:
- EFE-lite wins in at least **2/3 seeds**,
- the guided advantage is strongest at lower glimpse budgets,
- the per-glimpse advantage appears **early**, not only at the end.

---

## Limitations

### This repo is
- a clean teaching repo,
- a working predictive-coding toy,
- a working active-sensing benchmark,
- a strong foundation for extension.

### This repo is not
- a formal proof of active inference,
- a full Bayesian sequence model,
- a theory of consciousness,
- a paper-ready benchmark suite.

---

## Publishing to GitHub

### Replace placeholders in `pyproject.toml`
Update:
- `Homepage`
- `Issues`

with your actual GitHub repo URL.

### Optional badge polish
If your repo is public, you can later add badges for:
- CI status
- release version
- downloads
- docs build

---

## License

MIT
=======
# Active-Inference-Demo-Predictive-Coding-Expected-Free-Energy
A 2-level hierarchical predictive coding network with:  • Nonlinear generative model (tanh)  • Precision-weighted inference  • Active inference via EFE policy selection  • Saccade-like actions to reduce uncertainty  Features:  • Inner loop: Variational inference on latents  • Outer loop: Hebbian-like parameter learning 
>>>>>>> 29804b2232ad9e373c25ee63f1ea5af0e5bd7964
