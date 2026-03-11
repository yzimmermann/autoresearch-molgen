# autoresearch-molgen

Autonomous research loop for 3D molecule generation on a single accelerator.

This repo adapts the `autoresearch` workflow to generative modeling over:

- atom types
- 3D coordinates
- variable-size molecular point sets

The current benchmark is a deliberately simple QM9 setup for fast iteration on:

- Apple Silicon via `MLX`
- NVIDIA GPUs via `PyTorch CUDA`
- fallback `PyTorch MPS/CPU`

## What This Repo Is

The original idea is unchanged:

1. give an agent a small but real training setup
2. run fixed-time experiments
3. compare one scalar metric
4. keep only improvements
5. let the loop continue autonomously

What changes here is the domain:

- from text pretraining
- to unconditional 3D molecule generation

The current path is:

- benchmark: `QM9-noH` for the tiny dev loop
- official benchmark target: full `QM9`
- inner-loop metric: `val_loss`
- periodic chemistry metrics: planned in a separate evaluation script

## Quick Start

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and one
accelerator:

- Apple Silicon for the `MLX` path
- a single NVIDIA GPU for the `CUDA` path

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Prepare the tiny QM9-noH benchmark (one-time)
uv run prepare_mol.py --remove-h

# 4a. Manually run a single Apple Silicon MLX experiment (~5 min)
uv run train_mol_mlx.py --remove-h --time-budget 300

# 4b. Or manually run a single CUDA experiment (~5 min)
uv run train_mol.py --remove-h --device cuda --time-budget 300
```

If the above commands work, the setup is working and you can go into
autonomous molecule research mode.

For the full QM9 benchmark instead of the tiny dev benchmark:

```bash
uv run prepare_mol.py
uv run train_mol_mlx.py --with-h --time-budget 300
```

Or on CUDA:

```bash
uv run train_mol.py --with-h --device cuda --time-budget 300
```

## Repo Layout

```text
prepare_mol.py     — fixed QM9 preprocessing and dataloading utilities
train_mol.py       — Torch trainer for cuda/mps/cpu
train_mol_mlx.py   — MLX trainer for Apple Silicon
qm9_benchmark.md   — fixed benchmark contract
program.md         — autonomous research instructions for molecules
pyproject.toml     — dependencies
```

## Benchmark

The benchmark contract lives in
[qm9_benchmark.md](qm9_benchmark.md).

Summary:

- dataset: official QM9 XYZ archive
- tiny dev benchmark: `QM9-noH`
- official benchmark: full QM9 with hydrogens
- tensor format:
  - `atom_types`
  - `positions`
  - `mask`
  - `num_atoms`
- per-run objective: `val_loss`

## Training Commands

Use MLX on Apple Silicon by default. The Torch `mps` path works, but it is
currently much slower.

Prepare the tiny benchmark:

```bash
python3 prepare_mol.py --remove-h
```

This creates:

```text
~/.cache/autoresearch-mol/qm9/processed/qm9_noh.pt
```

Run MLX on Apple Silicon:

```bash
python3 train_mol_mlx.py --remove-h --time-budget 300
```

Run CUDA with PyTorch:

```bash
python3 train_mol.py --remove-h --device cuda --time-budget 300
```

Run Torch `mps` fallback on Apple GPU:

```bash
python3 train_mol.py --remove-h --device mps --time-budget 300
```

## What The Training Scripts Do

The current trainers implement a minimal EGNN-style denoising baseline.

They:

1. load fixed QM9 tensors
2. corrupt atom identities and coordinates
3. predict denoised positions and masked atom types
4. train for a fixed wall-clock budget
5. print a final summary with `val_loss`

Example summary:

```text
---
val_loss:         0.869344
val_pos_loss:     0.075149
val_atom_loss:    0.568748
training_seconds: 300.1
total_seconds:    302.2
peak_vram_mb:     0.0
samples_per_sec:  1771.5
num_steps:        2077
num_params_M:     1.7
max_nodes:        9
remove_h:         True
backend:          mlx
```

## Research Loop

The repo is set up to support an autonomous keep/discard loop:

1. edit the active training file
2. run a fixed 5-minute experiment
3. compare `val_loss`
4. keep only improvements
5. log runs to `results_mol.tsv`

The active instructions live in
[program.md](program.md).

## Running The Agent

Spin up Codex, Claude, or another coding agent in this repo, disable
permissions, and prompt it with [program.md](program.md).

For example:

```text
Hi have a look at program.md and let's kick off a new molecule experiment. Let's do the setup first.
```

`program.md` is the lightweight control file for the autonomous research loop.

## Current Status

What works now:

- fixed QM9 preprocessing
- tiny benchmark generation path
- MLX training on Apple Silicon
- Torch training on `cuda`, `mps`, and `cpu`
- molecule-specific `program.md`
- baseline autonomous loop setup

What is still missing:

- separate sample evaluation script
- stability/validity/uniqueness/novelty reporting
- stronger training-safety rails for longer unattended search
- larger benchmarks like GEOM-Drugs

## License

MIT
