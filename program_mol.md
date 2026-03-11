# autoresearch molecules

This repo now has a concrete target benchmark for 3D molecule generation:
[qm9_benchmark.md](/Users/yoel/autoresearch/qm9_benchmark.md).

## Setup

1. Read:
   - `README.md`
   - `qm9_benchmark.md`
   - `prepare_mol.py`
2. Prepare the data:
   - `python3 prepare_mol.py`
   - for faster dev iteration: `python3 prepare_mol.py --remove-h`
3. Confirm the processed dataset exists in `~/.cache/autoresearch-mol/qm9/processed/`

## Backends

Two training paths now exist:

1. Torch:
   - `python3 train_mol.py --device mps`
   - `python3 train_mol.py --device cuda`
2. MLX:
   - `python3 train_mol_mlx.py`

## Modeling Target

The target is **not** text and **not** SMILES.

The model should generate:

- `atom_types`
- `positions`
- `mask`

The preferred first model family is:

- E(3)-equivariant diffusion
- or E(3)-equivariant flow matching

## Experiment Metric

The keep/discard metric for short runs is:

- `val_loss`

Lower is better.

Do not use sample validity metrics as the inner-loop objective for every
5-minute experiment. They are too noisy.

## Sample Metrics

For checkpoints worth keeping, evaluate:

1. `mol_stable`
2. `atm_stable`
3. `validity`
4. `uniqueness`
5. `novelty`

## Simplicity Rule

Prefer:

- fewer moving parts
- stable losses
- dense padded tensors
- deterministic splits

Avoid:

- serialized text representations
- heavy chemistry preprocessing in the training loop
- expensive sample-based evaluation every run

## Current Baseline

`train_mol.py` now provides a minimal baseline that:

1. loads the processed QM9 tensors
2. trains for a fixed wall-clock budget
3. prints a summary with `val_loss`
4. works on `cpu`, `mps`, and `cuda`

`train_mol_mlx.py` provides the Apple Silicon MLX path for the same benchmark.

## Next Milestone

Improve `train_mol.py` while keeping the benchmark fixed:

1. improve `val_loss`
2. keep the loop stable for 5-minute autonomous runs
3. add sample-generation evaluation in a separate script
