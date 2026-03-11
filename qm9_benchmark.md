# QM9 Benchmark

## Name

`qm9-simple`

## Goal

A minimal benchmark for unconditional 3D molecule generation that is simple
enough for fast single-GPU iteration.

Each sample contains:

- atom types
- 3D positions
- a node mask

This is the molecule analogue of the small, fixed, easy-to-iterate text setup
in the original repo.

## Dataset

Use the official QM9 XYZ archive:

- source: [Ramakrishnan et al., Scientific Data 2014](https://www.nature.com/articles/sdata201422)
- download path used by `prepare_mol.py`: [figshare](https://ndownloader.figshare.com/files/3195389)

This benchmark intentionally uses a simple repo-local preprocessing path:

- parse all `.xyz` files from the archive
- keep a single equilibrium geometry per molecule
- center coordinates per molecule
- no bond graph preprocessing
- no RDKit requirement for training

## Splits

Use a deterministic random split with `seed=42`:

- train: 100,000 molecules
- val: 10,000 molecules
- test: remainder

This is not the only QM9 split used in the literature, but it is simple,
stable, and sufficient for fast internal iteration.

## Variants

### Official benchmark

Use hydrogens:

- atom vocabulary: `H, C, N, O, F`
- expected max nodes: about `29`

### Tiny dev benchmark

Use `--remove-h` for faster iteration:

- atom vocabulary: `C, N, O, F`
- expected max nodes: about `9`

This is the recommended fast loop during early model development, but the
official benchmark should stay the hydrogen-inclusive version.

## Tensor Format

Saved tensors are dense and padded:

- `atom_types`: `[num_mols, max_nodes]`, `int16`, pad value `-1`
- `positions`: `[num_mols, max_nodes, 3]`, `float32`
- `mask`: `[num_mols, max_nodes]`, `bool`
- `num_atoms`: `[num_mols]`, `int16`

The coordinates of valid atoms are centered to zero mean per molecule.

## Training Objective

Recommended first target:

- unconditional E(3)-equivariant diffusion or flow matching

The benchmark does not lock the exact model family, but it does lock the data
format and the keep/discard metric.

## Inner-Loop Metric

Use one cheap scalar for every 5-minute run:

- `val_loss`

Interpretation:

- lower is better
- this should be a fixed held-out denoising or generative validation loss

Do not use validity or novelty as the per-run optimization target.

## Periodic Sample Metrics

Run these less frequently on checkpoints worth keeping:

1. `mol_stable`
2. `atm_stable`
3. `validity`
4. `uniqueness`
5. `novelty`

Optional:

1. node-count JSD
2. atom-type JSD
3. pairwise-distance Wasserstein distance

## Expected Run Summary

The molecule training script should print:

```text
---
val_loss:         0.123456
training_seconds: 300.1
total_seconds:    321.4
peak_vram_mb:     8123.4
samples_per_sec:  1456.7
num_steps:        982
num_params_M:     12.8
max_nodes:        29
remove_h:         False
```

## Non-Goals

This benchmark is not:

- graph-conditioned conformer generation
- SMILES generation
- a chemistry-property benchmark

If the task later becomes conformer generation from a known molecular graph,
switch to `COV` and `MAT`. That is a different benchmark.
