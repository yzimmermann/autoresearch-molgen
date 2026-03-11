# Molecule Adaptation Plan

## Goal

Adapt `autoresearch` from text autoregressive pretraining to unconditional 3D small-molecule generation:

- outputs are `atom_types` and `positions`
- model should be permutation-invariant over atoms and E(3)-equivariant over coordinates
- the inner-loop metric must stay cheap and stable enough for 5-minute autonomous experiments

## Recommended Datasets

### 1. QM9 for shakeout and fast iteration

Use `QM9` first to debug the pipeline and establish a single-GPU baseline.

- about 134k equilibrium molecules
- small molecules only
- atom vocabulary is tiny: `H, C, N, O, F`
- max atom count is small enough for dense message passing

Why it fits this repo:

- easy to preprocess
- low memory pressure
- stable benchmark for early experiments

Limitation:

- too narrow to be the final target if the goal is realistic drug-like molecules

### 2. GEOM-Drugs for the real target benchmark

Use `GEOM-Drugs` once the training/eval loop is working.

- much more realistic drug-like chemistry
- standard benchmark for 3D de novo molecule generation
- supports full atom types and 3D coordinates
- the standard preprocessing keeps the lowest-energy conformers per molecule

Practical note:

- substantially heavier than QM9
- keep only 1 conformer per molecule for the first unconditional setup
- move to multiple conformers later if needed

### 3. Optional: QMugs if you want a medicinal-chemistry-heavy follow-up

`QMugs` is a reasonable later-scale dataset, but it is less standard than `GEOM-Drugs` for direct comparison with prior 3D generation papers.

## Recommended Task Definition

Do **not** force this into a text/token objective.

Use:

- variable-size atom sets
- categorical atom types
- continuous 3D coordinates
- masked batches with centered coordinates

Recommended first model family:

- E(3)-equivariant diffusion or flow matching
- EGNN-style message passing

Do **not** start with:

- autoregressive tokenization of atoms
- voxel grids
- plain GPT over serialized coordinates

Those choices throw away the symmetry structure that makes 3D molecular generation tractable.

## Metrics

### Primary inner-loop metric

Use a fixed validation generative loss:

- `val_nll` if using a likelihood-based diffusion/flow objective
- otherwise a fixed held-out denoising loss

Reason:

- cheap
- low variance
- suitable for 5-minute compare-and-keep experiments

Do **not** use validity/novelty as the keep/discard metric for every short run. Those are too noisy and too sample-dependent for the inner loop.

### Secondary sample-quality metrics

Run these on a fixed number of generated samples for checkpoints worth keeping:

1. `mol_stable`
2. `atm_stable`
3. `validity`
4. `uniqueness`
5. `novelty`

These are the standard de novo 3D metrics used in EDM-style evaluation.

### Distribution-matching metrics

Track at least:

1. node-count distribution divergence
2. atom-type distribution divergence
3. pairwise-distance distribution divergence

Good choices:

- JSD or symmetric KL for discrete histograms
- Wasserstein distance for pairwise distances

These catch degenerate generators that look valid chemically but miss the dataset geometry.

### Optional chemistry metrics

After reconstructing bonds with RDKit or distance-based bond inference, track:

1. molecular weight
2. logP
3. QED
4. SA score

These are useful on `GEOM-Drugs`, especially if the eventual goal is drug-like design.

### Only if you switch to conformer generation

If the task becomes:

- input: 2D molecular graph
- output: 3D conformers

then use `COV` and `MAT` as primary benchmark metrics instead. Those are for conformer generation, not unconditional molecule generation.

## Recommended Repo Changes

Keep the text path intact and add a molecule path in parallel.

### New files

- `prepare_mol.py`
- `train_mol.py`
- `eval_mol.py`
- `program_mol.md`

### `prepare_mol.py`

Responsibilities:

- download or ingest `QM9` and `GEOM-Drugs`
- preprocess to tensors:
  - `atom_types`
  - `positions`
  - `mask`
  - optional `charges`
- build deterministic train/val/test splits
- store a small fixed validation subset for the inner-loop metric

Avoid tokenizers entirely.

### `train_mol.py`

Responsibilities:

- define the EGNN/diffusion model
- train under the same fixed wall-clock budget idea
- print a summary like:
  - `val_nll`
  - `training_seconds`
  - `peak_vram_mb`
  - `samples_per_sec`
  - `num_params_M`

### `eval_mol.py`

Responsibilities:

- generate a fixed number of molecules
- compute:
  - stability
  - validity
  - uniqueness
  - novelty
  - distribution-matching metrics

This should be separate from the training loop so the 5-minute optimization metric stays cheap.

### `program_mol.md`

This should replace the text-specific instructions in `program.md` with:

- the allowed files to modify
- the main metric: `val_nll`
- the secondary metrics to report periodically
- the rule that sample-based metrics are not used for every keep/discard decision

## Minimal First Milestone

1. Implement `QM9` preprocessing.
2. Train an unconditional EGNN diffusion baseline on QM9.
3. Use `val_nll` as the inner-loop metric.
4. Periodically compute `mol_stable`, `atm_stable`, `validity`, `uniqueness`, `novelty`.
5. Once stable, port the same path to `GEOM-Drugs`.

## Sources

- QM9: [Ramakrishnan et al., Scientific Data 2014](https://www.nature.com/articles/sdata201422)
- GEOM: [Axelrod and Gomez-Bombarelli, NeurIPS 2022 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a9e1dff86b38b70c7ab2fffd4d2fbbde-Abstract-Datasets_and_Benchmarks.html)
- EDM: [Hoogeboom et al., ICML 2022](https://proceedings.mlr.press/v162/hoogeboom22a.html)
- GeoDiff: [Xu et al., ICLR 2022](https://openreview.net/forum?id=PzcvxEMzvQC)
- MOSES: [Polykovskiy et al., Frontiers in Pharmacology 2020](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full)
