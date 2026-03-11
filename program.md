# autoresearch molecules

This is an experiment to have the LLM do its own research on a 3D molecule
generator.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, e.g. `mar11`.
   The branch `autoresearch/mol-<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/mol-<tag>` from the current
   branch.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `qm9_benchmark.md` — fixed benchmark contract.
   - `prepare_mol.py` — fixed data prep and dataset utilities. Do not modify.
   - `train_mol_mlx.py` — the Apple Silicon MLX training file.
   - `train_mol.py` — the Torch training file for `cuda`, `mps`, and `cpu`.
4. **Choose the active backend for this machine**:
   - On Apple Silicon, default to `train_mol_mlx.py`.
   - On NVIDIA/CUDA, default to `train_mol.py --device cuda`.
   - Do not edit both training files in the same experiment unless the user
     explicitly asks for backend parity work.
5. **Verify data exists**:
   - For the tiny dev benchmark, check for
     `~/.cache/autoresearch-mol/qm9/processed/qm9_noh.pt`.
   - If it does not exist, run `python3 prepare_mol.py --remove-h`.
6. **Initialize `results_mol.tsv`**: create it with just the header row. The
   baseline will be recorded after the first run.
7. **Confirm and go**: confirm setup looks good.

Once the user confirms, kick off the experimentation.

## Experimentation

Each experiment runs on a single accelerator. The training script runs for a
**fixed 5-minute time budget** of measured training time. Use the tiny QM9-noH
benchmark first unless the user explicitly asks for full QM9 with hydrogens.

**Current default commands**

- Apple Silicon MLX:
  - `python3 train_mol_mlx.py --remove-h --time-budget 300`
- NVIDIA/CUDA:
  - `python3 train_mol.py --remove-h --device cuda --time-budget 300`

**What you CAN do**

- Modify the active training file:
  - `train_mol_mlx.py` on Apple Silicon
  - `train_mol.py` on CUDA
- Change model architecture, optimizer, hyperparameters, batch size, training
  schedule, and backend-local implementation details.

**What you CANNOT do**

- Modify `prepare_mol.py`. It is the fixed benchmark/data harness.
- Modify `qm9_benchmark.md` during experiment runs.
- Change the per-run metric from `val_loss`.
- Change the dataset split or silently switch benchmarks mid-run.

**The goal is simple: get the lowest `val_loss`.** Since the time budget is
fixed, experiments are compared on the same 5-minute budget.

**Memory** is a soft constraint. Some increase is acceptable if it improves
`val_loss`, but avoid wasteful blowups.

**Simplicity criterion**: all else equal, simpler is better. A tiny gain that
adds fragile complexity is usually not worth keeping. A comparable result with
less code is a win.

**The first run**: always establish the baseline before making changes.

## Output format

At the end of a run, the script should print a summary like:

```text
---
val_loss:         1.234567
val_pos_loss:     0.123456
val_atom_loss:    0.654321
training_seconds: 300.1
total_seconds:    319.0
peak_vram_mb:     8123.4
samples_per_sec:  1456.7
num_steps:        1200
num_params_M:     1.7
max_nodes:        9
remove_h:         True
backend:          mlx
```

Extract the key metric from the log with:

```bash
grep "^val_loss:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results_mol.tsv` (tab-separated).

The TSV has 6 columns:

```text
commit	backend	val_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. backend: `mlx`, `cuda`, `mps`, or `cpu`
3. `val_loss` achieved — use `0.000000` for crashes
4. peak memory in GB, round to `.1f` — use `0.0` for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what the experiment tried

Example:

```text
commit	backend	val_loss	memory_gb	status	description
a1b2c3d	mlx	1.612342	0.0	keep	baseline tiny qm9-noh
b2c3d4e	mlx	1.544210	0.1	keep	increase hidden dim to 256
c3d4e5f	mlx	1.620001	0.1	discard	add dropout
d4e5f6g	mlx	0.000000	0.0	crash	breaks masking logic
```

## The experiment loop

The experiment runs on a dedicated branch, e.g. `autoresearch/mol-mar11`.

LOOP FOREVER:

1. Look at the git state: current branch and commit.
2. Tune the active training file with one experimental idea.
3. git commit.
4. Run the experiment:
   - `python3 train_mol_mlx.py --remove-h --time-budget 300 > run.log 2>&1`
   - or `python3 train_mol.py --remove-h --device cuda --time-budget 300 > run.log 2>&1`
5. Read out the results:
   - `grep "^val_loss:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Read the traceback with:
   - `tail -n 50 run.log`
7. Record the results in `results_mol.tsv`. Do not commit the TSV file.
8. If `val_loss` improved (lower), keep the commit and advance the branch.
9. If `val_loss` is equal or worse, reset back to where the run started.

The idea is: try one idea, run a
fixed-time experiment, keep only wins, and iterate indefinitely.

**Timeout**: a run should be about 5 minutes of training time plus startup and
validation overhead. If a run exceeds 10 minutes wall clock, kill it and treat
it as a failure.

**Crashes**: if a run crashes for a fixable reason, fix it and rerun. If the
idea itself is bad, log `crash`, revert, and move on.

**NEVER STOP**: once the loop begins, do not pause to ask the human whether to
continue. Keep iterating until explicitly interrupted.

## Important nuance

`val_loss` is the inner-loop optimization target. Sample-quality metrics such
as `mol_stable`, `atm_stable`, `validity`, `uniqueness`, and `novelty` should
be added in a separate evaluation script and run periodically, not on every
5-minute experiment.
