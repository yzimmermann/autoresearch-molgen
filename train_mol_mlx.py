"""
Minimal MLX molecule-training script for the QM9 benchmark.

This mirrors the dense padded-tensor benchmark used by train_mol.py, but runs
on MLX so it can use Apple Silicon GPU directly.

Usage:
    python3 train_mol_mlx.py
    python3 train_mol_mlx.py --remove-h
    python3 train_mol_mlx.py --mlx-device cpu
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from prepare_mol import TIME_BUDGET, load_dataset

REMOVE_H = True
BATCH_SIZE = 256
HIDDEN_DIM = 192
N_LAYERS = 6
TIME_EMB_DIM = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
SIGMA_MIN = 0.02
SIGMA_MAX = 1.0
ATOM_CORRUPT_MIN = 0.15
ATOM_CORRUPT_MAX = 0.55
ATOM_LOSS_WEIGHT = 1.0
POS_LOSS_WEIGHT = 4.0


def load_split(remove_h: bool, split: str) -> dict[str, np.ndarray]:
    dataset = load_dataset(remove_h=remove_h)
    tensors = dataset["splits"][split]
    return {
        "atom_types": tensors["atom_types"].numpy().astype(np.int32),
        "positions": tensors["positions"].numpy().astype(np.float32),
        "mask": tensors["mask"].numpy().astype(np.bool_),
        "num_atoms": tensors["num_atoms"].numpy().astype(np.int32),
    }


def iterate_batches(split_data: dict[str, np.ndarray], batch_size: int, shuffle: bool, seed: int):
    size = split_data["num_atoms"].shape[0]
    rng = np.random.default_rng(seed)
    while True:
        indices = np.arange(size)
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, size, batch_size):
            batch_idx = indices[start:start + batch_size]
            if batch_idx.shape[0] < batch_size and shuffle:
                continue
            yield {
                key: mx.array(value[batch_idx])
                for key, value in split_data.items()
            }
        if not shuffle:
            break


def masked_mean(x: mx.array, mask: mx.array, axis: int, keepdims: bool = False) -> mx.array:
    denom = mx.maximum(mask.sum(axis=axis, keepdims=keepdims), 1.0)
    return (x * mask).sum(axis=axis, keepdims=keepdims) / denom


def remove_masked_mean(x: mx.array, mask: mx.array) -> mx.array:
    mask_f = mask[..., None].astype(mx.float32)
    return (x - masked_mean(x, mask_f, axis=1, keepdims=True)) * mask_f


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def __call__(self, sigma: mx.array) -> mx.array:
        sigma = mx.log(mx.maximum(sigma, 1e-6))
        half_dim = self.emb_dim // 2
        freq = mx.exp(
            mx.arange(half_dim, dtype=mx.float32)
            * (-math.log(10_000.0) / max(half_dim - 1, 1))
        )
        args = sigma * freq[None, :]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        if emb.shape[-1] < self.emb_dim:
            pad = mx.zeros((emb.shape[0], self.emb_dim - emb.shape[-1]), dtype=emb.dtype)
            emb = mx.concatenate([emb, pad], axis=-1)
        return self.proj(emb)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class EGNNBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, h: mx.array, x: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
        n_nodes = mask.shape[1]
        mask_f = mask.astype(mx.float32)
        edge_mask = (mask[:, :, None] & mask[:, None, :])[..., None]
        diag = mx.eye(n_nodes, dtype=mx.bool_)[None, ..., None]
        edge_mask = edge_mask & (~diag)

        x_i = mx.expand_dims(x, 2)
        x_j = mx.expand_dims(x, 1)
        diff = x_i - x_j
        dist2 = mx.sum(diff * diff, axis=-1, keepdims=True)

        h_i = mx.broadcast_to(mx.expand_dims(h, 2), (h.shape[0], n_nodes, n_nodes, h.shape[-1]))
        h_j = mx.broadcast_to(mx.expand_dims(h, 1), (h.shape[0], n_nodes, n_nodes, h.shape[-1]))
        edge_inputs = mx.concatenate([h_i, h_j, dist2], axis=-1)
        messages = self.edge_mlp(edge_inputs) * edge_mask.astype(mx.float32)

        denom = mx.maximum(edge_mask.astype(mx.float32).sum(axis=2), 1.0)
        coord_scale = self.coord_mlp(messages)
        coord_update = (diff * coord_scale * edge_mask.astype(mx.float32)).sum(axis=2) / denom
        x = remove_masked_mean(x + coord_update, mask)

        node_update = self.node_mlp(mx.concatenate([h, messages.sum(axis=2) / denom], axis=-1))
        h = self.norm(h + node_update)
        h = h * mask_f[..., None]
        return h, x


class MoleculeDenoiser(nn.Module):
    def __init__(self, num_atom_types: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.pad_token = num_atom_types
        self.mask_token = num_atom_types + 1

        self.atom_embed = nn.Embedding(num_atom_types + 2, hidden_dim)
        self.time_embed = TimeEmbedding(TIME_EMB_DIM)
        self.time_proj = nn.Linear(TIME_EMB_DIM, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim + 3, hidden_dim)
        self.layers = [EGNNBlock(hidden_dim) for _ in range(n_layers)]
        self.atom_head = MLP(hidden_dim, hidden_dim, num_atom_types)

    def __call__(self, atom_input: mx.array, x_noisy: mx.array, sigma: mx.array, mask: mx.array):
        mask_f = mask.astype(mx.float32)
        h = self.atom_embed(atom_input)
        t_emb = mx.expand_dims(self.time_proj(self.time_embed(sigma)), 1)
        h = self.input_proj(mx.concatenate([h + t_emb, x_noisy], axis=-1))
        h = h * mask_f[..., None]
        x = x_noisy
        for layer in self.layers:
            h, x = layer(h, x, mask)
        atom_logits = self.atom_head(h)
        x = remove_masked_mean(x, mask)
        return atom_logits, x


def corruption_prob(sigmas: mx.array) -> mx.array:
    frac = (mx.log(sigmas) - math.log(SIGMA_MIN)) / (math.log(SIGMA_MAX) - math.log(SIGMA_MIN))
    frac = mx.clip(frac, 0.0, 1.0)
    return ATOM_CORRUPT_MIN + frac * (ATOM_CORRUPT_MAX - ATOM_CORRUPT_MIN)


def corrupt_batch(atom_types: mx.array, positions: mx.array, mask: mx.array, model: MoleculeDenoiser, rng: np.random.Generator):
    sigma_u = mx.array(rng.random((atom_types.shape[0], 1), dtype=np.float32))
    sigma = mx.exp(math.log(SIGMA_MIN) + sigma_u * (math.log(SIGMA_MAX) - math.log(SIGMA_MIN)))
    noise = mx.array(rng.standard_normal(positions.shape, dtype=np.float32))
    noise = remove_masked_mean(noise * mask[..., None].astype(mx.float32), mask)
    x_noisy = remove_masked_mean(positions + sigma[..., None] * noise, mask)

    atom_input = atom_types
    atom_input = mx.where(mask, atom_input, mx.full(atom_input.shape, model.pad_token, dtype=atom_input.dtype))
    corrupt_p = corruption_prob(sigma)
    random_mask = mx.array(rng.random(mask.shape, dtype=np.float32))
    corrupt_mask = (random_mask < corrupt_p) & mask
    atom_input = mx.where(corrupt_mask, mx.full(atom_input.shape, model.mask_token, dtype=atom_input.dtype), atom_input)
    return atom_input, x_noisy, sigma, corrupt_mask


def compute_losses(
    atom_logits: mx.array,
    pred_positions: mx.array,
    atom_types: mx.array,
    clean_positions: mx.array,
    mask: mx.array,
    corrupt_mask: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    mask_f = mask[..., None].astype(mx.float32)
    pos_loss = mx.sum(((pred_positions - clean_positions) ** 2) * mask_f)
    pos_loss = pos_loss / (mx.maximum(mask_f.sum(), 1.0) * clean_positions.shape[-1])

    atom_loss_full = nn.losses.cross_entropy(atom_logits, atom_types, reduction="none")
    atom_loss = mx.sum(atom_loss_full * corrupt_mask.astype(mx.float32)) / mx.maximum(corrupt_mask.astype(mx.float32).sum(), 1.0)
    total_loss = POS_LOSS_WEIGHT * pos_loss + ATOM_LOSS_WEIGHT * atom_loss
    return total_loss, pos_loss, atom_loss


@dataclass
class TrainState:
    step: int = 0
    total_samples: int = 0
    smooth_loss: float = 0.0
    measured_training_time: float = 0.0


def loss_fn(model: MoleculeDenoiser, atom_types, positions, mask, atom_input, x_noisy, sigma, corrupt_mask):
    atom_logits, pred_positions = model(atom_input, x_noisy, sigma, mask)
    total_loss, _, _ = compute_losses(atom_logits, pred_positions, atom_types, positions, mask, corrupt_mask)
    return total_loss


def metrics_fn(model: MoleculeDenoiser, atom_types, positions, mask, atom_input, x_noisy, sigma, corrupt_mask):
    atom_logits, pred_positions = model(atom_input, x_noisy, sigma, mask)
    return compute_losses(atom_logits, pred_positions, atom_types, positions, mask, corrupt_mask)


def evaluate(model: MoleculeDenoiser, val_data: dict[str, np.ndarray], batch_size: int):
    model.eval()
    total_loss = 0.0
    total_pos = 0.0
    total_atom = 0.0
    total_batches = 0
    loader = iterate_batches(val_data, batch_size=batch_size, shuffle=False, seed=1234)
    for batch_idx, batch in enumerate(loader):
        rng = np.random.default_rng(1234 + batch_idx)
        atom_input, x_noisy, sigma, corrupt_mask = corrupt_batch(
            batch["atom_types"], batch["positions"], batch["mask"], model, rng
        )
        loss, pos_loss, atom_loss = metrics_fn(
            model,
            batch["atom_types"],
            batch["positions"],
            batch["mask"],
            atom_input,
            x_noisy,
            sigma,
            corrupt_mask,
        )
        mx.eval(loss, pos_loss, atom_loss)
        total_loss += float(loss.item())
        total_pos += float(pos_loss.item())
        total_atom += float(atom_loss.item())
        total_batches += 1
        if (batch_idx + 1) * batch_size >= val_data["num_atoms"].shape[0]:
            break
    model.train()
    return total_loss / total_batches, total_pos / total_batches, total_atom / total_batches


def count_parameters(tree) -> int:
    leaves = tree_leaves(tree)
    return int(sum(int(np.prod(leaf.shape)) for leaf in leaves))


def tree_leaves(tree):
    if isinstance(tree, dict):
        leaves = []
        for value in tree.values():
            leaves.extend(tree_leaves(value))
        return leaves
    if isinstance(tree, (list, tuple)):
        leaves = []
        for value in tree:
            leaves.extend(tree_leaves(value))
        return leaves
    return [tree]


def main():
    parser = argparse.ArgumentParser(description="Train a minimal QM9 molecule denoiser with MLX")
    parser.set_defaults(remove_h=REMOVE_H)
    parser.add_argument("--remove-h", action="store_true", dest="remove_h", help="Use the QM9-noH tiny benchmark")
    parser.add_argument("--with-h", action="store_false", dest="remove_h", help="Use the full QM9 benchmark with hydrogens")
    parser.add_argument("--mlx-device", choices=["gpu", "cpu"], default="gpu", help="MLX default device")
    parser.add_argument("--time-budget", type=float, default=TIME_BUDGET, help="Wall-clock training budget in seconds")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    if args.mlx_device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    np.random.seed(42)
    t_start = time.time()

    meta = load_dataset(remove_h=args.remove_h)["meta"]
    train_data = load_split(args.remove_h, "train")
    val_data = load_split(args.remove_h, "val")
    num_atom_types = len(meta["atom_decoder"])
    max_nodes = int(meta["max_nodes"])

    model = MoleculeDenoiser(num_atom_types=num_atom_types, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    num_params = count_parameters(model.parameters())

    print(f"Backend: mlx")
    print(f"Device: {mx.default_device()}")
    print(f"Dataset: qm9{'-noh' if args.remove_h else ''}")
    print(f"Model: hidden_dim={HIDDEN_DIM} layers={N_LAYERS} params={num_params / 1e6:.2f}M")
    print(f"Batch size: {args.batch_size}")
    print(f"Time budget: {args.time_budget}s")

    state = TrainState()
    batch_iter = iterate_batches(train_data, batch_size=args.batch_size, shuffle=True, seed=42)
    rng = np.random.default_rng(42)

    while True:
        batch = next(batch_iter)
        atom_input, x_noisy, sigma, corrupt_mask = corrupt_batch(
            batch["atom_types"], batch["positions"], batch["mask"], model, rng
        )
        t0 = time.time()
        loss, grads = loss_and_grad(
            model,
            batch["atom_types"],
            batch["positions"],
            batch["mask"],
            atom_input,
            x_noisy,
            sigma,
            corrupt_mask,
        )
        optimizer.update(model, grads)
        pos_loss, atom_loss = None, None
        total_loss, pos_loss, atom_loss = metrics_fn(
            model,
            batch["atom_types"],
            batch["positions"],
            batch["mask"],
            atom_input,
            x_noisy,
            sigma,
            corrupt_mask,
        )
        mx.eval(loss, total_loss, pos_loss, atom_loss, model.parameters(), optimizer.state)
        dt = time.time() - t0

        state.step += 1
        state.total_samples += int(batch["atom_types"].shape[0])
        state.measured_training_time += dt
        loss_value = float(total_loss.item())
        state.smooth_loss = 0.98 * state.smooth_loss + 0.02 * loss_value if state.step > 1 else loss_value
        samples_per_sec = args.batch_size / max(dt, 1e-6)

        print(
            f"\rstep {state.step:05d} | loss {state.smooth_loss:.4f} | "
            f"pos {float(pos_loss.item()):.4f} | atom {float(atom_loss.item()):.4f} | "
            f"samples/sec {samples_per_sec:.1f}",
            end="",
            flush=True,
        )

        if state.measured_training_time >= args.time_budget:
            break

    print()
    val_loss, val_pos_loss, val_atom_loss = evaluate(model, val_data, args.batch_size)
    total_seconds = time.time() - t_start

    print("---")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"val_pos_loss:     {val_pos_loss:.6f}")
    print(f"val_atom_loss:    {val_atom_loss:.6f}")
    print(f"training_seconds: {state.measured_training_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     0.0")
    print(f"samples_per_sec:  {state.total_samples / max(state.measured_training_time, 1e-6):.1f}")
    print(f"num_steps:        {state.step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"max_nodes:        {max_nodes}")
    print(f"remove_h:         {args.remove_h}")
    print(f"backend:          mlx")


if __name__ == "__main__":
    main()
