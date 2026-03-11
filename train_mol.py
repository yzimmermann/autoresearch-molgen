"""
Minimal molecule-training script for the QM9 benchmark.

This is a dense padded-tensor baseline intended for fast iteration:

- loads QM9 tensors from prepare_mol.py
- trains a small EGNN-style denoiser
- uses a fixed wall-clock budget
- reports final val_loss

Usage:
    python3 train_mol.py
    python3 train_mol.py --remove-h
    python3 train_mol.py --device mps
    python3 train_mol.py --device cuda
    python3 train_mol.py --time-budget 10 --cpu
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prepare_mol import TIME_BUDGET, load_dataset

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

REMOVE_H = True
BATCH_SIZE = 256
HIDDEN_DIM = 192
N_LAYERS = 6
TIME_EMB_DIM = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
DROPOUT = 0.0
SIGMA_MIN = 0.02
SIGMA_MAX = 1.0
ATOM_CORRUPT_MIN = 0.15
ATOM_CORRUPT_MAX = 0.55
ATOM_LOSS_WEIGHT = 1.0
POS_LOSS_WEIGHT = 4.0
NUM_WORKERS = 0
COMPILE_MODEL = False


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

class TensorDatasetSplit(Dataset):
    def __init__(self, split_tensors: dict[str, torch.Tensor]):
        self.split_tensors = split_tensors

    def __len__(self) -> int:
        return int(self.split_tensors["num_atoms"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "atom_types": self.split_tensors["atom_types"][idx],
            "positions": self.split_tensors["positions"][idx],
            "mask": self.split_tensors["mask"][idx],
            "num_atoms": self.split_tensors["num_atoms"][idx],
        }


def make_dataloaders(remove_h: bool, batch_size: int, device: torch.device):
    dataset = load_dataset(remove_h=remove_h)
    pin_memory = device.type == "cuda"
    train_dataset = TensorDatasetSplit(dataset["splits"]["train"])
    val_dataset = TensorDatasetSplit(dataset["splits"]["val"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    return dataset["meta"], train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    return (x * mask).sum(dim=dim, keepdim=keepdim) / denom


def remove_masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    return (x - masked_mean(x, mask_f, dim=1, keepdim=True)) * mask_f


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-6).log()
        half_dim = self.emb_dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=sigma.device, dtype=sigma.dtype)
            * (-math.log(10_000.0) / max(half_dim - 1, 1))
        )
        args = sigma * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.emb_dim:
            emb = F.pad(emb, (0, self.emb_dim - emb.shape[-1]))
        return self.proj(emb)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGNNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.float()
        n_nodes = mask.shape[1]
        edge_mask = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1)
        diag = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
        edge_mask = edge_mask & (~diag)

        x_i = x[:, :, None, :]
        x_j = x[:, None, :, :]
        diff = x_i - x_j
        dist2 = diff.square().sum(dim=-1, keepdim=True)

        h_i = h[:, :, None, :].expand(-1, -1, n_nodes, -1)
        h_j = h[:, None, :, :].expand(-1, n_nodes, -1, -1)
        edge_inputs = torch.cat([h_i, h_j, dist2], dim=-1)
        messages = self.edge_mlp(edge_inputs) * edge_mask.float()

        denom = edge_mask.float().sum(dim=2).clamp_min(1.0)
        coord_scale = self.coord_mlp(messages)
        coord_update = (diff * coord_scale * edge_mask.float()).sum(dim=2) / denom
        x = remove_masked_mean(x + coord_update, mask)

        node_update = self.node_mlp(torch.cat([h, messages.sum(dim=2) / denom], dim=-1))
        h = self.norm(h + node_update)
        h = h * mask_f.unsqueeze(-1)
        return h, x


class MoleculeDenoiser(nn.Module):
    def __init__(self, num_atom_types: int, hidden_dim: int, n_layers: int, max_nodes: int):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.pad_token = num_atom_types
        self.mask_token = num_atom_types + 1
        self.max_nodes = max_nodes

        self.atom_embed = nn.Embedding(num_atom_types + 2, hidden_dim)
        self.time_embed = TimeEmbedding(TIME_EMB_DIM)
        self.time_proj = nn.Linear(TIME_EMB_DIM, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim + 3, hidden_dim)
        self.layers = nn.ModuleList([EGNNBlock(hidden_dim, DROPOUT) for _ in range(n_layers)])
        self.atom_head = MLP(hidden_dim, hidden_dim, num_atom_types, dropout=DROPOUT)

    def forward(self, atom_input: torch.Tensor, x_noisy: torch.Tensor, sigma: torch.Tensor, mask: torch.Tensor):
        mask_f = mask.float()
        h = self.atom_embed(atom_input)
        t_emb = self.time_proj(self.time_embed(sigma)).unsqueeze(1)
        h = self.input_proj(torch.cat([h + t_emb, x_noisy], dim=-1))
        h = h * mask_f.unsqueeze(-1)
        x = x_noisy
        for layer in self.layers:
            h, x = layer(h, x, mask)
        atom_logits = self.atom_head(h)
        x = remove_masked_mean(x, mask)
        return atom_logits, x


# ---------------------------------------------------------------------------
# Corruption and losses
# ---------------------------------------------------------------------------

def sample_sigmas(batch_size: int, device: torch.device) -> torch.Tensor:
    u = torch.rand(batch_size, 1, device=device)
    log_sigma = math.log(SIGMA_MIN) + u * (math.log(SIGMA_MAX) - math.log(SIGMA_MIN))
    return log_sigma.exp()


def masked_noise_like(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x) * mask.unsqueeze(-1)
    return remove_masked_mean(noise, mask)


def corruption_prob(sigmas: torch.Tensor) -> torch.Tensor:
    frac = (sigmas.log() - math.log(SIGMA_MIN)) / (math.log(SIGMA_MAX) - math.log(SIGMA_MIN))
    frac = frac.clamp(0.0, 1.0)
    return ATOM_CORRUPT_MIN + frac * (ATOM_CORRUPT_MAX - ATOM_CORRUPT_MIN)


def corrupt_batch(
    atom_types: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    model: MoleculeDenoiser,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = sample_sigmas(atom_types.shape[0], atom_types.device)
    noise = masked_noise_like(positions, mask)
    x_noisy = positions + sigma.unsqueeze(-1) * noise
    x_noisy = remove_masked_mean(x_noisy, mask)

    atom_input = atom_types.clone()
    atom_input[~mask] = model.pad_token
    corrupt_p = corruption_prob(sigma)
    corrupt_mask = (torch.rand_like(mask.float()) < corrupt_p) & mask
    atom_input[corrupt_mask] = model.mask_token

    return atom_input, x_noisy, sigma, corrupt_mask


def corrupt_batch_eval(
    atom_types: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    model: MoleculeDenoiser,
    batch_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(batch_seed)
    sigma_u = torch.rand(atom_types.shape[0], 1, generator=generator)
    sigma = (
        math.log(SIGMA_MIN)
        + sigma_u * (math.log(SIGMA_MAX) - math.log(SIGMA_MIN))
    ).exp().to(atom_types.device)
    noise = torch.randn(positions.shape, generator=generator, dtype=positions.dtype).to(atom_types.device)
    noise = remove_masked_mean(noise * mask.unsqueeze(-1), mask)
    x_noisy = remove_masked_mean(positions + sigma.unsqueeze(-1) * noise, mask)

    atom_input = atom_types.clone()
    atom_input[~mask] = model.pad_token
    corrupt_p = corruption_prob(sigma).cpu()
    random_mask = torch.rand(mask.shape, generator=generator)
    corrupt_mask = ((random_mask < corrupt_p) & mask.cpu()).to(mask.device)
    atom_input[corrupt_mask] = model.mask_token
    return atom_input, x_noisy, sigma, corrupt_mask


def compute_losses(
    atom_logits: torch.Tensor,
    pred_positions: torch.Tensor,
    atom_types: torch.Tensor,
    clean_positions: torch.Tensor,
    mask: torch.Tensor,
    corrupt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_f = mask.unsqueeze(-1).float()
    pos_loss = ((pred_positions - clean_positions).square() * mask_f).sum()
    pos_loss = pos_loss / (mask_f.sum().clamp_min(1.0) * clean_positions.shape[-1])

    if corrupt_mask.any():
        atom_loss = F.cross_entropy(atom_logits[corrupt_mask], atom_types[corrupt_mask])
    else:
        atom_loss = atom_logits.sum() * 0.0

    total_loss = POS_LOSS_WEIGHT * pos_loss + ATOM_LOSS_WEIGHT * atom_loss
    return total_loss, pos_loss.detach(), atom_loss.detach()


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    step: int = 0
    total_samples: int = 0
    smooth_loss: float = 0.0
    measured_training_time: float = 0.0


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=(device.type == "cuda"))
    return moved


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model: MoleculeDenoiser, val_loader: DataLoader, device: torch.device, autocast_ctx):
    model.eval()
    total_loss = 0.0
    total_pos = 0.0
    total_atom = 0.0
    total_batches = 0
    for batch_idx, batch in enumerate(val_loader):
        batch = move_batch(batch, device)
        atom_types = batch["atom_types"].long()
        positions = batch["positions"]
        mask = batch["mask"]

        atom_input, x_noisy, sigma, corrupt_mask = corrupt_batch_eval(
            atom_types, positions, mask, model, batch_seed=1234 + batch_idx
        )
        with autocast_ctx():
            atom_logits, pred_positions = model(atom_input, x_noisy, sigma, mask)
            loss, pos_loss, atom_loss = compute_losses(
                atom_logits, pred_positions, atom_types, positions, mask, corrupt_mask
            )
        total_loss += loss.item()
        total_pos += pos_loss.item()
        total_atom += atom_loss.item()
        total_batches += 1
    return total_loss / total_batches, total_pos / total_batches, total_atom / total_batches


def main():
    parser = argparse.ArgumentParser(description="Train a minimal QM9 molecule denoiser")
    parser.set_defaults(remove_h=REMOVE_H)
    parser.add_argument("--remove-h", action="store_true", dest="remove_h", help="Use the QM9-noH tiny benchmark")
    parser.add_argument("--with-h", action="store_false", dest="remove_h", help="Use the full QM9 benchmark with hydrogens")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Torch device")
    parser.add_argument("--time-budget", type=float, default=TIME_BUDGET, help="Wall-clock training budget in seconds")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = select_device(args.device, force_cpu=args.cpu)
    def autocast():
        if device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    meta, train_loader, val_loader = make_dataloaders(args.remove_h, args.batch_size, device)
    num_atom_types = len(meta["atom_decoder"])
    max_nodes = int(meta["max_nodes"])

    model = MoleculeDenoiser(
        num_atom_types=num_atom_types,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        max_nodes=max_nodes,
    ).to(device)
    if COMPILE_MODEL and hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    num_params = sum(p.numel() for p in model.parameters())
    state = TrainState()
    train_iter = infinite_loader(train_loader)

    print(f"Device: {device}")
    print(f"Dataset: qm9{'-noh' if args.remove_h else ''}")
    print(f"Model: hidden_dim={HIDDEN_DIM} layers={N_LAYERS} params={num_params / 1e6:.2f}M")
    print(f"Batch size: {args.batch_size}")
    print(f"Time budget: {args.time_budget}s")

    while True:
        batch = move_batch(next(train_iter), device)
        atom_types = batch["atom_types"].long()
        positions = batch["positions"]
        mask = batch["mask"]

        synchronize(device)
        t0 = time.time()

        atom_input, x_noisy, sigma, corrupt_mask = corrupt_batch(atom_types, positions, mask, model)
        with autocast():
            atom_logits, pred_positions = model(atom_input, x_noisy, sigma, mask)
            loss, pos_loss, atom_loss = compute_losses(
                atom_logits, pred_positions, atom_types, positions, mask, corrupt_mask
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        synchronize(device)
        dt = time.time() - t0

        state.step += 1
        state.total_samples += atom_types.shape[0]
        state.measured_training_time += dt
        state.smooth_loss = 0.98 * state.smooth_loss + 0.02 * loss.item() if state.step > 1 else loss.item()
        samples_per_sec = args.batch_size / max(dt, 1e-6)

        print(
            f"\rstep {state.step:05d} | loss {state.smooth_loss:.4f} | "
            f"pos {pos_loss.item():.4f} | atom {atom_loss.item():.4f} | "
            f"samples/sec {samples_per_sec:.1f}",
            end="",
            flush=True,
        )

        if state.measured_training_time >= args.time_budget:
            break

    print()
    val_loss, val_pos_loss, val_atom_loss = evaluate(model, val_loader, device, autocast)
    peak_vram_mb = peak_memory_mb(device)
    total_seconds = time.time() - t_start

    print("---")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"val_pos_loss:     {val_pos_loss:.6f}")
    print(f"val_atom_loss:    {val_atom_loss:.6f}")
    print(f"training_seconds: {state.measured_training_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"samples_per_sec:  {state.total_samples / max(state.measured_training_time, 1e-6):.1f}")
    print(f"num_steps:        {state.step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"max_nodes:        {max_nodes}")
    print(f"remove_h:         {args.remove_h}")
    print(f"device:           {device.type}")


def select_device(device_arg: str, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_arg == "mps":
        if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0


if __name__ == "__main__":
    main()
