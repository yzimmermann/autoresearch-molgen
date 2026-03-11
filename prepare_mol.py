"""
One-time data preparation for QM9 molecule-generation experiments.

Downloads the official QM9 XYZ archive and converts it into dense padded tensors:

- atom_types: [num_mols, max_nodes] int16, pad value -1
- positions:  [num_mols, max_nodes, 3] float32
- mask:       [num_mols, max_nodes] bool
- num_atoms:  [num_mols] int16

Usage:
    python3 prepare_mol.py
    python3 prepare_mol.py --remove-h
    python3 prepare_mol.py --max-molecules 2048
"""

from __future__ import annotations

import argparse
import os
import re
import tarfile
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import requests
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 300
PAD_ATOM_TYPE = -1
DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE = 100_000
DEFAULT_VAL_SIZE = 10_000

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-mol")
QM9_DIR = os.path.join(CACHE_DIR, "qm9")
RAW_DIR = os.path.join(QM9_DIR, "raw")
PROCESSED_DIR = os.path.join(QM9_DIR, "processed")

QM9_ARCHIVE_NAME = "dsgdb9nsd.xyz.tar.bz2"
QM9_ARCHIVE_URL = "https://ndownloader.figshare.com/files/3195389"

ATOM_DECODER_WITH_H = ["H", "C", "N", "O", "F"]
ATOM_DECODER_NO_H = ["C", "N", "O", "F"]


@dataclass(frozen=True)
class SplitConfig:
    train_size: int = DEFAULT_TRAIN_SIZE
    val_size: int = DEFAULT_VAL_SIZE
    seed: int = DEFAULT_SEED


class MoleculeSplit(Dataset):
    def __init__(self, split_tensors: Dict[str, torch.Tensor]):
        self.split_tensors = split_tensors

    def __len__(self) -> int:
        return int(self.split_tensors["num_atoms"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "atom_types": self.split_tensors["atom_types"][idx],
            "positions": self.split_tensors["positions"][idx],
            "mask": self.split_tensors["mask"][idx],
            "num_atoms": self.split_tensors["num_atoms"][idx],
        }


def get_atom_decoder(remove_h: bool) -> List[str]:
    return ATOM_DECODER_NO_H if remove_h else ATOM_DECODER_WITH_H


def get_atom_encoder(remove_h: bool) -> Dict[str, int]:
    return {atom: idx for idx, atom in enumerate(get_atom_decoder(remove_h))}


def get_archive_path() -> str:
    return os.path.join(RAW_DIR, QM9_ARCHIVE_NAME)


def get_processed_path(remove_h: bool) -> str:
    name = "qm9_noh.pt" if remove_h else "qm9_with_h.pt"
    return os.path.join(PROCESSED_DIR, name)


def download_qm9() -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    archive_path = get_archive_path()
    if os.path.exists(archive_path):
        print(f"QM9: archive already present at {archive_path}")
        return archive_path

    print(f"QM9: downloading archive to {archive_path}")
    response = requests.get(QM9_ARCHIVE_URL, stream=True, timeout=60)
    response.raise_for_status()

    temp_path = archive_path + ".tmp"
    with open(temp_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    os.replace(temp_path, archive_path)
    return archive_path


def _member_sort_key(member_name: str) -> int:
    match = re.search(r"(\d+)", os.path.basename(member_name))
    return int(match.group(1)) if match else -1


def iter_xyz_members(archive_path: str) -> Iterable[tuple[str, str]]:
    with tarfile.open(archive_path, mode="r:bz2") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".xyz")]
        members.sort(key=lambda member: _member_sort_key(member.name))
        for member in members:
            handle = tar.extractfile(member)
            if handle is None:
                continue
            yield member.name, handle.read().decode("utf-8")


def parse_xyz_record(text: str, remove_h: bool) -> Dict[str, torch.Tensor] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return None

    num_atoms = int(lines[0])
    atom_lines = lines[2:2 + num_atoms]
    if len(atom_lines) != num_atoms:
        return None

    encoder = get_atom_encoder(remove_h)
    atom_types: List[int] = []
    positions: List[List[float]] = []
    for line in atom_lines:
        cols = line.split()
        if len(cols) < 4:
            return None
        symbol = cols[0]
        if remove_h and symbol == "H":
            continue
        if symbol not in encoder:
            return None
        atom_types.append(encoder[symbol])
        positions.append([parse_float(cols[1]), parse_float(cols[2]), parse_float(cols[3])])

    if not atom_types:
        return None

    pos = torch.tensor(positions, dtype=torch.float32)
    pos = pos - pos.mean(dim=0, keepdim=True)
    atom_tensor = torch.tensor(atom_types, dtype=torch.int16)

    return {
        "atom_types": atom_tensor,
        "positions": pos,
        "num_atoms": torch.tensor(len(atom_types), dtype=torch.int16),
    }


def parse_float(value: str) -> float:
    # QM9 XYZ files may use Fortran-style exponents such as 2.1997*^-6.
    return float(value.replace("*^", "e"))


def parse_archive(archive_path: str, remove_h: bool, max_molecules: int | None = None) -> List[Dict[str, torch.Tensor]]:
    records: List[Dict[str, torch.Tensor]] = []
    t0 = time.time()
    for idx, (_, text) in enumerate(iter_xyz_members(archive_path), start=1):
        record = parse_xyz_record(text, remove_h=remove_h)
        if record is not None:
            records.append(record)
        if idx % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"QM9: parsed {idx:,} files in {elapsed:.1f}s")
        if max_molecules is not None and len(records) >= max_molecules:
            break

    if not records:
        raise RuntimeError("QM9: no records parsed from archive")
    return records


def build_dense_tensors(records: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_nodes = max(int(record["num_atoms"]) for record in records)
    num_mols = len(records)

    atom_types = torch.full((num_mols, max_nodes), PAD_ATOM_TYPE, dtype=torch.int16)
    positions = torch.zeros((num_mols, max_nodes, 3), dtype=torch.float32)
    mask = torch.zeros((num_mols, max_nodes), dtype=torch.bool)
    num_atoms = torch.empty((num_mols,), dtype=torch.int16)

    for i, record in enumerate(records):
        n = int(record["num_atoms"])
        atom_types[i, :n] = record["atom_types"]
        positions[i, :n] = record["positions"]
        mask[i, :n] = True
        num_atoms[i] = n

    return {
        "atom_types": atom_types,
        "positions": positions,
        "mask": mask,
        "num_atoms": num_atoms,
    }


def split_indices(num_molecules: int, cfg: SplitConfig) -> Dict[str, torch.Tensor]:
    if cfg.train_size + cfg.val_size >= num_molecules:
        raise ValueError(
            f"Split sizes too large for dataset: train={cfg.train_size} val={cfg.val_size} total={num_molecules}"
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(num_molecules, generator=generator)
    train_end = cfg.train_size
    val_end = train_end + cfg.val_size
    return {
        "train": perm[:train_end],
        "val": perm[train_end:val_end],
        "test": perm[val_end:],
    }


def index_split(dense_tensors: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in dense_tensors.items()}


def summarize_split(name: str, split_tensors: Dict[str, torch.Tensor]) -> None:
    num_mols = int(split_tensors["num_atoms"].shape[0])
    max_nodes = int(split_tensors["mask"].shape[1])
    avg_nodes = float(split_tensors["num_atoms"].float().mean())
    print(f"{name:>5s}: {num_mols:>7,} molecules | avg_nodes={avg_nodes:>5.2f} | max_nodes={max_nodes}")


def prepare_qm9(remove_h: bool, split_cfg: SplitConfig, max_molecules: int | None = None) -> str:
    archive_path = download_qm9()
    records = parse_archive(archive_path, remove_h=remove_h, max_molecules=max_molecules)
    dense_tensors = build_dense_tensors(records)
    indices = split_indices(len(records), split_cfg)

    processed = {
        "meta": {
            "dataset": "qm9",
            "remove_h": remove_h,
            "atom_decoder": get_atom_decoder(remove_h),
            "atom_encoder": get_atom_encoder(remove_h),
            "pad_atom_type": PAD_ATOM_TYPE,
            "split_seed": split_cfg.seed,
            "max_nodes": int(dense_tensors["mask"].shape[1]),
            "train_size": int(indices["train"].numel()),
            "val_size": int(indices["val"].numel()),
            "test_size": int(indices["test"].numel()),
        },
        "splits": {
            split_name: index_split(dense_tensors, split_indices_)
            for split_name, split_indices_ in indices.items()
        },
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = get_processed_path(remove_h)
    torch.save(processed, out_path)

    print(f"QM9: saved processed dataset to {out_path}")
    for split_name, tensors in processed["splits"].items():
        summarize_split(split_name, tensors)
    return out_path


def load_dataset(remove_h: bool = False) -> Dict[str, object]:
    path = get_processed_path(remove_h)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed QM9 dataset not found at {path}. Run prepare_mol.py first.")
    return torch.load(path, map_location="cpu", weights_only=True)


def make_dataloader(split: str, batch_size: int, remove_h: bool = False, shuffle: bool | None = None) -> DataLoader:
    assert split in {"train", "val", "test"}
    dataset = load_dataset(remove_h=remove_h)
    split_tensors = dataset["splits"][split]
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(MoleculeSplit(split_tensors), batch_size=batch_size, shuffle=shuffle)


def print_dataset_summary(remove_h: bool) -> None:
    dataset = load_dataset(remove_h=remove_h)
    meta = dataset["meta"]
    print("QM9 summary:")
    for key in ["dataset", "remove_h", "max_nodes", "train_size", "val_size", "test_size"]:
        print(f"  {key:12s}: {meta[key]}")
    for split_name, split_tensors in dataset["splits"].items():
        summarize_split(split_name, split_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare QM9 tensors for molecule-generation experiments")
    parser.add_argument("--remove-h", action="store_true", help="Drop hydrogens for the tiny dev benchmark")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for split generation")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE, help="Training split size")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE, help="Validation split size")
    parser.add_argument("--max-molecules", type=int, default=None, help="Optional cap for quick local tests")
    parser.add_argument("--summary-only", action="store_true", help="Print the existing processed dataset summary")
    args = parser.parse_args()

    if args.summary_only:
        print_dataset_summary(remove_h=args.remove_h)
    else:
        prepare_qm9(
            remove_h=args.remove_h,
            split_cfg=SplitConfig(train_size=args.train_size, val_size=args.val_size, seed=args.seed),
            max_molecules=args.max_molecules,
        )
