import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List

DEFAULT_DATA_DIR = (
    "/mnt/c/Users/charl/PycharmProjects/"
    "Diffusion_Models_for_Gravitational_Waveform_Reconstruction/data/latest_data"
)

"""
Dataloader utilities for LIGO GW diffusion training with padding.

Key points:
- Read variable-length HDF5 (no padding in the file).
- Per-sample normalisation: sigma = std(noisy).
- Batch-time padding in collate_fn:
    * Left-pad with zeros so the final index is the merger (t=0).
    * Build a 1/0 mask with ones over valid (unpadded) samples.

Exports:
- resolve_h5_path(path)
- NoisyWaveDataset
- pad_collate
- make_dataloader(...)
"""

def resolve_h5_path(path: str) -> str:
    """
    If `path` is a file, return it.
    If `path` is a directory, return the newest .h5/.hdf5 file in it.
    """
    if os.path.isdir(path):
        cands = [f for f in os.listdir(path) if f.lower().endswith((".h5", ".hdf5"))]
        if not cands:
            raise FileNotFoundError(f"No .h5/.hdf5 files found in directory: {path}")
        cands_full = [os.path.join(path, f) for f in cands]
        cands_full.sort(key=os.path.getmtime, reverse=True)
        return cands_full[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 path not found: {path}")
    return path


class NoisyWaveDataset(Dataset):
    """
    Dataset object for training on 'noisy' (signal + PSD noise).

    Expects an HDF5 with vlen datasets written by the new gen.py:
      - required: 'noisy'
      - optional: none (mask is created in the collate step)

    Returns (per sample):
      noisy_norm: torch.FloatTensor [1, L]
      sigma     : torch.FloatTensor []   (scalar)
      length    : int                    (for collate to build masks/padding)
    """
    def __init__(self, h5_path: str):
        self.h5_path = resolve_h5_path(h5_path)
        self.h5 = None
        self._noisy = None

    def _ensure_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._noisy = self.h5["noisy"]

    def __len__(self) -> int:
        with h5py.File(self.h5_path, "r") as f:
            return f["noisy"].shape[0]

    def __getitem__(self, idx: int):
        self._ensure_open()
        noisy_np = self._noisy[idx]
        # shape -> [1, L]
        noisy = torch.from_numpy(noisy_np).float().unsqueeze(0)

        # per-sample sigma from noisy waveform (avoid zeros)
        s = noisy.std()
        sigma = s if s > 0 else torch.tensor(1.0, dtype=torch.float32)

        noisy_norm = noisy / sigma
        length = noisy.shape[-1]
        return noisy_norm, sigma, length

    def close(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass
        self.h5 = None
        self._noisy = None

    def __del__(self):
        self.close()


def pad_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Left-padding to the longest sequence in the batch so that the final index aligns
    across samples (merger at t=0).

    Input batch items:
      noisy_norm: [1, L_i]
      sigma     : []
      length    : int

    Returns:
      noisy_padded : [B, 1, Lmax]
      sigma        : [B]
      mask_padded  : [B, 1, Lmax]   (1 on valid region, 0 on left padding)
    """
    noisy_list, sigma_list, len_list = zip(*batch)
    Lmax = int(max(len_list))

    def _pad_left(x: torch.Tensor, target: int) -> torch.Tensor:
        pad = target - x.shape[-1]
        return F.pad(x, (pad, 0)) if pad > 0 else x

    # pad inputs and make masks
    noisy_pad = torch.stack([_pad_left(x, Lmax) for x in noisy_list], dim=0)   # [B,1,Lmax]
    mask_pad  = torch.stack([
        _pad_left(torch.ones_like(x, dtype=torch.float32), Lmax) for x in noisy_list
    ], dim=0)                                                                  # [B,1,Lmax]

    # stack sigmas
    sigma = torch.stack([
        s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
        for s in sigma_list
    ], dim=0)                                                                   # [B]

    return noisy_pad, sigma, mask_pad


def make_dataloader(
    h5_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    ds = NoisyWaveDataset(h5_path=h5_path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=pad_collate,
    )
    return loader
