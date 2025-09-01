import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List

from numpy.fft import rfft, irfft

def _mad_std(x: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.4826 * np.median(np.abs(x64 - np.median(x64))) + 1e-24

def resolve_h5_path(path: str) -> str:
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
    def __init__(self, h5_path: str, whiten: bool = False,
                 sigma_mode: str = "std", sigma_fixed: float = 1.0,
                 allow_no_signal: bool = False):
        self.h5_path = resolve_h5_path(h5_path)
        self.h5 = None
        self._signal = None
        self._noisy = None
        self.whiten = whiten
        self.sigma_mode = sigma_mode
        self.sigma_fixed = float(sigma_fixed)
        self.fs = None
        self.N = None
        self.allow_no_signal = allow_no_signal

    def _ensure_open(self):
        if self.h5 is None:
            try:
                self.h5 = h5py.File(self.h5_path, "r", swmr=True)
            except Exception:
                # graceful fallback if SWMR not supported
                self.h5 = h5py.File(self.h5_path, "r")
            if "noisy" not in self.h5:
                raise KeyError("HDF5 must have 'noisy' dataset.")
            self._noisy = self.h5["noisy"]

            if "signal" in self.h5:
                self._signal = self.h5["signal"]
            elif not self.allow_no_signal:
                raise KeyError("Missing 'signal' dataset. Set allow_no_signal=True for inference.")

            # length sanity
            self.N = self._noisy.shape[0]
            if self._signal is not None and self._signal.shape[0] != self.N:
                raise ValueError("Mismatched leading dimension between 'signal' and 'noisy'.")

            # sampling rate
            attrs = self.h5.attrs
            fs_attr = attrs.get("sampling_rate", 0.0)
            dt_attr = attrs.get("delta_t", 1.0 / 4096.0)
            self.fs = float(fs_attr) if float(fs_attr) > 0 else float(1.0 / float(dt_attr))

    def __len__(self) -> int:
        if self.N is None:
            self._ensure_open()
        return int(self.N)

    def _whiten_pair(self, y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = len(y)
        y64 = y.astype(np.float64)
        y64 = y64 - np.mean(y64)
        Y = rfft(y64)
        P = np.abs(Y) ** 2

        # cheap smoothing in frequency with small moving average
        if P.size > 9:
            kernel = np.ones(9, dtype=np.float64) / 9.0
            P = np.convolve(P, kernel, mode="same")
        P = np.maximum(P, 1e-20)  # slightly looser floor to avoid huge gains

        y_w = irfft(Y / np.sqrt(P), n=L)

        X = rfft((x.astype(np.float64) - np.mean(x, dtype=np.float64)))
        x_w = irfft(X / np.sqrt(P), n=L)
        return y_w.astype(np.float32), x_w.astype(np.float32)

    def __getitem__(self, idx: int):
        self._ensure_open()
        noisy_np = np.array(self._noisy[idx], dtype=np.float32)
        if self._signal is not None:
            clean_np = np.array(self._signal[idx], dtype=np.float32)
        else:
            clean_np = np.zeros_like(noisy_np, dtype=np.float32)  # placeholder for inference

        # NaN/Inf guard
        if not np.isfinite(noisy_np).all():
            noisy_np = np.nan_to_num(noisy_np, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(clean_np).all():
            clean_np = np.nan_to_num(clean_np, nan=0.0, posinf=0.0, neginf=0.0)

        if self.whiten:
            noisy_np, clean_np = self._whiten_pair(noisy_np, clean_np)

        # per-sample sigma
        if self.sigma_mode == "std":
            s = float(np.std(noisy_np.astype(np.float64)))
        elif self.sigma_mode == "mad":
            s = float(_mad_std(noisy_np))
        elif self.sigma_mode == "fixed":
            s = float(self.sigma_fixed)
        else:
            raise ValueError(f"Unknown sigma_mode: {self.sigma_mode}")
        if not np.isfinite(s) or s <= 0:
            s = 1.0

        clean = torch.from_numpy(clean_np).float().unsqueeze(0)
        noisy = torch.from_numpy(noisy_np).float().unsqueeze(0)
        sigma = torch.tensor(s, dtype=torch.float32)
        length = noisy.shape[-1]
        return clean, noisy, sigma, length


    def close(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass
        self.h5 = None
        self._signal = None
        self._noisy = None

    def __del__(self):
        self.close()


def pad_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_list, noisy_list, sigma_list, len_list = zip(*batch)
    Lmax = int(max(len_list))

    def _pad_left(x: torch.Tensor, target: int) -> torch.Tensor:
        pad = target - x.shape[-1]
        return F.pad(x, (pad, 0)) if pad > 0 else x

    clean_pad = torch.stack([_pad_left(x, Lmax) for x in clean_list], dim=0)
    noisy_pad = torch.stack([_pad_left(x, Lmax) for x in noisy_list], dim=0)
    mask_pad  = torch.stack([
        _pad_left(torch.ones_like(x, dtype=torch.float32), Lmax) for x in noisy_list
    ], dim=0)

    sigma = torch.stack([
        s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
        for s in sigma_list
    ], dim=0)

    return clean_pad, noisy_pad, sigma, mask_pad


def make_dataloader(
    h5_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = None,
    whiten: bool = False,
    sigma_mode: str = "std",
    sigma_fixed: float = 1.0,
    allow_no_signal: bool = False,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    ds = NoisyWaveDataset(
        h5_path=h5_path,
        whiten=whiten,
        sigma_mode=sigma_mode,
        sigma_fixed=sigma_fixed,
        allow_no_signal=allow_no_signal,
    )

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
