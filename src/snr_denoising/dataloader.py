import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List

from numpy.fft import rfft, irfft, rfftfreq

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
                 whiten_mode: str = "auto",
                 sigma_mode: str = "std", sigma_fixed: float = 1.0,
                 allow_no_signal: bool = False):
        """
        whiten_mode:
          - 'train' : basic in-file estimator PSD (FFT power + 9-tap smoothing)
          - 'model' : use saved per-sample model PSD (datasets 'psd_model' or legacy 'psd')
          - 'welch' : use saved per-sample Welch PSD (datasets 'psd_welch' + 'psd_welch_freqs')
          - 'auto'  : prefer welch -> model -> train
        """
        self.h5_path = resolve_h5_path(h5_path)
        self.h5 = None
        self._signal = None
        self._noisy = None
        self._psd_model = None          # vlen rfft grid
        self._psd_welch = None          # vlen arbitrary grid
        self._psd_welch_freqs = None
        self.whiten = bool(whiten)
        self.whiten_mode = str(whiten_mode).lower()
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
                self.h5 = h5py.File(self.h5_path, "r")
            if "noisy" not in self.h5:
                raise KeyError("HDF5 must have 'noisy' dataset.")
            self._noisy = self.h5["noisy"]

            if "signal" in self.h5:
                self._signal = self.h5["signal"]
            elif not self.allow_no_signal:
                raise KeyError("Missing 'signal' dataset. Set allow_no_signal=True for inference.")

            # optional PSDs
            self._psd_model = self.h5.get("psd_model", self.h5.get("psd", None))
            self._psd_welch = self.h5.get("psd_welch", None)
            self._psd_welch_freqs = self.h5.get("psd_welch_freqs", None)

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

    # whitening helpers
    @staticmethod
    def _whiten_train_like(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = len(y)
        y64 = y.astype(np.float64) - np.mean(y, dtype=np.float64)
        Y = rfft(y64)
        P = np.abs(Y) ** 2
        if P.size > 9:
            kernel = np.ones(9, dtype=np.float64) / 9.0
            P = np.convolve(P, kernel, mode="same")
        P = np.maximum(P, 1e-20)
        y_w = irfft(Y / np.sqrt(P), n=L)

        X = rfft((x.astype(np.float64) - np.mean(x, dtype=np.float64)))
        x_w = irfft(X / np.sqrt(P), n=L)
        return y_w.astype(np.float32), x_w.astype(np.float32)

    @staticmethod
    def _interp_psd_to_length(P: np.ndarray, L_src: int, L_tgt: int, fs: float) -> np.ndarray:
        """If model PSD length != rfft length for L_tgt, interpolate over frequency."""
        if L_src == (L_tgt // 2 + 1):
            return P.astype(np.float64)
        f_src = rfftfreq(L_src * 2 - 2, 1.0 / fs)  # invert rfft length back to time length
        f_tgt = rfftfreq(L_tgt, 1.0 / fs)
        return np.interp(f_tgt, f_src, P, left=P[0], right=P[-1]).astype(np.float64)

    def _whiten_with_model_psd(self, y: np.ndarray, x: np.ndarray, P_model: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = len(y)
        # Interp if needed
        P = self._interp_psd_to_length(np.asarray(P_model, dtype=np.float64), len(P_model), L, self.fs)
        Y = rfft(y.astype(np.float64))
        X = rfft(x.astype(np.float64))
        y_w = irfft(Y / np.sqrt(P + 1e-20), n=L)
        x_w = irfft(X / np.sqrt(P + 1e-20), n=L)
        return y_w.astype(np.float32), x_w.astype(np.float32)

    def _whiten_with_welch(self, y: np.ndarray, x: np.ndarray, f_w: np.ndarray, P_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = len(y)
        f_tgt = rfftfreq(L, 1.0 / self.fs)
        P = np.interp(f_tgt, np.asarray(f_w, dtype=np.float64), np.asarray(P_w, dtype=np.float64),
                      left=P_w[0], right=P_w[-1])
        Y = rfft(y.astype(np.float64))
        X = rfft(x.astype(np.float64))
        y_w = irfft(Y / np.sqrt(P + 1e-20), n=L)
        x_w = irfft(X / np.sqrt(P + 1e-20), n=L)
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
            mode = self.whiten_mode
            used = None
            if mode == "auto":
                if (self._psd_welch is not None) and (self._psd_welch_freqs is not None):
                    fw = np.array(self._psd_welch_freqs[idx], dtype=np.float64)
                    Pw = np.array(self._psd_welch[idx], dtype=np.float64)
                    noisy_np, clean_np = self._whiten_with_welch(noisy_np, clean_np, fw, Pw); used = "welch"
                elif self._psd_model is not None:
                    Pm = np.array(self._psd_model[idx], dtype=np.float64)
                    noisy_np, clean_np = self._whiten_with_model_psd(noisy_np, clean_np, Pm); used = "model"
                else:
                    noisy_np, clean_np = self._whiten_train_like(noisy_np, clean_np); used = "train"
            elif mode == "welch" and (self._psd_welch is not None) and (self._psd_welch_freqs is not None):
                fw = np.array(self._psd_welch_freqs[idx], dtype=np.float64)
                Pw = np.array(self._psd_welch[idx], dtype=np.float64)
                noisy_np, clean_np = self._whiten_with_welch(noisy_np, clean_np, fw, Pw); used = "welch"
            elif mode == "model" and (self._psd_model is not None):
                Pm = np.array(self._psd_model[idx], dtype=np.float64)
                noisy_np, clean_np = self._whiten_with_model_psd(noisy_np, clean_np, Pm); used = "model"
            else:
                noisy_np, clean_np = self._whiten_train_like(noisy_np, clean_np); used = "train"

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
        self._psd_model = None
        self._psd_welch = None
        self._psd_welch_freqs = None

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
    whiten_mode: str = "auto",
    sigma_mode: str = "std",
    sigma_fixed: float = 1.0,
    allow_no_signal: bool = False,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    ds = NoisyWaveDataset(
        h5_path=h5_path,
        whiten=whiten,
        whiten_mode=whiten_mode,
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
