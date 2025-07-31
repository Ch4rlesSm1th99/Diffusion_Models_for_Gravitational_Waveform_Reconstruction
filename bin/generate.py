import argparse
import json
import datetime
from pathlib import Path

import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from scipy.signal import chirp

def wave_one(t, f_base):
    '''
    Smooth wave which has one central hump then fades.
    --> Gaussian peak centered on the midpoint of 't'. Env determined by duration, center and width.
    --> Harmonic variance added with the 'carrier' comp.
    --> Amplitude variance added with 'am'.
    '''
    duration = t[-1] - t[0]
    center = 0.5 * (t[0] + t[-1])
    width = 0.1 * duration
    env = np.exp(-0.5 * ((t - center) / width) ** 2)
    carrier = np.sin(f_base * t) + 0.2 * np.sin(2 * f_base * t)
    am = 1.0 + 0.1 * np.sin(0.2 * t)
    return (env * carrier * am).astype(np.float32)

def wave_two(t, f_base):
    '''
    Smooth multipeak wave which has two humps then fades.
    --> Gaussian peaks at 1/3 and 2/3 of the time range.
    --> Harmonic variance added with the 'carrier' comp.
    --> Amplitude variance added with 'am'.
    --> Chirp comp adds an increasing pitch freq with timestep.
    '''
    duration = t[-1] - t[0]
    centers = [0.33 * (t[0] + t[-1]), 0.67 * (t[0] + t[-1])]
    amps = [1.0, 0.8]
    width = 0.1 * duration
    env = np.zeros_like(t, dtype=np.float32)
    for c, a in zip(centers, amps): env += a * np.exp(-0.5 * ((t - c) / width) ** 2)
    baseline = np.sin(f_base * t)
    chirp_comp = 0.3 * chirp(t, f0=0.5 * f_base, f1=1.5 * f_base, t1=t[-1], method='linear')
    return (env * (baseline + chirp_comp)).astype(np.float32)

def wave_three(t, f_base, delta):
    '''
    Beat signal from double sin components.
    --> Two similar frequences generate beat pattern.
    --> Introduce 3 peaks for this one, more chaotic than the other two on the whole.
    --> spark introduces much higher frequency component than the base sin waves for irregular features.
    '''
    beat = np.sin(f_base * t) + 0.5 * np.sin((f_base + delta) * t)
    duration = t[-1] - t[0]
    centers = [0.25 * (t[0] + t[-1]), 0.5 * (t[0] + t[-1]), 0.75 * (t[0] + t[-1])]
    amps = [1.0, 0.8, 0.6]
    width = 0.08 * duration
    gate = np.zeros_like(t, dtype=np.float32)
    for c, a in zip(centers, amps): gate += a * np.exp(-0.5 * ((t - c) / width) ** 2)
    spark = 0.05 * np.sin(20.0 * t)
    return ((beat * gate) + spark).astype(np.float32)

def main():
    p = argparse.ArgumentParser(
        description="Generate dataset with 3 waveform classes + optional SNR binning."
    )
    p.add_argument("--snr_low", type=float, default=10.0,
                   help="Lower bound on SNR")
    p.add_argument("--snr_high", type=float, default=10.0,
                   help="Upper bound on power‐SNR")
    p.add_argument("--snr_bins", type=str, default=None,
                   help="Comma‑sep list of SNR bins as LOW-HIGH, eg. '50-100,30-50... and so on'. "
                        "Evenly splits samples across bins if set.")
    p.add_argument("--points", type=int, default=500,
                   help="Number of time steps")
    p.add_argument("--samples", type=int, default=2000,
                   help="Total number of waveforms to generate, size of dataset")
    p.add_argument("--data", type=str,
                   default=r"C:\Users\charl\PycharmProjects\snr_denoising\bin\data",
                   help="Base output folder for dataset (default: latest project data)")        # point to your directory where you want to save set
    p.add_argument("--plot", action="store_true",
                   help="Plot sample waveforms after generation")
    args = p.parse_args()

    base = Path(args.data)         # saving path
    out_dir = base / "latest_dataset"
    if out_dir.exists():
        for f in out_dir.glob('*'): f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # init empty arrays for signals + labels
    t = np.linspace(0, args.duration if hasattr(args, 'duration') else 2*np.pi, args.points, dtype=np.float32)
    clean = np.zeros((args.samples, args.points), dtype=np.float32)
    labels = np.zeros(args.samples, dtype=np.int32)
    for i in range(args.samples):
        cls = np.random.choice([0,1,2]); labels[i] = cls        # randomly each sample in set to one of the classes
        if cls == 0:
            f_base = np.random.uniform(4.0, 6.0)
            clean[i] = wave_one(t, f_base)
        elif cls == 1:
            f_base = np.random.uniform(2.0, 4.0)
            clean[i] = wave_two(t, f_base)
        else:
            f_base = np.random.uniform(4.0, 6.0)
            delta  = np.random.uniform(0.2, 0.8)
            clean[i] = wave_three(t, f_base, delta)                 # gen clean singals

    # normalize waveform to unit peak amplitude of 1 --> ensure SNR scaling is proportional
    peaks = np.max(np.abs(clean), axis=1, keepdims=True) + 1e-12
    clean = (clean.T / peaks.T).T

    # sample SNR values for each waveform ie "a target range for each bin"
    if args.snr_bins:      # either defined bins or uniformly spaced from low SNR to high SNr
        bins = []
        for part in args.snr_bins.split(','):
            low, high = part.split('-'); bins.append((float(low), float(high)))
        n_bins = len(bins)
        per_bin = args.samples // n_bins
        extra = args.samples - per_bin * n_bins
        snr_list = [np.random.uniform(low, high, size=per_bin) for low, high in bins]
        if extra:
            low0, high0 = bins[0]
            snr_list.append(np.random.uniform(low0, high0, size=extra))
        snr_vals = np.concatenate(snr_list)
        np.random.shuffle(snr_vals)
    else:
        snr_vals = np.random.uniform(args.snr_low, args.snr_high, size=args.samples)

    # add the nosie for the SNR
    noise = np.random.randn(args.samples, args.points).astype(np.float32)
    p_signal = np.mean(clean**2, axis=1)
    sigma = np.sqrt(p_signal / snr_vals)
    noisy = clean + (noise.T * sigma).T

    # train/test split
    idxs = np.arange(args.samples)
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42, stratify=labels)

    # save in h5
    h5file = out_dir / "dataset.h5"
    with h5py.File(h5file, "w") as f:
        f.attrs['created']   = datetime.datetime.now().isoformat()
        f.attrs['snr_low']   = args.snr_low
        f.attrs['snr_high']  = args.snr_high
        f.attrs['classes']   = json.dumps({0: 'single_peak',
                                          1: 'dual_chirp_peaks',
                                          2: 'chaos_energy_peaks'})
        grp = f.create_group('time')
        grp.create_dataset('t', data=t)
        grp.create_dataset('clean_train', data=clean[tr])
        grp.create_dataset('noisy_train', data=noisy[tr])
        grp.create_dataset('clean_test', data=clean[te])
        grp.create_dataset('noisy_test', data=noisy[te])
        lbl = f.create_group('labels')
        lbl.create_dataset('train', data=labels[tr])
        lbl.create_dataset('test', data=labels[te])
    print(f"saved dataset to {h5file}")

    if args.plot:               # plot
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(6,1,figsize=(10,12), sharex=True)
        for cls in [0,1,2]:
            class_idxs = [i for i in tr if labels[i]==cls]
            sel = np.random.choice(class_idxs, 3, replace=False)
            ax_clean = axes[2*cls]; ax_noisy = axes[2*cls+1]
            for idx in sel:
                ax_clean.plot(t, clean[idx]); ax_noisy.plot(t, noisy[idx])
            ax_clean.set_title(f"Class {cls}"); ax_clean.set_ylabel("Clean")
            ax_noisy.set_ylabel("Noisy")
        axes[-1].set_xlabel("Time")
        plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
