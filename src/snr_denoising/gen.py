import matplotlib.pyplot as plt
import numpy as np
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.detector import Detector
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import sigma
import os
import h5py
import json

"""
gen.py — Generate time-domain GW waveforms with LIGO-like noise, and write HDF5 datasets with separate properties.

Modes:
  - fixed : repeat one wave config for the entire dataset
  - random: randomly sample wave configs within ranges
  - grid  : balanced coverage of (m1, m2) pairs in a grid (unordered: m2 <= m1)

Output:
  HDF5 containing variable-length signal/noise/noisy, per-sample times (seconds-relative with t=0 at merger),
  metadata and gen mode.

This version has **no fallbacks**. If a combo fails (e.g., SEOBNRv4 at given f_lower), it is skipped — or,
with --require-complete-grid, the run aborts and tells you to adjust --f-lower (or ranges).
"""

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# PSD cache to avoid recomputing PSD interpolation on every call
_PSD_CACHE = {}


def generate_ligo_waveform(
    mass1,
    mass2,
    target_snr,
    spin1z=0,
    spin2z=0,
    distance=410.0,
    f_lower=20.0,
    sampling_rate=4096,
    detector="H1",
    ra=0.0,
    dec=0.0,
    polarization=0.0,
    random_seed=42,
    plot=False
):
    """
    Gravitational waveform generator with LIGO noise injections using PyCBC PSD.
    NOTE: PyCBC's wheel breaks on MSVC -> Use Linux (e.g., WSL).
    """
    delta_t = 1.0 / sampling_rate

    hp, hc = get_td_waveform(
        approximant="SEOBNRv4",
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=distance,
        delta_t=delta_t,
        f_lower=f_lower
    )

    # Project onto chosen detector
    det = Detector(detector)
    signal = det.project_wave(hp, hc, ra, dec, polarization)
    waveform_epoch = signal._epoch

    # PSD (cache keyed by detector, sr, length, f_lower)
    psd_key = f"{detector}_{sampling_rate}_{len(signal)}_{f_lower}"
    if psd_key in _PSD_CACHE:
        psd = _PSD_CACHE[psd_key]
    else:
        df = 1.0 / (len(signal) * delta_t)
        psd = aLIGOZeroDetHighPower(len(signal)//2 + 1, df, f_lower)
        _PSD_CACHE[psd_key] = psd

    # Scale to target SNR (noise-weighted)
    current_snr = sigma(signal, psd=psd, low_frequency_cutoff=f_lower)
    scaled_signal = signal * (target_snr / current_snr)

    # Generate coloured Gaussian noise with given PSD, align epoch
    noise = noise_from_psd(len(signal), delta_t, psd, seed=random_seed)
    noise._epoch = waveform_epoch
    noisy_signal = scaled_signal + noise

    # Time array for plotting (absolute)
    start_time = float(waveform_epoch)
    times = np.arange(len(signal)) * delta_t + start_time

    if plot:
        signal_array = scaled_signal.numpy()
        noise_array = noise.numpy()
        noisy_array = noisy_signal.numpy()

        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(times, signal_array)
        plt.title(f'LIGO {detector} Signal (m1={mass1}M☉, m2={mass2}M☉)')
        plt.ylabel('Strain'); plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(times, noise_array)
        plt.title('Detector Noise')
        plt.ylabel('Strain'); plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(times, noisy_array)
        plt.title(f'Noisy Signal (SNR={target_snr:.1f})')
        plt.xlabel('Time (s)'); plt.ylabel('Strain'); plt.grid(True)
        plt.tight_layout(); plt.show()

    return {
        'signal': scaled_signal,
        'noise': noise,
        'noisy_signal': noisy_signal,
        'times': times,
        'snr': target_snr,
        'detector': detector,
        'epoch': waveform_epoch
    }


def _maybe_tqdm(iterable, total=None, desc=None, use_tqdm=True):
    if use_tqdm and _HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def collect_samples(
    sample_specs,
    plot_flag=False,
    progress_every=0,
    save_psd=False,
    psd_preview=0,
    psd_preview_dir=None,
    use_tqdm=True
):
    """
    Generate samples from a list of specs (kwargs for generate_ligo_waveform).
    Enforces m1 >= m2 for generation safety (labels may be unsorted if in args).

    Each spec must contain at least: mass1, mass2, target_snr, spin1z, spin2z
    """
    os.makedirs(psd_preview_dir, exist_ok=True) if (psd_preview and psd_preview_dir) else None

    sig_list, noise_list, noisy_list, t_list = [], [], [], []

    meta = {k: [] for k in [
        'mass1','mass2','snr','spin1z','spin2z',
        'label_m1','label_m2','label_s1','label_s2',
        'q','chirp_mass',
        'epoch',
        # PSD metadata
        'psd_len','psd_df','psd_f_lower'
    ]}
    detectors = []
    full_psd_list = [] if save_psd else None

    iterable = _maybe_tqdm(range(len(sample_specs)), total=len(sample_specs),
                           desc="Generating samples", use_tqdm=use_tqdm)

    for i in iterable:
        spec = sample_specs[i]
        # labels
        Lm1 = spec.get('label_m1', spec['mass1'])
        Lm2 = spec.get('label_m2', spec['mass2'])
        Ls1 = spec.get('label_s1', spec['spin1z'])
        Ls2 = spec.get('label_s2', spec['spin2z'])

        # sort for generator call
        m1, m2 = float(spec['mass1']), float(spec['mass2'])
        s1, s2 = float(spec['spin1z']), float(spec['spin2z'])
        if m1 < m2:
            m1, m2, s1, s2 = m2, m1, s2, s1

        f_lower = float(spec.get('f_lower', 20.0))
        sr = int(spec.get('sampling_rate', 4096))
        detector = spec.get('detector', 'H1')

        call_kwargs = {
            'mass1': m1, 'mass2': m2,
            'target_snr': float(spec['target_snr']),
            'spin1z': s1, 'spin2z': s2,
            'distance': float(spec.get('distance', 410.0)),
            'f_lower': f_lower,
            'sampling_rate': sr,
            'detector': detector,
            'ra': float(spec.get('ra', 0.0)),
            'dec': float(spec.get('dec', 0.0)),
            'polarization': float(spec.get('polarization', 0.0)),
        }

        try:
            plot_this = plot_flag and i < 3
            res = generate_ligo_waveform(**call_kwargs, random_seed=i, plot=plot_this)
        except Exception as e:
            if not use_tqdm and progress_every:
                print(f"generation failed idx={i} for labeled {spec}: {e}")
            continue

        sig_np = res['signal'].numpy()
        noi_np = res['noise'].numpy()
        noz_np = res['noisy_signal'].numpy()

        sig_list.append(sig_np)
        noise_list.append(noi_np)
        noisy_list.append(noz_np)
        t_list.append(res['times'])

        # PSD metadata
        N = len(sig_np)
        delta_t = 1.0 / sr
        df = 1.0 / (N * delta_t)
        meta['psd_len'].append(N // 2 + 1)
        meta['psd_df'].append(df)
        meta['psd_f_lower'].append(f_lower)
        detectors.append(detector.encode('utf-8'))

        # Optional full PSD vector
        if full_psd_list is not None:
            psd_vec = aLIGOZeroDetHighPower(N // 2 + 1, df, f_lower).numpy().astype(np.float32)
            full_psd_list.append(psd_vec)

        # Optional PSD previews
        if psd_preview and (len(sig_list) <= psd_preview) and psd_preview_dir:
            psd_vec = aLIGOZeroDetHighPower(N // 2 + 1, df, f_lower).numpy()
            f = np.arange(psd_vec.shape[0]) * df
            asd = np.sqrt(psd_vec + 1e-30)
            plt.figure(figsize=(6, 4))
            plt.loglog(f[1:], asd[1:])
            plt.xlabel("Frequency (Hz)"); plt.ylabel("ASD (1/√Hz)")
            plt.title(f"PSD (ASD) — {detector}, f_low={f_lower:g} Hz, N={N}, Δt={delta_t:.6g}")
            plt.grid(True, which='both', ls=':')
            os.makedirs(psd_preview_dir, exist_ok=True)
            plt.savefig(os.path.join(psd_preview_dir, f"psd_{len(sig_list):05d}.png"), dpi=150, bbox_inches='tight')
            plt.close()

        if (not _HAS_TQDM or not use_tqdm) and progress_every:
            if ((i + 1) % progress_every == 0) or ((i + 1) == len(sample_specs)):
                print(f"generated {i + 1}/{len(sample_specs)}")

        # metadata
        meta['epoch'].append(float(res['epoch']))
        meta['mass1'].append(m1);          meta['mass2'].append(m2)
        meta['spin1z'].append(s1);         meta['spin2z'].append(s2)
        meta['snr'].append(float(spec['target_snr']))
        meta['label_m1'].append(float(Lm1)); meta['label_m2'].append(float(Lm2))
        meta['label_s1'].append(float(Ls1)); meta['label_s2'].append(float(Ls2))

        q = m1 / m2
        M = m1 + m2
        eta = (m1 * m2) / (M * M)
        Mchirp = (eta ** (3.0/5.0)) * M
        meta['q'].append(q)
        meta['chirp_mass'].append(Mchirp)

    return sig_list, noise_list, noisy_list, t_list, meta, detectors, full_psd_list


def finalize_and_write(
    output_path,
    sig_list, noise_list, noisy_list, t_list, meta,
    sampling_rate,
    attrs_extra=None,
    detectors_bytes=None,
    full_psd_list=None
):
    """Write HDF5 with variable-length datasets (no padding), and times centered at merger event (t=0)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lengths = np.array([len(x) for x in sig_list], dtype=np.int32)
    delta_t = 1.0 / float(sampling_rate)

    def _write_common(f):
        for k, arr in meta.items():
            if arr:
                f.create_dataset(k, data=np.array(arr, dtype=np.float32))
        if detectors_bytes is not None:
            vlen_str = h5py.special_dtype(vlen=bytes)
            f.create_dataset('psd_detector', data=np.array(detectors_bytes, dtype=object), dtype=vlen_str)

        # PSD full vectors
        if full_psd_list is not None:
            vlen_f32 = h5py.special_dtype(vlen=np.float32)
            f.create_dataset('psd', (len(full_psd_list),), dtype=vlen_f32, data=full_psd_list)

        # attrs
        f.attrs['padding'] = 'none'
        f.attrs['sampling_rate'] = float(sampling_rate)
        f.attrs['delta_t'] = float(delta_t)
        f.attrs['time_axis'] = 'seconds-rel'  # t=0 at merger
        if attrs_extra:
            for k, v in attrs_extra.items():
                f.attrs[k] = v

    # variable-length storage per sample
    vlen_f32 = h5py.special_dtype(vlen=np.float32)
    vlen_f64 = h5py.special_dtype(vlen=np.float64)

    # always store seconds-relative time: t_rel = t - t[-1] (merger at 0)
    times_vlen = [t - t[-1] for t in t_list]

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('signal', (len(sig_list),), dtype=vlen_f32, data=sig_list)
        f.create_dataset('noise',  (len(noise_list),), dtype=vlen_f32, data=noise_list)
        f.create_dataset('noisy',  (len(noisy_list),), dtype=vlen_f32, data=noisy_list)
        f.create_dataset('times',  (len(times_vlen),), dtype=vlen_f64, data=times_vlen)
        f.create_dataset('lengths', data=lengths)
        _write_common(f)
    print(f"saved {len(sig_list)} samples (padding=none, time_axis=seconds-rel) --> {output_path}")


if __name__ == "__main__":
    import argparse

    class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        prog="gen.py",
        description=(
            "Generate LIGO-like time-domain GW waveforms and write an HDF5 dataset.\n\n"
            "MODES\n"
            "  fixed  : repeat a single (m1, m2, snr, spins) config N times\n"
            "  random : sample (m1, m2, snr, spins) uniformly within ranges\n"
            "  grid   : make an even grid over (m1, m2); balanced #samples per unordered pair (m2 <= m1)\n\n"
            "OUTPUT\n"
            "  HDF5 containing:\n"
            "    - signal/noise/noisy arrays (variable-length)\n"
            "    - times (seconds-relative, merger at t=0)\n"
            "    - metadata per sample (masses, spins, SNR, epoch, PSD metadata, etc.)\n"
        ),
        formatter_class=_HelpFmt,
    )

    # GENERAL
    g_general = parser.add_argument_group("General")
    g_general.add_argument('--mode', choices=['fixed', 'random', 'grid'], default='fixed',
                           help='Which generator to use.')
    g_general.add_argument('--num-samples', type=int, required=True,
                           help='How many samples to write.')
    g_general.add_argument('--output-path', type=str, required=True,
                           help='Destination HDF5 path (parent folders will be created).')
    g_general.add_argument('--seed', type=int, default=123,
                           help='Random seed for reproducibility across all modes.')

    # FIXED MODE
    g_fixed = parser.add_argument_group("Fixed mode (used only when --mode fixed)")
    g_fixed.add_argument('--mass1', type=float, default=20.0, help='Primary mass (Msun).')
    g_fixed.add_argument('--mass2', type=float, default=20.0, help='Secondary mass (Msun).')
    g_fixed.add_argument('--snr', type=float, default=8000.0, help='Target SNR for the repeated config.')

    # RANDOM/GRID RANGES
    g_ranges = parser.add_argument_group("Ranges (used by --mode random / --mode grid)")
    g_ranges.add_argument('--mass1-min', type=float, default=20.0, help='Min primary mass (Msun).')
    g_ranges.add_argument('--mass1-max', type=float, default=20.0, help='Max primary mass (Msun).')
    g_ranges.add_argument('--mass2-min', type=float, default=20.0, help='Min secondary mass (Msun).')
    g_ranges.add_argument('--mass2-max', type=float, default=20.0, help='Max secondary mass (Msun).')
    g_ranges.add_argument('--snr-min', type=float, default=8000.0, help='Min SNR draw.')
    g_ranges.add_argument('--snr-max', type=float, default=8000.0, help='Max SNR draw.')
    g_ranges.add_argument('--spin1-min', type=float, default=0.0, help='Min aligned spin of body 1.')
    g_ranges.add_argument('--spin1-max', type=float, default=0.0, help='Max aligned spin of body 1.')
    g_ranges.add_argument('--spin2-min', type=float, default=0.0, help='Min aligned spin of body 2.')
    g_ranges.add_argument('--spin2-max', type=float, default=0.0, help='Max aligned spin of body 2.')

    # GRID OPTIONS
    g_grid = parser.add_argument_group("Grid mode")
    g_grid.add_argument('--grid-steps', type=int, default=5,
                        help='Number of evenly spaced points between min/max for mass1 and mass2 (e.g. 9 gives 10,20,...,90).')
    g_grid.add_argument('--augment-symmetric', action='store_true',
                        help='Also include swapped labels in metadata (label_m1,label_m2); generation still uses sorted (m1>=m2).')
    g_grid.add_argument('--shuffle', action='store_true',
                        help='Shuffle the order of samples before writing.')
    g_grid.add_argument('--overgen-factor', type=float, default=1.05,
                        help='Over-generate by this fraction then trim back to exactly --num-samples.')
    g_grid.add_argument('--require-complete-grid', action='store_true',
                        help='If set, raise an error if any (m1,m2) pair fails during the probe step (tip: try adjusting --f-lower).')

    # removed padding args --> now variable length and padding is performed in loader

    # MISC / TUNABLES
    g_misc = parser.add_argument_group("Misc")
    g_misc.add_argument('--plot', action='store_true',
                        help='Plot the first few generated examples for a quick visual check.')
    g_misc.add_argument('--progress-every', type=int, default=0,
                        help='Print progress every N items when tqdm is unavailable or disabled.')
    g_misc.add_argument('--use-tqdm', action='store_true',
                        help='Use tqdm progress bars if installed.')
    g_misc.add_argument('--f-lower', type=float, default=20.0,
                        help='Base low-frequency cutoff passed to get_td_waveform.')
    g_misc.add_argument('--sampling-rate', type=int, default=4096,
                        help='Base sampling rate (Hz).')

    # PSD options
    g_psd = parser.add_argument_group("PSD options")
    g_psd.add_argument('--save-psd', action='store_true',
                       help='Store the full PSD vector per sample in the HDF5 (variable-length).')
    g_psd.add_argument('--psd-preview', type=int, default=0,
                       help='If > 0, save this many PSD ASD plots to disk.')
    g_psd.add_argument('--psd-preview-dir', type=str, default=None,
                       help='Directory for PSD preview images; defaults to <output_dir>/psd_plots.')

    args = parser.parse_args()

    args_json = json.dumps(vars(args), sort_keys=True)

    if args.mass2_min > args.mass1_max:
        raise ValueError("mass2_min must be <= mass1_max; otherwise no (m2 <= m1) pairs exist for the grid.")

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # default PSD preview dir next to output
    if args.psd_preview and not args.psd_preview_dir:
        base_dir = os.path.dirname(args.output_path)
        args.psd_preview_dir = os.path.join(base_dir, "psd_plots")

    if args.mode == 'fixed':
        sample_specs = [
            dict(
                mass1=args.mass1, mass2=args.mass2, target_snr=args.snr,
                spin1z=0.0, spin2z=0.0, distance=410.0,
                f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0
            )
            for _ in range(args.num_samples)
        ]
        sig_list, noise_list, noisy_list, t_list, meta, detectors, full_psd_list = collect_samples(
            sample_specs,
            plot_flag=args.plot,
            progress_every=args.progress_every,
            save_psd=args.save_psd,
            psd_preview=args.psd_preview,
            psd_preview_dir=args.psd_preview_dir,
            use_tqdm=args.use_tqdm
        )

        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            sampling_rate=args.sampling_rate,
            attrs_extra={
                'mode': 'fixed',
                'approximant': 'SEOBNRv4',
                'f_lower': float(args.f_lower),
                'sampling_rate': int(args.sampling_rate),
                'config_args': args_json
            },
            detectors_bytes=detectors,
            full_psd_list=full_psd_list
        )

    elif args.mode == 'random':
        sample_specs = []
        successes = 0
        attempts = 0
        max_attempts = args.num_samples * 10  # safety cap

        while successes < args.num_samples and attempts < max_attempts:
            attempts += 1
            m1_val = np.random.uniform(args.mass1_min, args.mass1_max)
            m2_val = np.random.uniform(args.mass2_min, m1_val)  # ensure m2 <= m1 (unordered)
            snr_val = np.random.uniform(args.snr_min, args.snr_max)
            s1_val = np.random.uniform(args.spin1_min, args.spin1_max)
            s2_val = np.random.uniform(args.spin2_min, args.spin2_max)

            spec = dict(
                mass1=m1_val, mass2=m2_val, target_snr=snr_val,
                spin1z=s1_val, spin2z=s2_val, distance=410.0,
                f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0
            )
            # Probe once; if waveform fails, skip it
            try:
                _ = generate_ligo_waveform(**spec, random_seed=attempts, plot=False)
            except Exception as e:
                print(f"skipped ({attempts}) m1={m1_val:.1f}, m2={m2_val:.1f}, s1={s1_val:.2f}, s2={s2_val:.2f}, snr={snr_val:.1f} --> {e}")
                continue

            sample_specs.append(spec)
            successes += 1
            if successes % 20 == 0 or successes == args.num_samples:
                print(f"{successes}/{args.num_samples} collected")

        if successes < args.num_samples:
            raise RuntimeError(
                f"unable to collect enough valid samples: {successes}/{args.num_samples} (attempted {attempts}). "
                f"please narrow ranges or adjust --f-lower."
            )

        sig_list, noise_list, noisy_list, t_list, meta, detectors, full_psd_list = collect_samples(
            sample_specs,
            plot_flag=args.plot,
            progress_every=args.progress_every,
            save_psd=args.save_psd,
            psd_preview=args.psd_preview,
            psd_preview_dir=args.psd_preview_dir,
            use_tqdm=args.use_tqdm
        )

        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            sampling_rate=args.sampling_rate,
            attrs_extra={
                'mode': 'random',
                'approximant': 'SEOBNRv4',
                'f_lower': float(args.f_lower),
                'sampling_rate': int(args.sampling_rate),
                'config_args': args_json
            },
            detectors_bytes=detectors,
            full_psd_list=full_psd_list
        )

    elif args.mode == 'grid':
        import itertools

        rng = np.random.default_rng(args.seed)
        n = max(2, int(args.grid_steps))
        m1_vals = np.linspace(args.mass1_min, args.mass1_max, n)
        m2_vals = np.linspace(args.mass2_min, args.mass2_max, n)

        # Build unordered pairs (m2 <= m1)
        combos_all = [(float(m1), float(m2)) for m1, m2 in itertools.product(m1_vals, m2_vals) if m2 <= m1]
        combos_all = sorted(combos_all, key=lambda t: (t[0], t[1]))  # deterministic order

        valid_combos = []
        missing = []

        iterable = _maybe_tqdm(enumerate(combos_all), total=len(combos_all),
                               desc="Probing grid", use_tqdm=args.use_tqdm)

        for pi, (m1, m2) in iterable:
            spec_probe = dict(
                mass1=max(m1, m2), mass2=min(m1, m2),
                target_snr=20.0,
                spin1z=0.0, spin2z=0.0,
                distance=410.0, f_lower=args.f_lower,
                sampling_rate=args.sampling_rate, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0
            )
            try:
                _ = generate_ligo_waveform(**spec_probe, random_seed=0, plot=False)
                valid_combos.append((m1, m2))
            except Exception as e:
                missing.append((m1, m2))
                if not _HAS_TQDM or not args.use_tqdm:
                    print(f"[probe] excluding combo (m1={m1}, m2={m2}) --> {e}", flush=True)

            if (not _HAS_TQDM or not args.use_tqdm) and args.progress_every:
                if ((pi + 1) % args.progress_every == 0) or ((pi + 1) == len(combos_all)):
                    print(f"[grid] probe {pi + 1}/{len(combos_all)} | valid={len(valid_combos)}", flush=True)

        if args.require_complete_grid and missing:
            raise RuntimeError(
                f"Grid not complete at f_lower={args.f_lower} Hz; missing {len(missing)} combos: {missing}\n"
                f"Tip: try adjusting --f-lower (e.g., a little higher) or narrowing mass ranges."
            )

        combos = valid_combos
        C = len(combos)
        if C == 0:
            raise RuntimeError("no valid (m1,m2) combos after probe. ADJUST --f-lower or ranges.")

        N_target = int(np.ceil(args.num_samples * args.overgen_factor))
        base = N_target // C
        rem = N_target % C

        def draw_spin(min_v, max_v):
            return float(min_v) if min_v == max_v else float(rng.uniform(min_v, max_v))

        sample_specs = []
        built = 0

        iterable2 = _maybe_tqdm(enumerate(combos), total=len(combos),
                                desc="Building specs", use_tqdm=args.use_tqdm)

        for idx, (m1, m2) in iterable2:
            count = base + (1 if idx < rem else 0)
            if count <= 0:
                continue

            if args.augment_symmetric:
                count_a = count // 2
                count_b = count - count_a

                for _ in range(count_a):
                    s1 = draw_spin(args.spin1_min, args.spin1_max)
                    s2 = draw_spin(args.spin2_min, args.spin2_max)
                    snr_val = float(rng.uniform(args.snr_min, args.snr_max))
                    spec = dict(
                        mass1=m1, mass2=m2, spin1z=s1, spin2z=s2,
                        target_snr=snr_val, distance=410.0,
                        f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                        ra=0.0, dec=0.0, polarization=0.0,
                        label_m1=m1, label_m2=m2, label_s1=s1, label_s2=s2
                    )
                    sample_specs.append(spec); built += 1

                for _ in range(count_b):
                    s1 = draw_spin(args.spin1_min, args.spin1_max)
                    s2 = draw_spin(args.spin2_min, args.spin2_max)
                    snr_val = float(rng.uniform(args.snr_min, args.snr_max))
                    spec = dict(
                        mass1=m2, mass2=m1, spin1z=s2, spin2z=s1,
                        target_snr=snr_val, distance=410.0,
                        f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                        ra=0.0, dec=0.0, polarization=0.0,
                        label_m1=m2, label_m2=m1, label_s1=s2, label_s2=s1
                    )
                    sample_specs.append(spec); built += 1
            else:
                for _ in range(count):
                    s1 = draw_spin(args.spin1_min, args.spin1_max)
                    s2 = draw_spin(args.spin2_min, args.spin2_max)
                    snr_val = float(rng.uniform(args.snr_min, args.snr_max))
                    spec = dict(
                        mass1=m1, mass2=m2, spin1z=s1, spin2z=s2,
                        target_snr=snr_val, distance=410.0,
                        f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                        ra=0.0, dec=0.0, polarization=0.0,
                        label_m1=m1, label_m2=m2, label_s1=s1, label_s2=s2
                    )
                    sample_specs.append(spec); built += 1

            if (not _HAS_TQDM or not args.use_tqdm) and args.progress_every:
                if ((idx + 1) % args.progress_every == 0) or ((idx + 1) == len(combos)):
                    print(f"built specs {built}/{N_target}", flush=True)

        if args.shuffle:
            rng.shuffle(sample_specs)

        sig_list, noise_list, noisy_list, t_list, meta, detectors, full_psd_list = collect_samples(
            sample_specs,
            plot_flag=args.plot,
            progress_every=args.progress_every,
            save_psd=args.save_psd,
            psd_preview=args.psd_preview,
            psd_preview_dir=args.psd_preview_dir,
            use_tqdm=args.use_tqdm
        )

        succ = len(sig_list)
        if succ >= args.num_samples:
            keep = args.num_samples
            sig_list = sig_list[:keep]; noise_list = noise_list[:keep]
            noisy_list = noisy_list[:keep]; t_list = t_list[:keep]
            for k in meta.keys():
                meta[k] = meta[k][:keep]
            if detectors is not None:
                detectors = detectors[:keep]
            if full_psd_list is not None:
                full_psd_list = full_psd_list[:keep]
        else:
            print(f"[WARNING] only generated {succ}/{args.num_samples} after filtering failures. "
                  f"Try adjusting --overgen-factor or re-running.")

        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            sampling_rate=args.sampling_rate,
            attrs_extra={
                'mode': 'grid',
                'grid_steps': int(n),
                'num_combos': int(len(combos)),
                'approximant': 'SEOBNRv4',
                'f_lower': float(args.f_lower),
                'sampling_rate': int(args.sampling_rate),
                'snr_min': float(args.snr_min),
                'snr_max': float(args.snr_max),
                'augment_symmetric': bool(args.augment_symmetric),
                'shuffle': bool(args.shuffle),
                'seed': int(args.seed),
                'overgen_factor': float(args.overgen_factor),
                'require_complete_grid': bool(args.require_complete_grid),
                'config_args': args_json
            },
            detectors_bytes=detectors,
            full_psd_list=full_psd_list
        )
