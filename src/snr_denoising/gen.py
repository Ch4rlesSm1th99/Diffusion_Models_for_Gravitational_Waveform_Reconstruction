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

"""
gen.py — Generate time-domain GW waveforms with LIGO-like noise, and write HDF5 datasets with separate properties.

Modes:
  - fixed : repeat one wave config for the entire dataset
  - random: randomly sample wave configs within ranges
  - grid  : balanced coverage of (m1, m2) pairs in a grid (unordered: m2 <= m1)

Output:
  HDF5 containing signal/noise/noisy, times or mask (depending on padding), metadata and gen mode.

This version has **no fallbacks**. If a combo fails (e.g., SEOBNRv4 at given f_lower), it is skipped.
Optionally require a complete grid via --require-complete-grid to hard-fail if anything is missing.
"""

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


def collect_samples(sample_specs, plot_flag=False, progress_every=0):
    """
    Generate samples from a list of specs (kwargs for generate_ligo_waveform).
    Enforces m1 >= m2 for generation safety (labels may be unsorted if in args).

    Each spec must contain at least: mass1, mass2, target_snr, spin1z, spin2z
    """
    sig_list, noise_list, noisy_list, t_list = [], [], [], []

    meta = {k: [] for k in [
        'mass1','mass2','snr','spin1z','spin2z',
        'label_m1','label_m2','label_s1','label_s2',
        'q','chirp_mass'
    ]}
    meta['epoch'] = []

    for i, spec in enumerate(sample_specs):
        # labels to record (may be swapped separately for symmetric augmentation)
        Lm1 = spec.get('label_m1', spec['mass1'])
        Lm2 = spec.get('label_m2', spec['mass2'])
        Ls1 = spec.get('label_s1', spec['spin1z'])
        Ls2 = spec.get('label_s2', spec['spin2z'])

        # sort for generator call
        m1, m2 = float(spec['mass1']), float(spec['mass2'])
        s1, s2 = float(spec['spin1z']), float(spec['spin2z'])
        if m1 < m2:
            m1, m2, s1, s2 = m2, m1, s2, s1

        call_kwargs = {
            'mass1': m1,
            'mass2': m2,
            'target_snr': float(spec['target_snr']),
            'spin1z': s1,
            'spin2z': s2,
            'distance': float(spec.get('distance', 410.0)),
            'f_lower': float(spec.get('f_lower', 20.0)),
            'sampling_rate': int(spec.get('sampling_rate', 4096)),
            'detector': spec.get('detector', 'H1'),
            'ra': float(spec.get('ra', 0.0)),
            'dec': float(spec.get('dec', 0.0)),
            'polarization': float(spec.get('polarization', 0.0)),
        }

        try:
            plot_this = plot_flag and i < 3
            res = generate_ligo_waveform(**call_kwargs, random_seed=i, plot=plot_this)
        except Exception as e:
            print(f"generation failed for labeled {spec}: {e}")
            continue

        sig_list.append(res['signal'].numpy())
        noise_list.append(res['noise'].numpy())
        noisy_list.append(res['noisy_signal'].numpy())
        t_list.append(res['times'])

        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == len(sample_specs)):
            print(f"generated {i + 1}/{len(sample_specs)}")

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

    return sig_list, noise_list, noisy_list, t_list, meta


def finalize_and_write(
    output_path,
    sig_list, noise_list, noisy_list, t_list, meta,
    padding_mode,
    sampling_rate,
    physical_len=None,
    attrs_extra=None
):
    """Apply padding strategy and write HDF5 with a consistent layout."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lengths = np.array([len(x) for x in sig_list], dtype=np.int32)

    if padding_mode == 'none':
        vlen_f32 = h5py.special_dtype(vlen=np.float32)
        vlen_f64 = h5py.special_dtype(vlen=np.float64)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('signal', (len(sig_list),), dtype=vlen_f32, data=sig_list)
            f.create_dataset('noise',  (len(noise_list),), dtype=vlen_f32, data=noise_list)
            f.create_dataset('noisy',  (len(noisy_list),), dtype=vlen_f32, data=noisy_list)
            f.create_dataset('times',  (len(t_list),),   dtype=vlen_f64, data=t_list)
            f.create_dataset('lengths', data=lengths)
            f.attrs['padding'] = 'none'
            f.attrs['sampling_rate'] = float(sampling_rate)
            if attrs_extra:
                for k, v in attrs_extra.items():
                    f.attrs[k] = v
            for k, arr in meta.items():
                if arr:
                    f.create_dataset(k, data=np.array(arr, dtype=np.float32))
        print(f"saved {len(sig_list)} samples (padding=none) --> {output_path}")
        return

    # padded modes
    if padding_mode == 'dataset_max':
        target_len = int(lengths.max())
        pad_attr_name, pad_attr_val = 'max_length', target_len
    elif padding_mode == 'physical_max':
        assert physical_len and physical_len > 0, "physical_len required for physical_max"
        target_len = int(physical_len)
        pad_attr_name, pad_attr_val = 'physical_length', target_len
    else:
        raise ValueError(f"unrecognised padding_mode: {padding_mode}")

    delta_t = 1.0 / float(sampling_rate)
    times = np.arange(target_len) * delta_t

    N = len(sig_list)
    signals = np.zeros((N, target_len), dtype=np.float32)
    noises  = np.zeros_like(signals)
    noisy   = np.zeros_like(signals)
    mask    = np.zeros_like(signals)

    for i, L in enumerate(lengths):
        Lc = min(L, target_len)
        signals[i, :Lc] = sig_list[i][:Lc]
        noises[i,  :Lc] = noise_list[i][:Lc]
        noisy[i,   :Lc] = noisy_list[i][:Lc]
        mask[i,    :Lc] = 1.0

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('times',   data=times)
        chunk_len = min(16384, target_len)
        dset_kwargs = dict(compression="gzip", compression_opts=4, shuffle=True, chunks=(1, chunk_len))

        f.create_dataset('signal', data=signals, **dset_kwargs)
        f.create_dataset('noise',  data=noises,  **dset_kwargs)
        f.create_dataset('noisy',  data=noisy,   **dset_kwargs)
        f.create_dataset('mask',   data=mask,    **dset_kwargs)

        f.create_dataset('lengths', data=lengths)
        f.attrs['padding'] = padding_mode
        f.attrs['sampling_rate'] = float(sampling_rate)
        f.attrs[pad_attr_name] = pad_attr_val
        if attrs_extra:
            for k, v in attrs_extra.items():
                f.attrs[k] = v
        for k, arr in meta.items():
            if arr:
                f.create_dataset(k, data=np.array(arr, dtype=np.float32))
    print(f"saved {N} samples (padding={padding_mode}, {pad_attr_name}={pad_attr_val}) --> {output_path}")


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
            "    - signal/noise/noisy arrays\n"
            "    - times (variable-length mode) OR (times, mask) in padded modes\n"
            "    - metadata per sample (masses, spins, SNR, epoch, symmetric helpers like q, chirp_mass)\n"
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
                        help='If set, raise an error if any (m1,m2) pair fails during the probe step.')

    # PADDING
    g_pad = parser.add_argument_group("Padding")
    g_pad.add_argument('--padding', choices=['dataset_max', 'none', 'physical_max'], default='dataset_max',
                       help=("Padding strategy:\n"
                             "  none         : variable-length datasets (vlen)\n"
                             "  dataset_max  : pad/truncate all to the max length observed in this run\n"
                             "  physical_max : pad/truncate all to a reference physical length (see --phys-*)"))
    g_pad.add_argument('--phys-m1', type=float, default=10.0, help='Reference mass1 for physical_max padding.')
    g_pad.add_argument('--phys-m2', type=float, default=10.0, help='Reference mass2 for physical_max padding.')
    g_pad.add_argument('--phys-s1', type=float, default=0.0, help='Reference spin1z for physical_max padding.')
    g_pad.add_argument('--phys-s2', type=float, default=0.0, help='Reference spin2z for physical_max padding.')

    # MISC physics controls (uniform across all modes)
    g_misc = parser.add_argument_group("Physics")
    g_misc.add_argument('--f-lower', type=float, default=20.0,
                        help='Low-frequency cutoff passed to get_td_waveform (uniform, no fallbacks).')
    g_misc.add_argument('--sampling-rate', type=int, default=4096,
                        help='Sampling rate (Hz).')

    args = parser.parse_args()

    import json
    args_json = json.dumps(vars(args), sort_keys=True)

    if args.mass2_min > args.mass1_max:
        raise ValueError("mass2_min must be <= mass1_max; otherwise no (m2 <= m1) pairs exist for the grid.")

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

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
        sig_list, noise_list, noisy_list, t_list, meta = collect_samples(
            sample_specs, plot_flag=args.plot, progress_every=args.progress_every
        )

        phys_len = None
        if args.padding == 'physical_max':
            phys_len = len(generate_ligo_waveform(
                mass1=args.phys_m1, mass2=args.phys_m2, target_snr=10.0,
                spin1z=args.phys_s1, spin2z=args.phys_s2, distance=410.0,
                f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0, random_seed=0, plot=False
            )['signal'])
        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            padding_mode=args.padding,
            sampling_rate=args.sampling_rate,
            physical_len=phys_len,
            attrs_extra={
                'mode': 'fixed',
                'approximant': 'SEOBNRv4',
                'f_lower': float(args.f_lower),
                'sampling_rate': int(args.sampling_rate),
                'config_args': args_json
            }
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
            raise RuntimeError(f"unable to collect enough valid samples: {successes}/{args.num_samples} (attempted {attempts}). "
                               f"please narrow ranges or adjust f-lower.")

        sig_list, noise_list, noisy_list, t_list, meta = collect_samples(
            sample_specs, plot_flag=args.plot, progress_every=args.progress_every
        )

        phys_len = None
        if args.padding == 'physical_max':
            phys_len = len(generate_ligo_waveform(
                mass1=args.phys_m1, mass2=args.phys_m2, target_snr=10.0,
                spin1z=args.phys_s1, spin2z=args.phys_s2, distance=410.0,
                f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0, random_seed=0, plot=False
            )['signal'])

        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            padding_mode=args.padding,
            sampling_rate=args.sampling_rate,
            physical_len=phys_len,
            attrs_extra={
                'mode': 'random',
                'approximant': 'SEOBNRv4',
                'f_lower': float(args.f_lower),
                'sampling_rate': int(args.sampling_rate),
                'config_args': args_json
            }
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
        total_probe = len(combos_all)

        for pi, (m1, m2) in enumerate(combos_all):
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
                print(f"[probe] excluding combo (m1={m1}, m2={m2}) --> {e}", flush=True)

            if args.progress_every and ((pi + 1) % args.progress_every == 0 or (pi + 1) == total_probe):
                print(f"[grid] probe {pi + 1}/{total_probe} | valid={len(valid_combos)}", flush=True)

        if args.require_complete_grid and missing:
            raise RuntimeError(f"Grid not complete at f_lower={args.f_lower} Hz; missing {len(missing)} combos: {missing}")

        combos = valid_combos
        C = len(combos)
        if C == 0:
            raise RuntimeError("no valid (m1,m2) combos after probe. ADJUST f_lower or ranges.")

        N_target = int(np.ceil(args.num_samples * args.overgen_factor))
        base = N_target // C
        rem = N_target % C

        def draw_spin(min_v, max_v):
            return float(min_v) if min_v == max_v else float(rng.uniform(min_v, max_v))

        sample_specs = []
        built = 0

        for idx, (m1, m2) in enumerate(combos):
            count = base + (1 if idx < rem else 0)
            if count <= 0:
                continue

            if args.augment_symmetric:
                # Half with labels (m1,m2), half with swapped labels (m2,m1)
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
                    if args.progress_every and (built % args.progress_every == 0): print(f"built specs {built}/{N_target}", flush=True)

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
                    if args.progress_every and (built % args.progress_every == 0): print(f"built specs {built}/{N_target}", flush=True)

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
                    if args.progress_every and (built % args.progress_every == 0): print(f"built specs {built}/{N_target}", flush=True)

        if args.shuffle:
            rng.shuffle(sample_specs)

        sig_list, noise_list, noisy_list, t_list, meta = collect_samples(
            sample_specs, plot_flag=args.plot, progress_every=args.progress_every
        )

        succ = len(sig_list)
        if succ >= args.num_samples:
            keep = args.num_samples
            sig_list = sig_list[:keep]; noise_list = noise_list[:keep]
            noisy_list = noisy_list[:keep]; t_list = t_list[:keep]
            for k in meta.keys():
                meta[k] = meta[k][:keep]
        else:
            print(f"[WARNING] only generated {succ}/{args.num_samples} after filtering failures. "
                  f"Try adjusting --overgen-factor or re-running.")

        phys_len = None
        if args.padding == 'physical_max':
            phys_len = len(generate_ligo_waveform(
                mass1=args.phys_m1, mass2=args.phys_m2, target_snr=10.0,
                spin1z=args.phys_s1, spin2z=args.phys_s2, distance=410.0,
                f_lower=args.f_lower, sampling_rate=args.sampling_rate, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0, random_seed=0, plot=False
            )['signal'])

        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            padding_mode=args.padding,
            sampling_rate=args.sampling_rate,
            physical_len=phys_len,
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
            }
        )
