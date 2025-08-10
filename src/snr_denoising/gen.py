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

# PSD cache to append interpolated values. this will avoid calling the interpolator at every waveform computation.
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
    Gravitational waveform generator with LIGO noise injections.  Uses PyCBC's LIGO PSD noise.
    Note PyCBC's wheel breaks when run with microsoft compiler --> USE LINUX ENVIRONMENT
    """
    # time step
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

    # project onto the chosen detector
    det = Detector(detector)
    signal = det.project_wave(hp, hc, ra, dec, polarization)
    waveform_epoch = signal._epoch

    # retrieve PSD
    psd_key = f"{detector}_{sampling_rate}_{len(signal)}"
    if psd_key in _PSD_CACHE:
        psd = _PSD_CACHE[psd_key]
    else:
        df = 1.0 / (len(signal) * delta_t)
        psd = aLIGOZeroDetHighPower(len(signal)//2 + 1, df, f_lower)
        _PSD_CACHE[psd_key] = psd

    # scale to target SNR
    current_snr = sigma(signal, psd=psd, low_frequency_cutoff=f_lower)
    scaled_signal = signal * (target_snr / current_snr)

    # generate and align noise
    noise = noise_from_psd(len(signal), delta_t, psd, seed=random_seed)
    noise._epoch = waveform_epoch
    noisy_signal = scaled_signal + noise

    # prepare time array
    start_time = float(waveform_epoch)
    times = np.arange(len(signal)) * delta_t + start_time

    # convert for plot
    signal_array = scaled_signal.numpy()
    noise_array = noise.numpy()
    noisy_array = noisy_signal.numpy()

    # Optional plotting
    if plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(times, signal_array)
        plt.title(f'LIGO {detector} Signal (m1={mass1}M☉, m2={mass2}M☉)')
        plt.ylabel('Strain')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(times, noise_array)
        plt.title('Detector Noise')
        plt.ylabel('Strain')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(times, noisy_array)
        plt.title(f'Noisy Signal (SNR={target_snr:.1f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return {
        'signal': scaled_signal,
        'noise': noise,
        'noisy_signal': noisy_signal,
        'times': times,
        'snr': target_snr,
        'detector': detector,
        'epoch': waveform_epoch
    }


def collect_samples(sample_specs):
    """Generate all samples from a list of specs (each spec is the kwargs for generate_ligo_waveform)."""
    sig_list, noise_list, noisy_list, t_list = [], [], [], []
    meta = {k: [] for k in ['mass1','mass2','snr','spin1z','spin2z']}
    for i, spec in enumerate(sample_specs):
        try:
            plot_this = args.plot and i < 3
            res = generate_ligo_waveform(**spec, random_seed=i, plot=plot_this)
        except Exception as e:
            print(f"generation failed for {spec}: {e}")
            continue
        sig_list.append(res['signal'].numpy())
        noise_list.append(res['noise'].numpy())
        noisy_list.append(res['noisy_signal'].numpy())
        t_list.append(res['times'])
        for k in meta.keys():
            if k in spec: meta[k].append(spec[k])
    return sig_list, noise_list, noisy_list, t_list, meta


def finalize_and_write(
    output_path,
    sig_list, noise_list, noisy_list, t_list, meta,
    padding_mode,
    sampling_rate,
    physical_len=None,
    attrs_extra=None
):
    """Apply padding strategy once and write a HDF5 with consistent layout for the modes."""
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
                for k,v in attrs_extra.items(): f.attrs[k] = v
            for k, arr in meta.items():
                if arr: f.create_dataset(k, data=np.array(arr, dtype=np.float32))
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
        raise ValueError(f"Unknown padding_mode: {padding_mode}")

    delta_t = 1.0 / float(sampling_rate)
    times = np.arange(target_len) * delta_t

    N = len(sig_list)
    signals = np.zeros((N, target_len), dtype=np.float32)
    noises  = np.zeros_like(signals)
    noisy   = np.zeros_like(signals)
    mask    = np.zeros_like(signals)

    for i, L in enumerate(lengths):
        Lc = min(L, target_len)   # (only truncates if target is too small)
        signals[i, :Lc] = sig_list[i][:Lc]
        noises[i,  :Lc] = noise_list[i][:Lc]
        noisy[i,   :Lc] = noisy_list[i][:Lc]
        mask[i,    :Lc] = 1.0

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('times',   data=times)
        f.create_dataset('signal',  data=signals)
        f.create_dataset('noise',   data=noises)
        f.create_dataset('noisy',   data=noisy)
        f.create_dataset('mask',    data=mask)
        f.create_dataset('lengths', data=lengths)
        f.attrs['padding'] = padding_mode
        f.attrs['sampling_rate'] = float(sampling_rate)
        f.attrs[pad_attr_name] = pad_attr_val
        if attrs_extra:
            for k,v in attrs_extra.items(): f.attrs[k] = v
        for k, arr in meta.items():
            if arr: f.create_dataset(k, data=np.array(arr, dtype=np.float32))
    print(f"Saved {N} samples (padding={padding_mode}, {pad_attr_name}={pad_attr_val}) → {output_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LIGO waveform datasets")
    parser.add_argument('--mode', choices=['fixed', 'random', 'grid'], default='fixed')
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    # Fixed mode params
    parser.add_argument('--mass1', type=float, default=20.0)
    parser.add_argument('--mass2', type=float, default=20.0)
    parser.add_argument('--snr', type=float, default=8000.0)
    # Random mode ranges
    parser.add_argument('--mass1-min', type=float, default=20.0)
    parser.add_argument('--mass1-max', type=float, default=20.0)
    parser.add_argument('--mass2-min', type=float, default=20.0)
    parser.add_argument('--mass2-max', type=float, default=20.0)
    parser.add_argument('--snr-min', type=float, default=8000.0)
    parser.add_argument('--snr-max', type=float, default=8000.0)
    parser.add_argument('--spin1-min', type=float, default=0.0)
    parser.add_argument('--spin1-max', type=float, default=0.0)
    parser.add_argument('--spin2-min', type=float, default=0.0)
    parser.add_argument('--spin2-max', type=float, default=0.0)
    # Padding options
    parser.add_argument('--padding', choices=['dataset_max', 'none', 'physical_max'],
                        default='dataset_max',
                        help='How to pad variable-length waveforms.')
    parser.add_argument('--phys-m1', type=float, default=10.0)
    parser.add_argument('--phys-m2', type=float, default=10.0)
    parser.add_argument('--phys-s1', type=float, default=0.0)
    parser.add_argument('--phys-s2', type=float, default=0.0)
    # Plotting
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.mode == 'fixed':
        sample_specs = [
            dict(
                mass1=args.mass1, mass2=args.mass2, target_snr=args.snr,
                spin1z=0.0, spin2z=0.0, distance=410.0,
                f_lower=20.0, sampling_rate=4096, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0
            )
            for _ in range(args.num_samples)
        ]
        sig_list, noise_list, noisy_list, t_list, meta = collect_samples(sample_specs)

        phys_len = None
        if args.padding == 'physical_max':
            phys_len = len(generate_ligo_waveform(
                mass1=args.phys_m1, mass2=args.phys_m2, target_snr=10.0,
                spin1z=args.phys_s1, spin2z=args.phys_s2, distance=410.0,
                f_lower=20.0, sampling_rate=4096, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0, random_seed=0, plot=False
            )['signal'])
        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            padding_mode=args.padding,
            sampling_rate=4096,
            physical_len=phys_len,
            attrs_extra={'mode': 'fixed'}
        )


    elif args.mode == 'random':
        sample_specs = []
        successes = 0
        attempts = 0
        max_attempts = args.num_samples * 10  # safety cap

        while successes < args.num_samples and attempts < max_attempts:
            attempts += 1
            m1_val = np.random.uniform(args.mass1_min, args.mass1_max)
            m2_val = np.random.uniform(args.mass2_min, m1_val)  # ensure m2 <= m1
            snr_val = np.random.uniform(args.snr_min, args.snr_max)
            s1_val = np.random.uniform(args.spin1_min, args.spin1_max)
            s2_val = np.random.uniform(args.spin2_min, args.spin2_max)

            spec = dict(
                mass1=m1_val, mass2=m2_val, target_snr=snr_val,
                spin1z=s1_val, spin2z=s2_val, distance=410.0,
                f_lower=20.0, sampling_rate=4096, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0
            )
            # quick probe to avoid counting failures
            try:
                _ = generate_ligo_waveform(**spec, random_seed=attempts, plot=False)
            except Exception as e:
                print(
                    f"skipped ({attempts}) m1={m1_val:.1f}, m2={m2_val:.1f}, s1={s1_val:.2f}, s2={s2_val:.2f}, snr={snr_val:.1f} --> {e}")
                continue
            sample_specs.append(spec)
            successes += 1
            if successes % 20 == 0 or successes == args.num_samples:
                print(f"{successes}/{args.num_samples} collected")

        if successes < args.num_samples:
            raise RuntimeError(f"unable to collect enough valid samples: {successes}/{args.num_samples} "

                               f"(attempted {attempts}). please narrow ranges.")

        sig_list, noise_list, noisy_list, t_list, meta = collect_samples(sample_specs)

        phys_len = None
        if args.padding == 'physical_max':
            phys_len = len(generate_ligo_waveform(
                mass1=args.phys_m1, mass2=args.phys_m2, target_snr=10.0,
                spin1z=args.phys_s1, spin2z=args.phys_s2, distance=410.0,
                f_lower=20.0, sampling_rate=4096, detector="H1",
                ra=0.0, dec=0.0, polarization=0.0, random_seed=0, plot=False
            )['signal'])

        finalize_and_write(
            output_path=args.output_path,
            sig_list=sig_list, noise_list=noise_list, noisy_list=noisy_list, t_list=t_list, meta=meta,
            padding_mode=args.padding,
            sampling_rate=4096,
            physical_len=phys_len,
            attrs_extra={'mode': 'random'}
        )


    elif args.mode == 'grid':
        pass

