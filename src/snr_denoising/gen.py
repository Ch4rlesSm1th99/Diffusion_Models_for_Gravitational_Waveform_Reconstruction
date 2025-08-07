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
    Gravitational waveform generator with LIGO noise injections.  Uses PyCBC's LIGO PSD.
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


def generate_dataset(
    num_samples,
    output_path,
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
    polarization=0.0
):
    """
    Generates dataset of num_noisy samples and plots examples depending on
    data gen mode arg
    """
    # Warm-up call to generate waveform and cache PSD
    base = generate_ligo_waveform(
        mass1=mass1,
        mass2=mass2,
        target_snr=target_snr,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=distance,
        f_lower=f_lower,
        sampling_rate=sampling_rate,
        detector=detector,
        ra=ra,
        dec=dec,
        polarization=polarization,
        random_seed=0,
        plot=False
    )
    times = base['times']
    npt = len(times)

    # Prepare arrays
    signals = np.zeros((num_samples, npt), dtype=np.float32)
    noises = np.zeros_like(signals)
    noisy = np.zeros_like(signals)

    # Generate and collect
    for i in range(num_samples):
        res = generate_ligo_waveform(
            mass1=mass1,
            mass2=mass2,
            target_snr=target_snr,
            spin1z=spin1z,
            spin2z=spin2z,
            distance=distance,
            f_lower=f_lower,
            sampling_rate=sampling_rate,
            detector=detector,
            ra=ra,
            dec=dec,
            polarization=polarization,
            random_seed=i,
            plot=False
        )
        signals[i] = res['signal'].numpy()
        noises[i] = res['noise'].numpy()
        noisy[i]  = res['noisy_signal'].numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('times',  data=times)
        f.create_dataset('signal', data=signals)
        f.create_dataset('noise',  data=noises)
        f.create_dataset('noisy',  data=noisy)
        params = dict(
            mass1=mass1, mass2=mass2, target_snr=target_snr,
            spin1z=spin1z, spin2z=spin2z, distance=distance,
            f_lower=f_lower, sampling_rate=sampling_rate,
            detector=detector, ra=ra, dec=dec, polarization=polarization
        )
        for k, v in params.items():
            f.attrs[k] = v

    print(f"Saved {num_samples} samples at {output_path}")

    for idx in range(min(3, num_samples)):
        signal_array = signals[idx]
        noise_array  = noises[idx]
        noisy_array  = noisy[idx]

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
    args = parser.parse_args()

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.mode == 'fixed':
        generate_dataset(
            num_samples   = args.num_samples,
            output_path   = args.output_path,
            mass1         = args.mass1,
            mass2         = args.mass2,
            target_snr    = args.snr,
            spin1z        = 0.0,
            spin2z        = 0.0,
            distance      = 410.0,
            f_lower       = 20.0,
            sampling_rate = 4096,
            detector      = "H1",
            ra            = 0.0,
            dec           = 0.0,
            polarization  = 0.0
        )

    elif args.mode == 'random':

        sig_list, noise_list, noisy_list = [], [], []
        meta_m1, meta_m2 = [], []
        meta_snr, meta_s1, meta_s2 = [], [], []

        i = 0
        while i < args.num_samples:
            if (i + 1) % 100 == 0 or (i + 1) == args.num_samples:
                print(f"{i + 1}/{args.num_samples} generated")
            # sample masses so that m1 >= m2
            m1_val = np.random.uniform(args.mass1_min, args.mass1_max)
            m2_val = np.random.uniform(args.mass2_min, m1_val)

            # sample SNR and spins
            snr_val = np.random.uniform(args.snr_min, args.snr_max)
            s1_val = np.random.uniform(args.spin1_min, args.spin1_max)
            s2_val = np.random.uniform(args.spin2_min, args.spin2_max)

            # raise value error with m1 > m2
            try:
                res = generate_ligo_waveform(
                    mass1=m1_val,
                    mass2=m2_val,
                    target_snr=snr_val,
                    spin1z=s1_val,
                    spin2z=s2_val,
                    random_seed=i,
                    plot=False
                )
            except RuntimeError as e:
                print(f"attempt {i} failed (m1={m1_val:.1f}, m2={m2_val:.1f}, snr={snr_val:.1f}): {e}")
                continue  # do NOT increment i, retry

            sig_list.append(res['signal'].numpy())
            noise_list.append(res['noise'].numpy())
            noisy_list.append(res['noisy_signal'].numpy())
            meta_m1.append(m1_val)
            meta_m2.append(m2_val)
            meta_snr.append(snr_val)
            meta_s1.append(s1_val)
            meta_s2.append(s2_val)
            i += 1

        # determine max length for padding
        lengths = [len(x) for x in sig_list]
        max_len = max(lengths)

        sampling_rate = 4096  # if it changes in gen_ligo_waveform remember to change to that
        delta_t = 1.0 / sampling_rate
        times = np.arange(max_len) * delta_t

        # assign padded arrays
        signals = np.zeros((args.num_samples, max_len), dtype=np.float32)
        noises = np.zeros_like(signals)
        noisy = np.zeros_like(signals)

        for i, L in enumerate(lengths):
            signals[i, :L] = sig_list[i]
            noises[i, :L] = noise_list[i]
            noisy[i, :L] = noisy_list[i]

        # metadata
        meta_m1 = np.array(meta_m1, dtype=np.float32)
        meta_m2 = np.array(meta_m2, dtype=np.float32)
        meta_snr = np.array(meta_snr, dtype=np.float32)
        meta_s1 = np.array(meta_s1, dtype=np.float32)
        meta_s2 = np.array(meta_s2, dtype=np.float32)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with h5py.File(args.output_path, 'w') as f:
            f.create_dataset('times', data=np.arange(max_len) * (1 / 4096.))
            f.create_dataset('signal', data=signals)
            f.create_dataset('noise', data=noises)
            f.create_dataset('noisy', data=noisy)
            f.create_dataset('mass1', data=meta_m1)
            f.create_dataset('mass2', data=meta_m2)
            f.create_dataset('snr', data=meta_snr)
            f.create_dataset('spin1z', data=meta_s1)
            f.create_dataset('spin2z', data=meta_s2)
            f.attrs['mode'] = 'random'
            f.attrs['max_length'] = max_len
        print(f"Saved random dataset ({args.num_samples} samples, padded to {max_len}) at {args.output_path}")

        for idx in range(min(3, args.num_samples)):
            # retrieve the padded arrays
            signal_array = signals[idx]
            noise_array = noises[idx]
            noisy_array = noisy[idx]

            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1)
            plt.plot(times, signal_array)
            plt.title(
                f'Random Sample {idx}: Clean (m1={meta_m1[idx]:.1f}, m2={meta_m2[idx]:.1f}, SNR={meta_snr[idx]:.1f})')
            plt.ylabel('Strain')
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(times, noise_array)
            plt.title('Detector Noise')
            plt.ylabel('Strain')
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(times, noisy_array)
            plt.title('Noisy Signal')
            plt.xlabel('Time Sample Index')
            plt.ylabel('Strain')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    elif args.mode == 'grid':
        pass

