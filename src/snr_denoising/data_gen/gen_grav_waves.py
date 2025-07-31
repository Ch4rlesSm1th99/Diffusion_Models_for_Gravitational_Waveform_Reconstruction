import matplotlib.pyplot as plt
import numpy as np
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.detector import Detector
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import sigma

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
    # Time step
    delta_t = 1.0 / sampling_rate

    # Generate the plus & cross polarizations
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

    # Project onto the chosen detector
    det = Detector(detector)
    signal = det.project_wave(hp, hc, ra, dec, polarization)
    waveform_epoch = signal._epoch

    # Build or retrieve PSD
    psd_key = f"{detector}_{sampling_rate}_{len(signal)}"
    if psd_key in _PSD_CACHE:
        psd = _PSD_CACHE[psd_key]
    else:
        df = 1.0 / (len(signal) * delta_t)
        psd = aLIGOZeroDetHighPower(len(signal)//2 + 1, df, f_lower)
        _PSD_CACHE[psd_key] = psd

    # Scale to target SNR
    current_snr = sigma(signal, psd=psd, low_frequency_cutoff=f_lower)
    scaled_signal = signal * (target_snr / current_snr)

    # Generate and align noise
    noise = noise_from_psd(len(signal), delta_t, psd, seed=random_seed)
    noise._epoch = waveform_epoch
    noisy_signal = scaled_signal + noise

    # Prepare time array
    start_time = float(waveform_epoch)
    times = np.arange(len(signal)) * delta_t + start_time

    # Convert to numpy for plotting
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


if __name__ == "__main__":
    result = generate_ligo_waveform(
        mass1=20.0,
        mass2=20.0,
        target_snr=8000,
        plot=True
    )
    print(f"Generated waveform with SNR = {result['snr']}")
