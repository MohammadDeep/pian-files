import numpy as np
from typing import Tuple
from scipy.signal import butter, sosfiltfilt, hilbert
from preproses_signals.function.Normalization import decimate_signal
from preproses_signals.function.Filtering_alignment import preprocess
from preproses_signals.function import  cvxEDA
from preproses_signals.function.function import dphEDA
from typing import Optional, Literal,Tuple
from scipy.signal import medfilt

def compute_TVSymp(
    eda_signal: np.ndarray,
    Fs: float,
    band: Tuple[float, float] = (0.08, 0.24),
    n_centers: int = 5,
    lp_cutoff: float = 0.5,
    order: int = 4
    ) -> np.ndarray:
    """
    Compute the time-varying sympathetic activity (TVSymp) from an EDA signal.

    This implementation demodulates the EDA around several center frequencies
    within a target band, low-pass filters the demodulated signals using a
    Butterworth filter in SOS form (for numerical stability), sums them, and
    extracts the instantaneous amplitude via the Hilbert transform.

    Parameters
    ----------
    eda_signal : np.ndarray
        1D preprocessed EDA samples (e.g., filtered/resampled).
    Fs : float
        Sampling rate in Hz (must be > 0).
    band : (float, float), optional
        Target frequency band in Hz, e.g., (0.08, 0.24).
    n_centers : int, optional
        Number of center frequencies within `band` for demodulation (>=1).
    lp_cutoff : float, optional
        Low-pass cutoff (Hz) applied after demodulation (must be < Fs/2).
    order : int, optional
        Butterworth low-pass filter order (>=1).

    Returns
    -------
    np.ndarray
        TVSymp time series (instantaneous amplitude of the combined component).

    Notes
    -----
    - Uses SOS form + `sosfiltfilt` for stable zero-phase filtering.
    - Ensure `band` lies well below Nyquist to avoid edge effects.
    """
    # ---- validate & prepare ----
    x = np.asarray(eda_signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("eda_signal must be a 1D array.")
    if Fs <= 0:
        raise ValueError("Fs must be positive.")
    if order < 1:
        raise ValueError("order must be >= 1.")
    if n_centers < 1:
        raise ValueError("n_centers must be >= 1.")
    nyq = 0.5 * Fs
    if not (0 < lp_cutoff < nyq):
        raise ValueError(f"lp_cutoff must be between (0, {nyq}).")
    f_lo, f_hi = band
    if not (0 < f_lo < f_hi < nyq):
        raise ValueError(f"band must satisfy 0 < {f_lo} < {f_hi} < {nyq} (Hz).")

    N = x.size
    t = np.arange(N, dtype=np.float64) / Fs

    # ---- low-pass filter (SOS form) ----
    wn = lp_cutoff / nyq
    sos = butter(order, wn, btype='low', output='sos')

    # ---- demodulate, low-pass, and combine ----
    center_freqs = np.linspace(f_lo, f_hi, n_centers)
    combined = np.zeros(N, dtype=np.complex128)

    # Precompute exp(j*2π f t) efficiently per frequency
    for f0 in center_freqs:
        demod = x * np.exp(-1j * 2.0 * np.pi * f0 * t)   # shift f0 -> 0 Hz
        filt = sosfiltfilt(sos, demod)                   # zero-phase LP
        combined += filt

    # ---- envelope via Hilbert transform ----
    analytic = hilbert(combined.real)
    TVSymp = np.abs(analytic)   # type: ignore
    return TVSymp

def perpros_EDA(
        input_signal: np.ndarray
):
    

    eda_downsampled,fs1= decimate_signal(
                                    input_signal,
                                    input_fs  = 256,
                                    factor  = 62,
                                    ftype = 'fir')
    

   
    signal_filtered = medfilt(eda_downsampled, 
                              kernel_size=3)

    eda_downsampled_2Hz,fs2 = decimate_signal(signal_filtered,
                                              fs1, 2, ftype='fir') 
    



    signal_filtered = preprocess(
                                eda_raw = eda_downsampled_2Hz,
                                fs_in = fs2,
                                fs_out= fs2,
                                hp_cutoff  = 0.01,
                                hp_order = 4)




    signal_filtered_normaliz = (signal_filtered - np.mean(signal_filtered)) / np.std(signal_filtered)

 
    [r, p, t_est, l, d, e, obj] = cvxEDA.cvxEDA(signal_filtered_normaliz
                                                , 1.0 / fs2)

    dphEDA_r = dphEDA(r, fs2, 2) # type: ignore

    TVSymp = compute_TVSymp(signal_filtered , fs2)

    return {'signalfiltered':signal_filtered, 
            'signalfiltered_normaliz':signal_filtered_normaliz,
            'r' : r, 'p': p, 't_est': t_est, 'l': l, 'd': d, 'e':e , 
            'obj': obj,
            'dphEDA_r':dphEDA_r,
            'TVSymp':TVSymp}
