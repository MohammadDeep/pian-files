import numpy as np
from typing import Optional, Literal,Tuple
from scipy.signal import decimate

def decimate_signal(
    input_signal: np.ndarray,
    input_fs: int,
    factor: int,
    ftype: Literal['iir', 'fir'] = 'iir'
) -> Tuple[np.ndarray, int]:
    """
    Downsample a 1D signal by an integer factor using scipy.signal.decimate.

    Parameters
    ----------
    input_signal : np.ndarray
        1D array of samples representing the input signal.
    input_fs : int
        Input sampling rate (Hz) of the signal.
    factor : int
        Integer decimation factor (output_fs = input_fs / factor).
        Must be an integer >= 2.
    ftype : {'iir', 'fir'}, optional
        Anti-aliasing filter type used by `scipy.signal.decimate`.
        'iir' (default) is faster; 'fir' has linear phase.

    Returns
    -------
    np.ndarray
        The downsampled signal.

    Notes
    -----
    - This function requires an *integer* factor. For fractional
      resampling (e.g., 130 Hz -> 4 Hz), use `scipy.signal.resample_poly`.
    - An anti-aliasing low-pass filter is applied internally to reduce aliasing.
    """
    print("-" * 50)
    print("decimate_signal:")

    # Validate inputs
    if not isinstance(input_signal, np.ndarray):
        raise TypeError("input_signal must be a numpy.ndarray.")
    if input_signal.ndim != 1:
        raise ValueError("input_signal must be 1D.")
    if not isinstance(input_fs, int) or input_fs <= 0:
        raise ValueError("input_fs must be a positive integer (Hz).")
    if not isinstance(factor, int) or factor < 2:
        raise ValueError("factor must be an integer >= 2.")
    if ftype not in ('iir', 'fir'):
        raise ValueError("ftype must be 'iir' or 'fir'.")

    # Perform decimation
    signal_down = decimate(input_signal, factor, ftype=ftype)

    # Informative prints (optional)
    output_fs = input_fs / factor

    print(f"Input length:  {len(input_signal)} samples :{input_fs} Hz")
    print(f"Output length: {len(signal_down)} samples : {output_fs:.6g} Hz output samples{int(output_fs)} ")
    print(f"Factor: {factor} | Filter type: {ftype}")
    print("-" * 50)

    return signal_down,int(output_fs)
