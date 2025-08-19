import numpy as np
from scipy.signal import decimate, medfilt, butter, filtfilt

def preprocess(
    eda_raw: np.ndarray,
    fs_in: float,
    fs_out: float = 2.0,
    hp_cutoff: float = 0.01,
    hp_order: int = 4
) -> np.ndarray:
    """
    Preprocess EDA signal: downsample -> median filter -> high-pass filter.

    Parameters
    ----------
    eda_raw : np.ndarray
        Raw EDA signal (1D).
    fs_in : float
        Input sampling rate (Hz).
    fs_out : float, optional
        Desired output sampling rate (Hz). Default = 2 Hz.
    hp_cutoff : float, optional
        High-pass filter cutoff frequency (Hz). Default = 0.01.
    hp_order : int, optional
        Butterworth high-pass filter order. Default = 4.

    Returns
    -------
    np.ndarray
        Preprocessed EDA signal at fs_out Hz, drift removed.
    """
    x = np.asarray(eda_raw, dtype=np.float64)

    # ---- 1. Downsample ----
    factor = int(round(fs_in / fs_out))
    if not np.isclose(fs_in / factor, fs_out, rtol=1e-3):
        raise ValueError("fs_in/fs_out ratio must be integer (for decimate).")


    # ---- 3. High-pass filter ----
    nyq = 0.5 * fs_out
    wn = hp_cutoff / nyq
    b, a = butter(hp_order, wn, btype="high") # type: ignore
    x = filtfilt(b, a, x)

    return x