

import numpy as np


def delet_nan(df):
  '''
  df : pd.DataFrame 
  delet row have a None
  '''
  non_d = df.isna().sum()
  print(f'numebr of nan data : {non_d}')
  df = df.dropna()
  return df


def data_d(data):
    
    """
    Input:
        data : list or numpy array of numbers (int/float)
    Output:
        new_data : numpy array of differences between consecutive elements
    Description:
        This function calculates the difference between each pair
        of consecutive elements in the input sequence and returns
        them as a new numpy array.
    """
    data = np.array(data)  # ensure input is a numpy array
    return np.diff(data)   # vectorized difference calculation


import numpy as np
from typing import Sequence

def dphEDA(signal: Sequence[float], fs: float = 2.0, padding: int = 2) -> np.ndarray:
    """
    Compute the first derivative of an EDA signal using a 4th-order
    central-difference stencil and zero-padding at both ends.

    Parameters
    ----------
    signal : Sequence[float]
        Input 1D EDA samples.
    fs : float, optional
        Sampling rate in Hz (default: 2.0).
    padding : int, optional
        Number of zeros to prepend/append so the output length can match
        the input length when padding=2 (default: 2).

    Returns
    -------
    np.ndarray
        The derivative estimate. Length = len(signal) - 4 + 2*padding.
        With padding=2, length equals the input length.

    Notes
    -----
    The 4th-order central-difference kernel is:
        [-1, -8, 0, 8, 1] * (fs / 12)
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if len(x) < 5:
        # Not enough points for a 4th-order stencil; return zeros with padding applied
        core = np.zeros(max(0, len(x) - 4), dtype=np.float64)

        return np.pad(core, (padding, padding), mode="constant")

    # 4th-order central difference kernel, scaled by fs/12
    kernel = np.array([-1.0, -8.0, 0.0, 8.0, 1.0], dtype=np.float64) * (fs / 12.0)

    # Convolve with 'valid' to use only fully available neighbors
    core = np.convolve(x, kernel, mode="valid")  # length = N - 4

    # Zero-pad at both ends (same behavior as your original function)
    y = np.pad(core, (padding, padding), mode="constant")

    return y
