

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


def data_df(data,
           save_len = True):
    
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
    df_data = np.diff(data)   # vectorized difference calculation
    if save_len:
        df_data = np.append(df_data, df_data[-1]) 
    return df_data


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

def show_df_pec(signal, perc):
  new_signal = []
  df_signall = data_df(signal)

  max_df = max(df_signall)
  min_df = min(df_signall)
  sum = 0
  for i in df_signall:
    sum = sum + i
  mean = sum / len(df_signall)
  #print(mean)
  for i in df_signall:
    if i > max_df * perc:
      new_signal.append(abs(i))
    elif i < min_df * perc:
      new_signal.append(abs(i))
    else :
      new_signal.append(abs(mean))
  return new_signal


def dic_show_df_pec(dic_data, perc = 0.4):
  dic_df_pec = {}
  for label, signall in dic_data.items():
    dic_df_pec['show_df_pec_' + label] = show_df_pec(signall, perc)
  return dic_df_pec