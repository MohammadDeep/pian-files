import numpy as np
def signal_window(signal, window_size, step = 1, save_len = True):
    # تعداد نمونه‌های سیگنال
    len_signall = len(signal)

    values = []
    if save_len  and step != 1 :
      #print('save len is avalabel   -> step = 1')
      step = 1
    # حرکت پنجره بر روی سیگنال
    for i in range((len_signall - window_size)//step + 1):
        window = signal[i * step:i * step + window_size]  # سیگنال در پنجره
        values.append(window)
    if save_len :
      number_add_st = (window_size -1) // 2
      number_add_end = window_size - 1 - number_add_st
      values_str = values[0]
      values_end = values[-1]
      for i in range(number_add_st):
        values.insert(0, values_str[:-i - 1])
      for i in range(number_add_end):

        values.append(values_end[i + 1:])




    list_array  = []
    for i in values:
      list_array.append(np.array(i))
    return list_array



def range_window(signal, window_size):
  try :
    a = len(signal[1])
    windows = signal
  except:
    #print('create_window')
    windows = signal_window(signal, window_size)
  values = []
  for window in windows:

    values.append(np.max(window) - np.min(window))  # محاسبه واریانس پنجره

  return np.array(values)

import numpy as np

def ave_window(signal, window_size):
  try :
    a = len(signal[1])
    windows = signal
  except:
    #print('create_window')
    windows = signal_window(signal, window_size)
  values = []
  for window in windows:

    values.append(np.mean(window))  # محاسبه واریانس پنجره

  return np.array(values)

def max_window(signal, window_size):
  try :
    a = len(signal[1])
    windows = signal
  except:
    #print('create_window')
    windows = signal_window(signal, window_size)
  values = []
  for window in windows:

    values.append(np.max(window) )  # محاسبه واریانس پنجره

  return np.array(values)
def min_window(signal, window_size):
  try :
    a = len(signal[1])
    windows = signal
  except:
    #print('create_window')
    windows = signal_window(signal, window_size)
  values = []
  for window in windows:

    values.append(np.min(window))  # محاسبه واریانس پنجره

  return np.array(values)


import numpy as np

def variance_window(signal, window_size):
  try :
    a = len(signal[1])
    windows = signal
  except:
    #print('create_window')
    windows = signal_window(signal, window_size)
  values = []
  for window in windows:

    values.append(np.var(window))  # محاسبه واریانس پنجره

  return np.array(values)


def dic_variance_window(dic_data, window_size):
  dic_variance = {}
  for label, signall in dic_data.items():
    ##print('signal',signall)
    dic_variance['var_' + label] = variance_window(signall, window_size)
  return dic_variance