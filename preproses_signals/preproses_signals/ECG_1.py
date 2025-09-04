# data denoising بر اساس مقاله ECG Multi_Emotion Recognition ....
import numpy as np
import pywt
import matplotlib.pyplot as plt
from preproses_signals.function.Multiscale_windowing import signal_window
from preproses_signals.function.function import data_df
from preproses_signals.show_plot import show_plotes
from preproses_signals.function import Multiscale_windowing as mw
from preproses_signals.function import function

## denozis ecg
wavelet='sym8'
def wavelet_denoising(ecg_signal, wavelet='sym8', level=5):
    """
    Perform wavelet-based denoising on an ECG signal.

    Parameters:
    - ecg_signal: numpy array, the raw ECG signal.
    - wavelet: str, the name of the wavelet to use (default: 'sym8').
    - level: int, the level of wavelet decomposition (default: 5).

    Returns:
    - denoised_signal: numpy array, the denoised ECG signal.
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    # Estimate noise standard deviation using the first detail coefficient
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Calculate universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(ecg_signal)))

    # Apply soft thresholding
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c
                       for i, c in enumerate(coeffs)]

    # Wavelet reconstruction
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)






    return denoised_signal


## R_R ecg

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===================== CHANGED: ave_signal (بدون بازگشت بی‌نهایت) =====================
import numpy as np
import matplotlib.pyplot as plt

def ave_signal(signal, Percentage=0.6, nim_Distance=100, window=100, step=1,
               number_max_signal=1, number_add=None, sampel_rate=256,
               show_plot=False, size_plot=(12, 3), max_tries=5):
    """
    همان خروجی قبلی را می‌دهد اما به جای recursion از یک حلقه با سقف تلاش استفاده می‌کند
    تا هرگز گیر نکند.
    """
    signal = np.asarray(signal, dtype=float)
    L = len(signal)
    if L < 2:
        averaged_signal = np.full(L, signal.mean() if L else 0.0, dtype=float)
        max_signal = averaged_signal.copy()
        r_peaks = np.array([], dtype=int)
        return averaged_signal.tolist(), max_signal.tolist(), r_peaks

    def _run_once(number_max_signal):
        # moving average با پنجره sliding (حفظ منطق قبلی)
        if window <= 0: 
            w = 1
        else:
            w = min(window, L)
        num_segments = max(0, (L - w) // max(1, step))
        averaged_signal = []
        for i in range(num_segments + 1):
            s = i * step
            e = s + w
            averaged_signal.extend([signal[s:e].mean()] * step)

        # بخش انتهایی (حفظ رفتار قبلی)
        tail = min(step * w * 2, max(1, L - len(averaged_signal)))
        for i in range(tail):
            e = i + 1
            s = max(0, L - (i + w))
            averaged_signal.append(signal[s:L - i].mean())
        while len(averaged_signal) < L:
            averaged_signal.append(signal[max(0, L - w):].mean())

        averaged_signal = averaged_signal[:L]
        averaged_signal[0] = averaged_signal[1] if L > 1 else averaged_signal[0]

        # آستانه
        if number_add is not None:
            add_ave = number_add
        else:
            if number_max_signal > 1:
                # top-k بیشینه‌ها
                idx_sorted = np.argsort(signal)[::-1][:number_max_signal]
                diffs = Percentage * (signal[idx_sorted] - np.take(averaged_signal, idx_sorted))
                add_ave = float(diffs.mean()) if diffs.size else 0.0
            else:
                mx = float(np.max(signal))
                mxi = int(np.argmax(signal))
                add_ave = Percentage * (mx - averaged_signal[mxi])

        max_signal = (np.asarray(averaged_signal) + add_ave).tolist()

        # R-peaks با منطق قبلی
        r_peaks = []
        for i in range(1, L - 1):
            if signal[i] > max_signal[i] and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                if not r_peaks or (i - r_peaks[-1]) > nim_Distance:
                    r_peaks.append(i)
        return averaged_signal, max_signal, np.asarray(r_peaks, dtype=int)

    tries = 0
    while True:
        averaged_signal, max_signal, r_peaks = _run_once(number_max_signal)
        enough = (L / sampel_rate) / 1.2 <= len(r_peaks)
        if enough or tries >= max_tries:
            break
        number_max_signal *= 5
        tries += 1

    if show_plot:
        time = np.linspace(0, L / sampel_rate, L)
        plt.figure(figsize=size_plot)
        plt.plot(time, signal, label='ECG')
        plt.plot(time, averaged_signal, label='ave ECG')
        plt.plot(time, max_signal, label=f'max ECG * {Percentage}')
        if len(r_peaks):
            plt.scatter(time[r_peaks], signal[r_peaks], color='red', label='Detected R-peaks')
        plt.legend(); plt.grid(True); plt.show()

    return averaged_signal, max_signal, r_peaks

# ===================== CHANGED: sheb_window (برداری، سریع) =====================
import numpy as np
def sheb_window(signal, point, window):
    """
    نسخه‌ی برداری‌شده (بدون حلقه‌های پایتونی بزرگ) و امن در لبه‌ها.
    """
    sig = np.asarray(signal, dtype=float)
    L = len(sig)
    win = max(1, int(window))
    l = max(0, point - win)
    r = min(L, point + win + 1)
    if r - l < 2:
        return 0.0, 0.0, 0.0

    d = np.diff(sig[l:r])  # طول r-l-1
    right_len = point - l
    left_len  = r - (point + 1)

    right = d[:right_len].sum() / win if right_len > 0 else 0.0
    left  = (-d[right_len:]).sum() / win if left_len > 0 else 0.0
    return right, left, right - left

# ===================== CHANGED: stpq_function (ایمن و بدون گیر) =====================
import time
import numpy as np
import matplotlib.pyplot as plt

def stpq_function(number_RR, r_peaks, denoised_ecg, ave_denoised_ecg,
                  plot_show=True, number_ofset=.10, range_window_t=.20,
                  range_start_t=.3, size_sheb_window=.2, number_to_ave_sheb=.05,
                  loop_timeout_sec=0.5):
    if number_RR + 1 >= len(r_peaks):
        return [0, 0, 0, 0]

    a = int(r_peaks[number_RR]); b = int(r_peaks[number_RR + 1])
    if b <= a + 2:
        return [a, a, a, a]

    y2 = ave_denoised_ecg[a:b]
    y3 = denoised_ecg[a:b]
    L = len(y3)

    # نقاط تغییر شیب
    sheb = []
    g1 = None
    for i in range(L - 1):
        g0 = 1 if y3[i] > y3[i + 1] else -1
        if g1 is None:
            g1 = g0
        if g0 != g1:
            sheb.append(i)
            g1 = g0

    if not sheb:
        return [a, a, a, a]

    s, q = sheb[0], sheb[-1]
    sorted_sheb_t_p, sorted_sheb_s, sorted_sheb_q = {}, {}, {}

    for i in sheb:
        number = y3[i] - y2[i]
        ave_sheb, right_sheb, left_sheb = sheb_window(y3, i, int(L * size_sheb_window))
        score = number_to_ave_sheb * number + (1 - number_to_ave_sheb) * ave_sheb
        if i >= (number_ofset * L) and i <= ((1 - number_ofset) * L):
            sorted_sheb_t_p[i] = (number, right_sheb, left_sheb, ave_sheb, score)
        if i <= (number_ofset * L):
            sorted_sheb_s[i] = (number, right_sheb, left_sheb, ave_sheb, score)
        if i >= ((1 - number_ofset) * L):
            sorted_sheb_q[i] = (number, right_sheb, left_sheb, ave_sheb, score)

    # مرتب‌سازی بر اساس score
    sorted_sheb_t_p = dict(sorted(sorted_sheb_t_p.items(), key=lambda x: x[1][4]))
    sorted_sheb_s   = dict(sorted(sorted_sheb_s.items(),   key=lambda x: x[1][4]))
    sorted_sheb_q   = dict(sorted(sorted_sheb_q.items(),   key=lambda x: x[1][4]))

    keys = list(sorted_sheb_t_p.keys())
    m = len(keys)
    if m == 0:
        return [a, a, a, a]
    p = keys[-1]
    t = keys[-2] if m >= 2 else p

    # انتخاب t با فاصله‌ی حداقلی و سقف زمان/گام
    min_gap = int(range_window_t * L)
    t0 = time.perf_counter()
    for k in range(m - 3, -1, -1):
        if time.perf_counter() - t0 > loop_timeout_sec:
            break
        cand = keys[k]
        if abs(p - cand) >= min_gap:
            t = cand
            break
    if abs(p - t) < min_gap:
        t = max(0, p - min_gap)

    if len(sorted_sheb_s) > 0:
        s = list(sorted_sheb_s.keys())[0]
    if len(sorted_sheb_q) > 0:
        q = list(sorted_sheb_q.keys())[0]

    if t > p:
        t, p = p, t

    if plot_show:
        x_range = np.arange(a, b)
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, y2, label='ave_long')
        plt.plot(x_range, y3, label='denoised_ecg')
        plt.scatter(np.array(sheb) + a, y3[sheb], label='sheb==0')
        plt.scatter([s + a], [y3[s]], label='S')
        plt.scatter([q + a], [y3[q]], label='Q')
        plt.scatter([t + a], [y3[t]], label='t')
        plt.scatter([p + a], [y3[p]], label='p')
        plt.title('R to R plot'); plt.legend(); plt.grid(True); plt.show()

    # انتقال به مختصات global
    return [s + a, t + a, p + a, q + a]


def stpq_function_all(r_peaks, denoised_ecg, ave_denoised_ecg, plot_all_show = True, stort_plot = 10, long_plot = 10):
    '''
    input :
        r_peaks = RR data
        denoised_ecg = signal ecg
        ave_denoised_ecg = ave_signal_ecg
        ave_denoised_ecg_window_short = ave_denoised_ecg_window

    output:
      data = [[R,s,t, p,q],
                ......    ]
    '''
    data_stpq = []
    for i in range(len(r_peaks)-1):
        stpq = stpq_function(i, r_peaks, denoised_ecg, ave_denoised_ecg, plot_show = False)
        data_stpq.append(stpq)

    data_stpq = np.array(data_stpq)
    R = r_peaks
    s = data_stpq[:,0]
    t = data_stpq[:,1]
    p = data_stpq[:,2]
    q = data_stpq[:,3]

    if plot_all_show :
        n = stort_plot
        m = long_plot
        x_range = range(len(denoised_ecg))
        plt.figure(figsize=( m* 5, 6))

        plt.plot(x_range[R[n] : R[n + m]],denoised_ecg[R[n] : R[n + m]] , label='denoised_ecg')

        #plt.scatter(x_sheb,y3[sheb], label='sheb == 0')

        plt.scatter(R[n:n + m],denoised_ecg[R[n:n + m]], label='R')
        plt.scatter(s[n:n + m],denoised_ecg[s[n:n + m]], label='S')
        plt.scatter(t[n:n + m],denoised_ecg[t[n:n + m]], label='t')
        plt.scatter(p[n:n + m],denoised_ecg[p[n:n + m]], label='p')
        plt.scatter(q[n:n + m],denoised_ecg[q[n:n + m]], label='q')
        plt.title('R to R plot')
        plt.legend()
        plt.grid(True)
        plt.show()
    data = np.zeros((len(s), 5))
    data[:,0] = R[:-1]
    data[:,1] = s
    data[:,2] = t
    data[:,3] = p
    data[:,4] = q
    return data



def RR(R_list, list_label ):
    HRV_list = []
    for i in range(len(R_list)-1):
        HRV_list.append((R_list[i+1] - R_list[i]) )
    list_out = []
    for i in list_label:
      new_signal = []
      for i1 in R_list[:-1]:
        new_signal.append(i[i1])
      list_out.append(new_signal)
    return HRV_list, list_out

# ===================== CHANGED: fix_len_for_hrv (سریع با np.interp) =====================
import numpy as np
def fix_len_for_hrv(seconds, list_data, len_data, N=0.0):
    """
    درون‌یابی برداری به طول len_data. seconds و list_data باید هم‌طول باشند.
    """
    seconds = np.asarray(seconds, dtype=float)
    values  = np.asarray(list_data, dtype=float)
    if seconds.size == 0:
        return [float(N)] * int(len_data)
    x = np.arange(int(len_data), dtype=float)
    y = np.interp(x, seconds, values, left=values[0], right=values[-1])
    return y.tolist()


'''
#### time
- SDNN (انحراف معیار فاصله‌های RR)
- RMSSD (ریشه میانگین مربع تفاوت‌های فاصله‌های RR)
- PNN50 (درصد تفاوت‌های RR که از یک آستانه خاص عبور می‌کنند)

'''


import numpy as np

def windowed_sdnn(rr_intervals, window_size):
    """
    محاسبه SDNN به‌صورت پنجره‌ای (Sliding Window).

    :param rr_intervals: آرایه‌ای از فاصله‌های RR
    :param window_size: اندازه پنجره (تعداد نمونه‌ها)
    :return: آرایه‌ای از SDNN برای هر پنجره
    """
    # بررسی اندازه داده‌ها
    if len(rr_intervals) < window_size:
        raise ValueError("اندازه داده‌ها باید بزرگتر از اندازه پنجره باشد.")

    # آرایه برای ذخیره SDNN ها
    sdnn_values = []

    try :
      a = len(rr_intervals[1])
      windows = rr_intervals
    except:
      #print('create_window')
      windows = signal_window(rr_intervals, window_size)
    for window in windows:
      # محاسبه انحراف معیار (SDNN) برای پنجره
      sdnn = np.std(window)

      # افزودن SDNN به لیست نتایج
      sdnn_values.append(sdnn)





    return np.array(sdnn_values)

import numpy as np

def windowed_rmssd(rr_intervals, window_size):
    """
    محاسبه RMSSD به‌صورت پنجره‌ای (Sliding Window).

    :param rr_intervals: آرایه‌ای از فاصله‌های RR
    :param window_size: اندازه پنجره (تعداد نمونه‌ها)
    :return: آرایه‌ای از RMSSD برای هر پنجره
    """
    # بررسی اندازه داده‌ها
    if len(rr_intervals) < window_size:
        raise ValueError("اندازه داده‌ها باید بزرگتر از اندازه پنجره باشد.")

    # آرایه برای ذخیره RMSSD ها
    rmssd_values = []

    try :
      a = len(rr_intervals[1])
      windows = rr_intervals
    except:
      #print('create_window')

      windows = signal_window(rr_intervals, window_size)
    for window in windows:
      if len(window) < 2:
        rmssd_values.append(0)
        continue
      # محاسبه تفاوت‌های بین فاصله‌های RR
      rr_diff = np.diff(window)

      # محاسبه RMSSD برای پنجره
      rmssd = np.sqrt(np.mean(rr_diff**2))

      # افزودن RMSSD به لیست نتایج
      rmssd_values.append(rmssd)

    return np.array(rmssd_values)


import numpy as np

def windowed_pnn50(rr_intervals, window_size, threshold=50):
    """
    محاسبه PNN50 به‌صورت پنجره‌ای (Sliding Window).

    :param rr_intervals: آرایه‌ای از فاصله‌های RR
    :param window_size: اندازه پنجره (تعداد نمونه‌ها)
    :param threshold: آستانه برای عبور تفاوت‌ها (پیش‌فرض 50 میلی‌ثانیه)
    :return: آرایه‌ای از PNN50 برای هر پنجره
    """
    # بررسی اندازه داده‌ها
    if len(rr_intervals) < window_size:
        raise ValueError("اندازه داده‌ها باید بزرگتر از اندازه پنجره باشد.")

    # آرایه برای ذخیره PNN50 ها
    pnn50_values = []
    try :
      a = len(rr_intervals[1])
      windows = rr_intervals
    except:
      #print('create_window')

      windows = signal_window(rr_intervals, window_size)
    for window in windows:
      if len(window) < 2:
        pnn50_values.append(0)
        continue

      # محاسبه تفاوت‌های میان فاصله‌های RR در پنجره
      rr_diff = np.diff(window)

      # شمارش تفاوت‌هایی که از آستانه عبور می‌کنند
      count_above_threshold = np.sum(np.abs(rr_diff) > threshold)

      # محاسبه PNN50 برای پنجره
      pnn50_value = (count_above_threshold / len(rr_diff)) * 100

      # افزودن PNN50 به لیست نتایج
      pnn50_values.append(pnn50_value)

    return np.array(pnn50_values)


#### frequency


import numpy as np
from scipy.signal import welch
# ===================== CHANGED: windowed_lf_hf_ratio (کم‌هزینه‌تر) =====================
import numpy as np
from scipy.signal import welch

def windowed_lf_hf_ratio(rr_intervals, window_size, fs=1.0, nperseg=256):
    """
    Welch با nperseg کنترل‌شده. روی پنجره‌های کوتاه‌تر سریع‌تر است.
    """
    if len(rr_intervals) < window_size:
        raise ValueError("اندازه داده‌ها باید بزرگتر از اندازه پنجره باشد.")
    lf_hf_ratios, hf_ratios, lf_ratios = [], [], []
    # اگر ورودی از پیش پنجره‌بندی شده بود:
    try:
        _ = len(rr_intervals[1])
        windows = rr_intervals
    except Exception:
        from preproses_signals.function.Multiscale_windowing import signal_window
        windows = signal_window(rr_intervals, window_size)

    for w in windows:
        w = np.asarray(w, dtype=float)
        if w.size < 4:
            lf_hf_ratios.append(0.0); hf_ratios.append(0.0); lf_ratios.append(0.0)
            continue
        f, Pxx = welch(w, fs=fs, nperseg=min(nperseg, w.size))
        mask_lf = (f >= 0.04) & (f <= 0.15)
        mask_hf = (f >= 0.15) & (f <= 0.40)
        lf = float(Pxx[mask_lf].sum())
        hf = float(Pxx[mask_hf].sum())
        ratio = (lf / hf) if hf > 0 else 0.0
        lf_hf_ratios.append(ratio); hf_ratios.append(hf); lf_ratios.append(lf)
    return lf_hf_ratios, hf_ratios, lf_ratios

### S_Q_R_T

from collections import defaultdict

def S_Q_R_T(data, ecg):
    # فرض می‌کنیم data آرایه‌ی N×5 است با ستون‌های [R, s, t, p, q]
    R, s, t, p, q = data.T

    # defaultdict به‌صورت خودکار برای هر کلید جدید یک لیست خالی می‌سازد
    signalles: dict[str, list] = defaultdict(list)
    list_s_t =[]
    for i in range(len(R) - 1):
        # ۱) فاصلهٔ S[i+1] تا Q[i]
        signalles['SQ'].append(s[i + 1] - q[i])

        # ۲) فاصلهٔ T[i+1] تا Q[i]
        signalles['TQ'].append(t[i + 1] - q[i])

        # ۳) تغییرات زاویه‌ی S نسبت به R
        signalles['delta_s_R'].append((s[i + 1] - R[i + 1]) - (s[i] - R[i]))

        # ۴) تغییرات زاویه‌ی T نسبت به R
        signalles['delta_t_R'].append((t[i + 1] - R[i + 1]) - (t[i] - R[i]))

        # ۵) تغییرات زاویه‌ی Q نسبت به R
        signalles['delta_q_R'].append((q[i + 1] - R[i + 1]) - (q[i] - R[i]))

        # ۶) برش سیگنال ECG بین s[i] تا t[i]
        # دقت کنید که مقادیر s[i] و t[i] را به int تبدیل می‌کنیم
        signal_i = ecg[int(s[i]) : int(t[i])]
        list_s_t.append(signal_i)

    # اگر لازم است خروجی به دیکشنری معمولی تبدیل شود:
    return dict(signalles), list_s_t
# ===================== CHANGED: futer_s_t_singal (ایمن روی آرایه‌ی خالی) =====================
import numpy as np
def futer_s_t_singal(signal):
    s = np.asarray(signal, dtype=float).ravel()
    if s.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return float(np.var(s)), float(np.mean(s)), float(np.sum(s)), float(np.min(s)), float(np.max(s))

def abs_df_futer_s_t_singal(signal):
  list_df_signal = data_df(signal)
  ar_df_signal = np.array(list_df_signal)
  abs_ar_df_signal = np.abs(ar_df_signal)
  return futer_s_t_singal(abs_ar_df_signal)

def futer_s_t(
    cut_signal,
    label = '(s_t)'

  ):
  list_var, list_mean, list_sum , list_min, list_max = [],[],[],[],[]
  list_df_var, list_df_mean, list_df_sum , list_df_min, list_df_max = [],[],[],[],[]
  for signal in cut_signal:
    var, mean, sum , min, max = futer_s_t_singal(signal)
    list_var.append(var)
    list_mean.append(mean)
    list_sum.append(sum)
    list_min.append(min)
    list_max.append(max)
    df_var, df_mean, df_sum , df_min, df_max = abs_df_futer_s_t_singal(signal)
    list_df_var.append(df_var)
    list_df_mean.append(df_mean)
    list_df_sum.append(df_sum)
    list_df_min.append(df_min)
    list_df_max.append(df_max)


  dic_dataset = {
                  'list_var' + label :    list_var,
                  'list_mean'+ label:    list_mean,
                  'list_sum' + label:   list_sum,
                  'list_min' + label:   list_min,
                  'list_max' + label:   list_max,

                  'list_df_var' + label:    list_df_var,
                  'list_df_mean'+ label:    list_df_mean,
                  'list_df_sum' + label:   list_df_sum,
                  'list_df_min' + label:   list_df_min,
                  'list_df_max'+ label :   list_df_max,

                 }
  return dic_dataset


# ===================== CHANGED: number_df_signalles (ایمن‌سازی برش‌ها) =====================
from collections import defaultdict
import numpy as np

def number_df(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2: return 0
    d = np.diff(x)
    return int(np.sum(d[:-1] * d[1:] < 0))

def number_df_signalles(signal, data, window):
    R = data[:, 0]; s = data[:, 1]; t = data[:, 2]; p = data[:, 3]; q = data[:, 4]
    L = len(signal); w2 = int(window) // 2
    out = defaultdict(list)

    def _safe_slice(center):
        a = max(0, int(center) - w2)
        b = min(L, int(center) + w2)
        if b <= a: 
            return np.empty((0,), dtype=float)
        return signal[a:b]

    for i in range(len(R)):
        out['number_df_R'].append(number_df(_safe_slice(R[i])))
        out['number_df_s'].append(number_df(_safe_slice(s[i])))
        out['number_df_t'].append(number_df(_safe_slice(t[i])))
        out['number_df_p'].append(number_df(_safe_slice(p[i])))
        out['number_df_q'].append(number_df(_safe_slice(q[i])))
    return dict(out)



## final function
from typing import Any
def feature_ecg(
    ecg,
    #df,
    window_size = 1000,
    show = False,
    fix_len = True
    ):
  len_data = len(ecg)
  features:  dict[str, Any] = {}
  # denise
  ecg_denoise = wavelet_denoising(ecg)
  ecg_denoise = ecg_denoise[:len_data]
  features['ecg_denoise'] = ecg_denoise
  # range
  ecg_range = mw.range_window(ecg_denoise, window_size)
  features['ecg_range'] = ecg_range
  # ave
  ecg_ave = mw.ave_window(ecg_denoise, window_size)
  features['ecg_ave'] = ecg_ave
  # variance
  ecg_variance = mw.variance_window(ecg_denoise, window_size)
  features['ecg_variance'] = ecg_variance
  # max
  ecg_max = mw.max_window(ecg_denoise, window_size)
  features['ecg_max'] = ecg_max
  # min
  ecg_min = mw.min_window(ecg_denoise, window_size)
  features['ecg_min'] = ecg_min


  # R_R
  ave_ecg, max_ecg, r_peaks = ave_signal(ecg_denoise, Percentage = 0.4 ,show_plot = show, size_plot = (120 , 3))

  ave_denoised_ecg = mw.ave_window(ecg_denoise,200)
  data = stpq_function_all(r_peaks, ecg_denoise, ave_denoised_ecg, plot_all_show = show, stort_plot=0, long_plot=100)

  # R - R
  RR_list, list_label = RR(data[:,0].astype(int), [list(range(len(ecg))),  ecg])
  Seconds_RR = list_label[0]

  features['Ecg_RR'] = list_label[1]


  var_Ecg_RR = mw.variance_window(list_label[1], 10)
  features['var_Ecg_RR'] = var_Ecg_RR


  # HRV
  ## time

  HRV_SDNN = windowed_sdnn(RR_list, window_size = 10)
  features['HRV_SDNN'] = HRV_SDNN


  HRV_RMSSD = windowed_rmssd(RR_list, window_size = 10)
  features['HRV_RMSSD'] = HRV_RMSSD


  HRV_PNN50 = windowed_pnn50(RR_list, window_size = 20, threshold=10)
  features['HRV_PNN50'] = HRV_PNN50


  HRV_LF_HF,HRV_HF, HRV_LF = windowed_lf_hf_ratio(RR_list, window_size = 20, fs=1.0)
  features['HRV_LF_HF'] = HRV_LF_HF
  features['HRV_HF'] = HRV_HF
  features['HRV_LF'] = HRV_LF


  ## S Q R T
  dic_sqrt, list_s_t = S_Q_R_T(data, ecg_denoise)
  features.update(dic_sqrt)

  dic_var_sqrt = mw.dic_variance_window(dic_sqrt, 10)
  features.update(dic_var_sqrt)

  show_df_pec_sqrt = function.dic_show_df_pec(dic_sqrt, 0.4)
  features.update(show_df_pec_sqrt)
  ## [s, t]
  dic_s_t = futer_s_t(list_s_t)
  features.update(dic_s_t)

  dic_number_df = number_df_signalles( ecg_denoise, data,  100)
  features.update(dic_number_df)

  dic_var_number_df = mw.dic_variance_window(dic_number_df, 10)
  features.update(dic_var_number_df)

  if fix_len:
    ##print('in fix len functiones')
    # Create a new dictionary to store the fixed-length features
    fixed_features = {}
    for key, value in features.items():
      if len(value) != len_data:
        ##print(key, len(value), len_data)
        fixed_features[key + '_fix_len'] = fix_len_for_hrv(Seconds_RR, value, len_data)
      else:
        fixed_features[key] = value
    # Update the original features dictionary with the new features
    #features.update(fixed_features)

  if show:
    show_plotes(fixed_features) # type: ignore
  return fixed_features # type: ignore




