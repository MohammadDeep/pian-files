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


def ave_signal(signal,Percentage = 0.6 , nim_Distance = 100,  window = 100 , step = 1,number_max_signal = 1, number_add = None, sampel_rate = 256,
               show_plot = False, size_plot = (12, 3),number_try = 5, n_i = 0):
    '''
    توضیحات پارامترها
        signal:
        آرایه یا لیستی از مقادیر سیگنال که قرار است پردازش شود.

        Percentage (پیش‌فرض 0.6):
        ضریب برای محاسبه مقدار افزایشی آستانه. این مقدار تفاوت بین بیشینه سیگنال و میانگین محلی را تعدیل می‌کند.

        nim_Distance (پیش‌فرض 100):
        حداقل فاصله (به تعداد نمونه) بین دو نقطه اوج متوالی. به‌منظور جلوگیری از شناسایی اوج‌های نزدیک به یکدیگر استفاده می‌شود.

        window (پیش‌فرض 100):
        طول پنجره استفاده‌شده در محاسبه میانگین متحرک.

        step (پیش‌فرض 1):
        تعداد نمونه‌هایی که پنجره به جلو حرکت می‌کند (گام حرکت) در محاسبه میانگین متحرک.

        number_max_signal (پیش‌فرض 1):
        تعداد مقادیر بیشینه سیگنال که در محاسبه مقدار افزایشی آستانه در نظر گرفته می‌شود. در صورتی که مقدار بیشتری نیاز باشد، این پارامتر می‌تواند تغییر یابد.

        number_add (پیش‌فرض None):
        در صورتی که مقدار مشخصی ارائه شود، به عنوان مقدار افزایشی آستانه استفاده می‌شود و محاسبات مربوط به number_max_signal و Percentage نادیده گرفته می‌شود.

        sampel_rate (پیش‌فرض 256):
        نرخ نمونه‌برداری سیگنال (به هرتز). این مقدار برای تعیین مدت زمان سیگنال و تعداد اوج‌های مورد انتظار استفاده می‌شود.

        مقادیر خروجی
        averaged_signal:
        آرایه‌ای که شامل مقادیر میانگین متحرک سیگنال بر اساس پنجره و گام داده شده است.

        max_signal:
        آرایه‌ای شامل آستانه‌های محاسبه‌شده که برابر است با averaged_signal به علاوه مقدار افزایشی آستانه.

        r_peaks:
        آرایه‌ای (معمولاً از نوع numpy.array) از اندیس‌های نقاطی که به عنوان اوج (یا نقاط R) در سیگنال تشخیص داده شده‌اند.
    '''
    num_segments = (len(signal) - window) // step  # Number of segments

    averaged_signal = []  # Stores moving average values

    # Sliding window approach
    for i in range(num_segments + 1):

        start_idx = i * step
        end_idx = start_idx + window
        avg_value = np.mean(signal[start_idx:end_idx])  # Compute moving average

        # Store repeated average for plotting
        averaged_signal.extend([avg_value] * step)

    for i in range(step* window * 2):


        averaged_signal[-i] = np.mean(signal[-(i+window):-i])
    for i in range(window -1):
        averaged_signal.append(np.mean(signal[-(i+window):-i]))
    if number_add:
        add_ave = number_add
    else:
        averaged_signal[0] = averaged_signal[1]
        if number_max_signal > 1:
            add_ave_list = []
            sort_signal = sorted(signal, reverse=True)[:number_max_signal]
            for i in sort_signal:
                max_indes = [i1 for i1, num in enumerate(signal) if num == i]
                add_ave_list.append( Percentage * (i - averaged_signal[max_indes[0]]))
            add_ave = sum(add_ave_list)/len(add_ave_list)
        else:
            # Use np.max instead of max to avoid potential conflicts with variables
            max_value = np.max(signal)
            max_indes = [i1 for i1, num in enumerate(signal) if num == max_value]
            averaged_signal[0] = averaged_signal[1]

            add_ave = Percentage * (max_value - averaged_signal[max_indes[0]])

    max_signal = []
    for i in averaged_signal:
        max_signal.append(i + add_ave)
    r_peaks = []
    for i in range(len(max_signal)):
        if signal[i] > max_signal[i] :
            if i != 0 and i!= (len(max_signal)-1) and signal[i] > signal [i+1] and signal[i] > signal[i-1]:

                if len(r_peaks) == 0 :
                    r_peaks.append(i)
                elif nim_Distance < ( i - r_peaks[-1]):
                    r_peaks.append(i)
    
    if (len(signal)/sampel_rate)/1.2 > len(r_peaks) :
            n_i += 1
            #print(f'try to fin r in ecg try : {n_i}')
            if n_i ==number_try:
               print('can fine R in ecg in function ave_ecg') 
               raise ValueError('in file ECG.py in function ave_ecg')
            number_max_signal  = number_max_signal  * 10
            averaged_signal, max_signal, r_peaks = ave_signal(signal,Percentage , nim_Distance ,  window  , step ,number_max_signal = number_max_signal, number_add = number_add, sampel_rate = sampel_rate,n_i = n_i)

    r_peaks = np.array(r_peaks)
    if show_plot:
        time = np.linspace(0, len(signal) / sampel_rate, len(signal))
        plt.figure(figsize = size_plot)
        plt.plot(time, signal, label='ECG')
        plt.plot(time, averaged_signal, label='ave ECG')
        plt.plot(time, max_signal, label=f'max ECG * {Percentage}')
        plt.scatter(time[r_peaks], signal[r_peaks], color='red', label='Detected R-peaks')
        plt.show()
    return averaged_signal, max_signal, r_peaks


## sheb_window
def sheb_window(signal , point , window):
    start_rit = point - window
    if start_rit < 0:
        start_rit = 0
    end_rit = point
    start_lef = point
    end_lef = point + window
    if end_lef > len(signal)-1:
        end_lef = len(signal)-1
    sheb_right = 0
    sheb_left = 0
    for i in range(start_rit, end_rit):
        sheb_right = (signal[i+1] - signal[i])/ window  + sheb_right
    for i in range(start_lef, end_lef):
        sheb_left = (signal[i] - signal[i-1])/ window  + sheb_left
    return sheb_right, sheb_left, sheb_right - sheb_left


def stpq_function(number_RR, r_peaks, denoised_ecg, ave_denoised_ecg,plot_show = True, number_ofset = .10, range_window_t = .20,range_start_t = .3, size_sheb_window = .2,number_to_ave_sheb = .05 ):

  
    y2 = ave_denoised_ecg[r_peaks[number_RR]: r_peaks[number_RR + 1] ]
    y3 = denoised_ecg[r_peaks[number_RR]: r_peaks[number_RR + 1] ]
    x_range = range(r_peaks[number_RR], r_peaks[number_RR + 1] )


    g0, g1 = 0,0
    sheb = []
    x_sheb = []
    for i in range(len(y3)):
        if  i == len(y3)- 1:
            break

        if y3[i] > y3[i + 1]:
            g0  = 1
        else:
            g0 = -1
        if i == 0:
            g1 = g0
        if g0 != g1:
            sheb.append(i)
            x_sheb.append(i + r_peaks[number_RR])
            g1 = g0
    ##print(sheb[0:-1])

    t,p = 0,0
    ##print(t, p)
    s,q = sheb[0], sheb[-1]
    ##print(s,q)

    sorted_sheb_t_p = {}
    sorted_sheb_s = {}
    sorted_sheb_q = {}

    for i in sheb:
        number = y3[i] - y2[i]
        ave_sheb, right_sheb, left_sheb = sheb_window(y3 , i, int(len(y3) * size_sheb_window))
        if i >= (number_ofset * len(y3))  and i <= ((1-number_ofset) * len(y3) ):
            sorted_sheb_t_p[i] = number, right_sheb, left_sheb, ave_sheb, number_to_ave_sheb* number + (1 - number_to_ave_sheb) * ave_sheb
        if i <= (number_ofset * len(y3)):
            sorted_sheb_s[i] = number, right_sheb, left_sheb, ave_sheb,number_to_ave_sheb* number + (1 - number_to_ave_sheb) * ave_sheb
        if i >= ((1-number_ofset) * len(y3)):
            sorted_sheb_q[i] = number, right_sheb, left_sheb, ave_sheb,number_to_ave_sheb* number + (1 - number_to_ave_sheb) * ave_sheb



    sorted_sheb_t_p = dict(sorted(sorted_sheb_t_p.items(), key=lambda item: item[1][4]))
    sorted_sheb_s = dict(sorted(sorted_sheb_s.items(), key=lambda item: item[1][4]))
    sorted_sheb_q = dict(sorted(sorted_sheb_q.items(), key=lambda item: item[1][4]))
    p = list(sorted_sheb_t_p.keys())[-1]
    t = list(sorted_sheb_t_p.keys())[-2]
   # #print(t, p)
    ii = 1
    import time

    timeout_sec = 2  * 60
    t0 = time.perf_counter()
    while abs(p-t) < range_window_t * len(y3):
        
        if time.perf_counter() - t0 > timeout_sec:
            raise TimeoutError(f"while took more than {timeout_sec} s")
        index = -2 - ii
        t = list(sorted_sheb_t_p.keys())[index]
        ii = + ii + 1
    if len(sorted_sheb_s) > 0:
        s = list(sorted_sheb_s.keys())[0]
    if len(sorted_sheb_q) > 0:
        q = list(sorted_sheb_q.keys())[0]
    '''#print(sorted_sheb_t_p)
    #print(sorted_sheb_s)
    #print(sorted_sheb_q)'''



    if t> p:
        number = t
        t = p
        p = number



    if plot_show :
        #print(sheb[0:-1])
        #print(p,t)
        plt.figure(figsize=(10, 6))

        plt.plot(x_range, y2, label='ave_long')
        plt.plot(x_range, y3, label='denoised_ecg')

        plt.scatter(x_sheb,y3[sheb], label='sheb == 0')
        plt.scatter([s+ r_peaks[number_RR]],y3[[s]], label='S')
        plt.scatter([q+ r_peaks[number_RR]],y3[[q]], label='Q')
        plt.scatter([t + r_peaks[number_RR]],y3[t], label='t')
        plt.scatter([p + r_peaks[number_RR]],y3[p], label='p')
        plt.scatter([int(number_ofset * len(y3)) + r_peaks[number_RR],
                     int((1- number_ofset) * len(y3)) +  r_peaks[number_RR]],
                    y3[[int(number_ofset * len(y3)),int((1- number_ofset) * len(y3))] ], label='ofset')
        plt.title('R to R plot')
        plt.legend()
        plt.grid(True)
        plt.show()
    data_list = [s, t, p, q]
    for i in range(len(data_list)):
        data_list[i] = data_list[i] + r_peaks[number_RR]
    return data_list



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


def fix_len_for_hrv(seconds, list_data, len_data, N = 0):
  new_signall = []
  for i in range(len_data):
    new_signall.append(N)
  mean_data = sum(list_data)/len(list_data)
  for i in range(seconds[0]):
    new_signall[i]= mean_data
  for i in range(seconds[-1], len(new_signall)):
    new_signall[i]= mean_data
  for i in range(len(seconds) -1):
    d = list_data[i + 1] -  list_data[i]
    sh = d / (seconds[i + 1] - seconds[i])
    for i1 in range(int(seconds[i]), int(seconds[i +1])):
      new_signall[i1] = list_data[i] + sh * (i1 - seconds[i])

  return new_signall


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

def windowed_lf_hf_ratio(rr_intervals, window_size, fs=1.0):
    """
    محاسبه ویژگی‌های LF, HF و LF/HF Ratio به‌صورت پنجره‌ای (Sliding Window).

    :param rr_intervals: آرایه‌ای از فاصله‌های RR
    :param window_size: اندازه پنجره (تعداد نمونه‌ها)
    :param fs: نرخ نمونه‌برداری (پیش‌فرض 1 هرتز)
    :return: آرایه‌ای از LF, HF و LF/HF Ratio برای هر پنجره
    """
    # بررسی اندازه داده‌ها
    if len(rr_intervals) < window_size:
        raise ValueError("اندازه داده‌ها باید بزرگتر از اندازه پنجره باشد.")

    # آرایه برای ذخیره نتایج
    lf_hf_ratios = []
    hf_ratios = []
    lf_ratios = []
    try :
      a = len(rr_intervals[1])
      windows = rr_intervals
    except:
      #print('create_window')

      windows = signal_window(rr_intervals, window_size)
    for window in windows:

      # محاسبه طیف قدرت با استفاده از Welch
      f, Pxx = welch(window, fs)

      # استخراج قدرت برای باند LF و HF
      lf_band = np.sum(Pxx[(f >= 0.04) & (f <= 0.15)])  # LF range: 0.04-0.15 Hz
      hf_band = np.sum(Pxx[(f >= 0.15) & (f <= 0.40)])  # HF range: 0.15-0.40 Hz

      # محاسبه نسبت LF/HF
      if hf_band != 0:  # جلوگیری از تقسیم بر صفر
          lf_hf_ratio = lf_band / hf_band
      else:
          #lf_hf_ratio = np.nan  # در صورتی که hf_band صفر باشد
          lf_hf_ratio = 0
      # افزودن نتایج به لیست
      lf_hf_ratios.append(lf_hf_ratio)
      hf_ratios.append(hf_band)
      lf_ratios.append(lf_band)
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
def futer_s_t_singal(signal):
  var  = np.var(signal)
  mean = np.mean(signal)
  sum  = np.sum(signal)
  min  = np.min(signal)
  max  = np.max(signal)
  return var, mean, sum , min, max
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



def number_df(signal):
  D = []
  number = 0
  for i in range(len(signal) - 1):
    D.append(signal[i + 1] - signal[i])
  for i in range(len(D) - 1):
    if D[i] * D [i + 1] < 0:
      number = number + 1
  return number
def number_df_signalles(signal, data, window):
  R = data[:, 0]
  s = data[:, 1]
  t = data[:, 2]
  p = data[:, 3]
  q = data[:, 4]
  signalles: dict[str, list] = defaultdict(list)
  for i in range(len(R)):
    signalles['number_df_R'].append(number_df(signal[int(R[i] - window //2): int(R[i] + window // 2)]))
    signalles['number_df_s'].append(number_df(signal[int(s[i] - window //2): int(s[i] + window // 2)]))
    signalles['number_df_t'].append(number_df(signal[int(t[i] - window //2): int(t[i] + window // 2)]))
    signalles['number_df_p'].append(number_df(signal[int(p[i] - window //2): int(p[i] + window // 2)]))
    signalles['number_df_q'].append(number_df(signal[int(q[i] - window //2): int(q[i] + window // 2)]))
  return dict(signalles)





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
  #features['ecg_range'] = ecg_range
  # ave
  ecg_ave = mw.ave_window(ecg_denoise, window_size)
  #features['ecg_ave'] = ecg_ave
  # variance
  ecg_variance = mw.variance_window(ecg_denoise, window_size)
  features['ecg_variance'] = ecg_variance
  # max
  ecg_max = mw.max_window(ecg_denoise, window_size)
  #features['ecg_max'] = ecg_max
  # min
  ecg_min = mw.min_window(ecg_denoise, window_size)
  #features['ecg_min'] = ecg_min


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
  #features.update(dic_sqrt)

  dic_var_sqrt = mw.dic_variance_window(dic_sqrt, 10)
  features.update(dic_var_sqrt)

  show_df_pec_sqrt = function.dic_show_df_pec(dic_sqrt, 0.4)
  features.update(show_df_pec_sqrt)
  ## [s, t]
  dic_s_t = futer_s_t(list_s_t)
  #features.update(dic_s_t)

  dic_number_df = number_df_signalles( ecg_denoise, data,  100)
  #features.update(dic_number_df)

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




