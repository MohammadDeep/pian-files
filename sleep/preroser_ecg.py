import os, glob
import numpy as np

import os, glob
import numpy as np
from numpy.lib.format import open_memmap   # برای ساخت .npy ممری‌مپ‌شده
from tqdm import tqdm  
import os, glob
import numpy as np
from numpy.lib.format import open_memmap
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np
import signal





# ---------- helpers ----------
def ensure_KT(a, T):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim == 2 and a.shape[-1] == T:
        return a.astype(np.float32, copy=False)
    if a.ndim == 2 and a.shape[0] == T:
        return a.T.astype(np.float32, copy=False)
    raise ValueError(f"Unexpected shape from feature fn: {a.shape}, expected (K,T) or (T,K) with T={T}")














src_train = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/train"
src_val   = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/val"

dst_train = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1_feat/train"
dst_val   = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1_feat/val"

SELECTED_CH = [7]  # ECG
from preproses_signals.ECG import feature_ecg
import numpy as np

def dict_to_KT(feat_dict: dict, T: int):
    """
    feat_dict: دیکشنریِ ویژگی‌ها؛ هر مقدار می‌تواند یکی از شکل‌های (T,), (K,T), (T,K) باشد.
    T: طول زمانی هدف (باید ثابت بماند)

    خروجی:
      X: np.ndarray با شکل (K_total, T) و dtype=float32
      keys_sorted: ترتیب کلیدها (برای ردیابی)
    """
    # اگر ensure_KT را از قبل داری، از همان استفاده کن؛ فرض می‌کنیم در اسکُوپ فعلی هست.
    keys_sorted = sorted(feat_dict.keys())   # ترتیبِ ثابت و قابل تکرار
    mats = []
    for k in keys_sorted:
        a = feat_dict[k]
        if len(a)+1 == T:
            a.append(a[-1])
        a_KT = ensure_KT(a, T)               # (Kc, T) می‌سازد یا خطا می‌دهد
        if a_KT.shape[1] != T:
            raise ValueError(f"{k}: expected time length T={T}, got {a_KT.shape[1]}")
        mats.append(a_KT.astype(np.float32, copy=False))
    X = np.vstack(mats)                       # (K_total, T)
    return X, keys_sorted


def feat_ecg (ecg , T = 32 * 256):
    feat_dict = feature_ecg(ecg)
    x_feat, keys_sorted = dict_to_KT(feat_dict, T)

    return x_feat

import os, glob
def precompute_features_with_fn(src_train
                                , dst_train
                                , SELECTED_CH
                                , feat_ecg):


    xs = sorted(glob.glob(os.path.join(src_train, "X_*.npy")))
    ys = sorted(glob.glob(os.path.join(src_train, "y_*.npy")))

    # فقط فایل‌ها (نه پوشه‌ها)
    xs = [p for p in xs if os.path.isfile(p)]
    ys = [p for p in ys if os.path.isfile(p)]

    for x_path, y_path in zip(xs, ys):
        X = np.load(x_path, mmap_mode='r')   # OK: فایل npy
        y = np.load(y_path, mmap_mode='r')
        print(f"[READ] {os.path.basename(x_path)} shape={X.shape} dtype={X.dtype}")
        print(f"[READ] {os.path.basename(y_path)} shape={y.shape} dtype={y.dtype}")
        X_ch = X[:,SELECTED_CH,:]
        print(f"[READ] {os.path.basename(x_path)} shape={X_ch.shape} dtype={X_ch.dtype}")
precompute_features_with_fn(src_train, dst_train, SELECTED_CH, feat_ecg)
precompute_features_with_fn(src_val,   dst_val,   SELECTED_CH, feat_ecg)
