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

# --- تایم‌اوت سطح پردازش (Unix only) ---
class FnTimeoutError(RuntimeError): pass

def _raise_timeout(signum, frame):
    raise FnTimeoutError("function timed out")

def call_with_timeout(seconds, fn, *args, **kwargs):
    """fn را با تایم‌اوت اجرا می‌کند؛ در صورت تاخیر بیش از حد، FnTimeoutError می‌اندازد."""
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))  # دقت زیرثانیه
    try:
        return fn(*args, **kwargs)
    finally:
        # تایمر و هندلر را حتماً برگردان
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

time = 60 * 3



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

def _writable_1d(x):
    # از ممری‌مپ جدا، writeable و contiguous
    return np.ascontiguousarray(np.array(x, dtype=np.float32, copy=True))

def _discover_K_T_first(xs, sel, ecg_fn):
    """از اولین فایل T و K را کشف می‌کند و تعداد ویژگی‌های هر کانال انتخابی را برمی‌گرداند."""
    X0 = np.load(xs[0], mmap_mode='r')   # (N, C, T)
    _, _, T = X0.shape
    k_per_ch = []
    for c in sel:
        ecg0 = _writable_1d(X0[0, c, :])
        try:
            out = call_with_timeout(time, ecg_fn, ecg0)
            A = ensure_KT(ecg_fn(ecg0), T)   # (Kc, T)
        except:
            print('data0')
            A = ensure_KT(np.zeros((49,256 *32), dtype=np.float32), T)   # (Kc, T)
        k_per_ch.append(int(A.shape[0]))
    K = int(sum(k_per_ch))
    # offsetهای نوشتن برای هر کانال (برای پر کردن بافر بدون vstack)
    starts = np.cumsum([0] + k_per_ch[:-1])
    ends   = starts + np.array(k_per_ch, dtype=int)
    return T, K, starts, ends

def _process_one_shard(args):
    """Worker برای هر شارد (فرایند جدا)."""
    x_path, y_path, dst_dir, sel, T, K, starts, ends, ecg_fn = args

    X = np.load(x_path, mmap_mode='r')              # (N, C, T) read-only
    y = np.load(y_path, mmap_mode='r')              # (N,)
    N = X.shape[0]

    out_path = os.path.join(dst_dir, os.path.basename(x_path).replace("X_", "Xfeat_"))
    Y_out    = os.path.join(dst_dir, os.path.basename(y_path))

    Xout = open_memmap(out_path, mode='w+', dtype=np.float32, shape=(N, K, T))

    # یک بافر موقت که هر بار پر می‌کنیم؛ از تخصیص‌های مکرر جلوگیری می‌کند
    tmp = np.empty((K, T), dtype=np.float32)

    for i in tqdm(range(N)):
        # پر کردن tmp با استفاده از offsetهای از پیش محاسبه‌شده
        for j, c in enumerate(sel):
            ecg = _writable_1d(X[i, c, :])
            try:
                out = call_with_timeout(time, ecg_fn, ecg)
                A = ensure_KT(ecg_fn(ecg), T)        # (Kc, T)
            except:
                print('data0')
                A = ensure_KT(np.zeros((49,256 *32), dtype=np.float32),T)
            s, e = starts[j], ends[j]
            tmp[s:e, :] = A                      # کپی مستقیم به بافر
        Xout[i, :, :] = tmp

    Xout.flush()
    del Xout
    np.save(Y_out, np.asarray(y, dtype=np.int32))
    return out_path

# ---------- main precompute (سریع و موازی روی شاردها) ----------
def precompute_features_with_fn(src_dir, dst_dir, channel_idx, ecg_fn, x_pat="X_*.npy", y_pat="y_*.npy", max_workers=None):
    os.makedirs(dst_dir, exist_ok=True)
    xs = sorted(glob.glob(os.path.join(src_dir, x_pat)))
    ys = sorted(glob.glob(os.path.join(src_dir, y_pat)))
    assert len(xs) == len(ys) and len(xs) > 0, "X/y files mismatch or empty!"

    sel = np.asarray(channel_idx, dtype=np.int64)
    # فقط یک بار K و T و offsetها را کشف کن
    T, K, starts, ends = _discover_K_T_first(xs, sel, ecg_fn)
    print(f"[precompute] Feature channels per sample = {K}, T = {T}, shards = {len(xs)}")

    # آرگومان‌های هر شارد
    shard_args = [(x_path, y_path, dst_dir, sel, T, K, starts, ends, ecg_fn) for x_path, y_path in zip(xs, ys)]

    # موازی‌سازی روی شاردها (process-based)
    if max_workers is None:
        max_workers = min(len(xs), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_process_one_shard, shard_args), total=len(shard_args), desc="Shards"))

# ---------- نمونه استفاده ----------
# از wrapper خودت استفاده کن که dict → (K,T) می‌سازد
# from preproses_signals.ECG import feature_ecg
# def dict_to_KT(...):   # مثل قبل ولی بدون append روی لیست! (از np.pad یا برش استفاده کن)
# def feat_ecg(ecg, T=32*256):
#     feat_dict = feature_ecg(ecg)
#     x_feat, _ = dict_to_KT(feat_dict, T)
#     return x_feat

# src_train = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/train"
# src_val   = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/val"
# dst_train = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1_feat/train"
# dst_val   = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1_feat/val"
# SELECTED_CH = [7]

# precompute_features_with_fn(src_train, dst_train, SELECTED_CH, feat_ecg)
# precompute_features_with_fn(src_val,   dst_val,   SELECTED_CH, feat_ecg)

# --- استفاده ---
# فرض: تابع آماده‌ی تو اسمش ecg_fn است و ورودی 1D (T,) می‌گیرد و خروجی (K,T) یا (T,K) می‌دهد.
# مسیرها را طبق ساختار خودت بزن:
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

precompute_features_with_fn(src_train, dst_train, SELECTED_CH, feat_ecg)
precompute_features_with_fn(src_val,   dst_val,   SELECTED_CH, feat_ecg)
