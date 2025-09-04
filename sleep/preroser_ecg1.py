# ======== fast precompute on 24 cores (per-shard multiprocessing) ========
import os, glob, json
import numpy as np
from pathlib import Path
from numpy.lib.format import open_memmap
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 👉 توصیه: این متغیرها را قبل از import numpy ست کنی (اینجا هم می‌گذاریم در صورت اجراي ماژول):
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- helpers ----------
def ensure_KT(a, T):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim == 2 and a.shape[-1] == T:
        return a.astype(np.float32, copy=False)
    if a.ndim == 2 and a.shape[0] == T:
        return a.T.astype(np.float32, copy=False)
    # اجازه بده با پد/برش یک T دقیق بسازیم:
    if a.ndim == 2:
        # اگر محور زمان اشتباه است، ترنسپوز کن
        if a.shape[0] in (T-1, T, T+1) and a.shape[1] not in (T-1, T, T+1):
            a = a.T
        L = a.shape[-1]
        if L < T:
            pad = T - L
            a = np.pad(a, ((0,0),(0,pad)), mode='edge')
        elif L > T:
            a = a[..., :T]
        return a.astype(np.float32, copy=False)
    raise ValueError(f"Unexpected shape from feature fn: {a.shape}, expected around (K,T) with T={T}")

def _writable_1d(x):
    # ممری‌مپ read-only → کپی writeable و contiguous
    return np.ascontiguousarray(np.array(x, dtype=np.float32, copy=True))

def _discover_K_T_first(xs, selected_ch, feat_fn):
    """از اولین شارد، T و K (جمع Kهای هر کانال انتخابی) را کشف می‌کند؛
       همچنین شروع/پایان هر زیر-بخش K برای پر کردن بافر را برمی‌گرداند."""
    X0 = np.load(xs[0], mmap_mode='r')      # (N, C, T)
    _, C, T = X0.shape
    sel = np.asarray(selected_ch, dtype=np.int64)
    k_per_ch = []
    for c in sel:
        if not (0 <= c < C):
            raise IndexError(f"selected channel {c} out of range 0..{C-1}")
        ecg0 = _writable_1d(X0[0, c, :])
        A = ensure_KT(feat_fn(ecg0), T)     # (Kc, T)
        k_per_ch.append(int(A.shape[0]))
    K = int(sum(k_per_ch))
    starts = np.cumsum([0] + k_per_ch[:-1])
    ends   = starts + np.array(k_per_ch, dtype=int)
    return T, K, sel, starts, ends

def _process_one_shard(args):
    """
    یک شارد را در پردازهٔ جدا پردازش می‌کند.
    args: (x_path, y_path, dst_dir, T, K, sel, starts, ends, feat_fn)
    """
    x_path, y_path, dst_dir, T, K, sel, starts, ends, feat_fn = args

    X = np.load(x_path, mmap_mode='r')          # (N, C, T)
    y = np.load(y_path, mmap_mode='r')          # (N,)
    N = X.shape[0]

    out_x = Path(dst_dir) / Path(x_path).name.replace("X_", "Xfeat_")
    out_y = Path(dst_dir) / Path(y_path).name
    out_json = Path(dst_dir) / (Path(x_path).stem.replace("X_", "badidx_") + ".json")

    Xout = open_memmap(str(out_x), mode="w+", dtype=np.float32, shape=(N, K, T))
    bad = []                                    # (i, ch) های ناموفق

    # بافر موقت یک‌بار بساز (برای هر نمونه پر می‌شود)
    tmp = np.empty((K, T), dtype=np.float32)

    for i in tqdm(range(N)):
        # برای هر کانال انتخاب‌شده، فیچر را جای مناسبش بریز
        for j, c in enumerate(sel):
            s, e = starts[j], ends[j]
            try:
                ecg = _writable_1d(X[i, c, :])
                A = ensure_KT(feat_fn(ecg), T)        # (Kc, T)
                # اگر Kc با (e-s) جور نبود، امن کن
                Kc = e - s
                if A.shape[0] != Kc:
                    # هم‌ترازی با برش/پد روی محور K
                    if A.shape[0] > Kc:
                        A = A[:Kc, :]
                    else:
                        padK = Kc - A.shape[0]
                        A = np.pad(A, ((0,padK),(0,0)), mode='edge')
                tmp[s:e, :] = A
            except Exception:
                bad.append((i, int(c)))
                tmp[s:e, :] = 0.0
        Xout[i, :, :] = tmp

    Xout.flush(); del Xout
    # لیبل‌ها را مستقیم ذخیره کن (نوع int32/16 اختیاری)
    np.save(str(out_y), np.asarray(y, dtype=np.int32))
    # ایندکس‌های خراب را لاگ کن
    if bad:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(bad, f, ensure_ascii=False, indent=2)
    return str(out_x)

def precompute_features_mp(src_dir, dst_dir, selected_ch, feat_fn, max_workers=24):
    os.makedirs(dst_dir, exist_ok=True)
    xs = sorted(p for p in glob.glob(os.path.join(src_dir, "X_*.npy")) if os.path.isfile(p))
    ys = sorted(p for p in glob.glob(os.path.join(src_dir, "y_*.npy")) if os.path.isfile(p))
    assert xs and len(xs) == len(ys), f"X/Y mismatch or empty in {src_dir}"

    # یک‌بار T و K را کشف کن
    T, K, sel, starts, ends = _discover_K_T_first(xs, selected_ch, feat_fn)
    print(f"[discover] T={T}  K={K}  shards={len(xs)}  sel={list(sel)}")

    tasks = [(x_path, y_path, dst_dir, T, K, sel, starts, ends, feat_fn) for x_path, y_path in zip(xs, ys)]
    workers = min(max_workers, len(tasks))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(tqdm(ex.map(_process_one_shard, tasks), total=len(tasks), desc="Shards"))

# ----------------- استفاده -----------------
# تابع فیچر خودت (dict → (K,T))؛ همان که قبلاً داشتی:
# from preproses_signals.ECG import feature_ecg
def dict_to_KT(feat_dict: dict, T: int):
    keys_sorted = sorted(feat_dict.keys())
    mats = []
    for k in keys_sorted:
        a = feat_dict[k]
        A = ensure_KT(a, T)          # شامل پد/برش امن
        mats.append(A.astype(np.float32, copy=False))
    return np.vstack(mats), keys_sorted
from preproses_signals.ECG import feature_ecg
def feat_ecg(ecg, T=32*256):
    feat_dict = feature_ecg(ecg)     # تابعِ خودت از ماژول ECG
    x_feat, _ = dict_to_KT(feat_dict, T)
    return x_feat

# مثال اجرا:
src_train = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/train"
src_val   = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/val"
dst_train = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1_feat/train"
dst_val   = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1_feat/val"
SELECTED_CH = [7]  # ECG

precompute_features_mp(src_train, dst_train, SELECTED_CH, feat_ecg, max_workers=24)
precompute_features_mp(src_val,   dst_val,   SELECTED_CH, feat_ecg, max_workers=24)
