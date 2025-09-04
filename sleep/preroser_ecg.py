import os, glob
import numpy as np

def ensure_KT(a, T):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]               # [1, T?]
    if a.shape[-1] == T and a.ndim == 2:
        # [K, T]
        return a.astype(np.float32, copy=False)
    if a.shape[0] == T and a.ndim == 2:
        # [T, K] -> [K, T]
        return a.T.astype(np.float32, copy=False)
    raise ValueError(f"Unexpected shape from feature fn: {a.shape}, expected (K,T) or (T,K) with T={T}")

def precompute_features_with_fn(src_dir, dst_dir, channel_idx, ecg_fn, x_pat="X_*.npy", y_pat="y_*.npy"):
    os.makedirs(dst_dir, exist_ok=True)
    xs = sorted(glob.glob(os.path.join(src_dir, x_pat)))
    ys = sorted(glob.glob(os.path.join(src_dir, y_pat)))
    assert len(xs) == len(ys) and len(xs) > 0

    # کشف K و T از اولین نمونه
    X0 = np.load(xs[0], mmap_mode='r')   # (N, C, T)
    N0, C0, T = X0.shape
    sel = np.asarray(channel_idx, dtype=np.int64)
    test_feat = []
    for c in sel:
        test_feat.append(ensure_KT(ecg_fn(X0[0, c, :]), T))
    K = sum(arr.shape[0] for arr in test_feat)   # مجموع کانال‌های فیچر برای هر نمونه
    print(f"[precompute] Feature channels per sample = {K}, T = {T}")

    for x_path, y_path in zip(xs, ys):
        X = np.load(x_path, mmap_mode='r')      # (N, C, T)
        y = np.load(y_path, mmap_mode='r')      # (N,)
        N = X.shape[0]

        out_path = os.path.join(dst_dir, os.path.basename(x_path).replace("X_", "Xfeat_"))
        Y_out    = os.path.join(dst_dir, os.path.basename(y_path))  # لیبل‌ها همانند قبل

        # فایل خروجی را از پیش اختصاص بده (ممکن است چند گیگ باشد؛ یک‌بار برای همیشه)
        Xout = np.memmap(out_path, mode='w+', dtype=np.float32, shape=(N, K, T))

        write_idx = 0
        for i in range(N):
            parts = []
            for c in sel:
                parts.append(ensure_KT(ecg_fn(X[i, c, :]), T))
            Xi = np.vstack(parts)                # [K, T]
            Xout[write_idx, :, :] = Xi
            write_idx += 1

        del Xout
        np.save(Y_out, np.asarray(y, dtype=np.int32))
        print(f"[precompute] Wrote {out_path} & {Y_out}")

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
