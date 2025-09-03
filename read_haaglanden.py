from pathlib import Path



root = Path("/home/asr/mohammadBalaghi/dataset_signal/newdatahaag")   # مسیر پوشه
#root  = Path('/media/mohammad/NewVolume/signalDataset/haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings')


dst_dir = Path('/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1')

n_file = 26


import numpy as np
import pandas as pd



import os, glob, numpy as np
def save_shard(dst_dir, shard_id, Xs, Ys):
    # Xs, Ys ممکن است لیستِ چند آرایه باشند → اول کانکت
    if isinstance(Xs, list):
        Xs = np.concatenate(Xs, axis=0)
    if isinstance(Ys, list):
        Ys = np.concatenate(Ys, axis=0)

    Xs = np.ascontiguousarray(Xs, dtype=np.float32)
    Ys = np.ascontiguousarray(Ys, dtype=np.int32)  # برای آموزش بعداً به torch.long تبدیل کن

    np.save(os.path.join(dst_dir, f"X_{shard_id:03d}.npy"), Xs)
    np.save(os.path.join(dst_dir, f"y_{shard_id:03d}.npy"), Ys)

def read_data_haaglanden(
     root: Path,
     number_persion: int = 1,
     print_data_analez: bool = False,
     stage_map = {
        "Sleep stage W": 0, "W": 0,
        "Sleep stage N1": 1, "Sleep stage 1": 1, "N1": 1,
        "Sleep stage N2": 2, "Sleep stage 2": 2, "N2": 2,
        "Sleep stage N3": 3, "Sleep stage 3": 3, "Sleep stage 4": 3, "N3": 3,
        "Sleep stage R": 4, "R": 4, "REM": 4,
     },
     win: int = 32 *256
):
    psg_file = root / f"SN{number_persion:03d}.edf"
    raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None, verbose=False)
    data_x = raw.get_data().astype(np.float32, copy=False)   # [C, T_total]
    C, T_total = data_x.shape

    scoring_edf = root / f"SN{number_persion:03d}_sleepscoring.edf"
    ann = mne.read_annotations(scoring_edf)

    df = pd.DataFrame({
        "start_sec": ann.onset,
        "duration_sec": ann.duration,
        "label": ann.description
    })

    df = df[df["label"].isin(stage_map)].copy()
    df["label"] = df["label"].map(stage_map)

    starts = df["start_sec"].to_numpy(dtype=np.int64)
    labels = df["label"].to_numpy(dtype=np.int32)

    # فقط پنجره‌هایی که کامل داخل سیگنال هستند
    mask = (starts + win) <= T_total
    starts = starts[mask]
    labels = labels[mask]
    n = len(starts)
    if n == 0:
        return None  # یا هر هندلینگ مناسب

    # پیش‌اختصاص و پر کردن سریع
    X = np.empty((n, C, win), dtype=np.float32)
    for j, s in enumerate(starts):
        X[j] = data_x[:, s:s+win]

    y = labels
    start_idx  = starts.astype(np.int32)
    subject_id = np.full(n, number_persion, dtype=np.int32)

    if print_data_analez:
        print('-'*50)
        print('x shape:', data_x.shape, 'X windows:', X.shape)
        print('label counts:', df["label"].value_counts().to_dict())

    return X, y, start_idx, subject_id
fs = 256

import numpy as np
import mne

from pathlib import Path
import os, numpy as np, mne, pandas as pd

root = Path("/home/asr/mohammadBalaghi/dataset_signal/newdatahaag")
dst_dir = Path('/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1')
dst_dir.mkdir(parents=True, exist_ok=True)

persion = 154
n_file  = 26
fs = 256

X_buf, y_buf = [], []
shard_id = 0

for number_persion in range(75, persion+1):
    print('-'*50, number_persion)
    try:
        out = read_data_haaglanden(
            root=root,
            number_persion=number_persion,
            print_data_analez=False,
            win=32*fs
        )
        if out is None:
            print(f"no windows for {number_persion}")
            continue

        X, y, start_idx, subject_id = out

        # (اختیاری) اگر حتماً می‌خواهی npz per-subject نگه داری، بدون فشرده‌سازی:
        # np.savez(dst_dir / f"s{number_persion}.npz", X=X, y=y,
        #          subject_id=subject_id, start_idx=start_idx)

        X_buf.append(X)
        y_buf.append(y)

        if number_persion % n_file == 0 or number_persion == persion:
            save_shard(dst_dir, shard_id, X_buf, y_buf)
            shard_id += 1
            X_buf, y_buf = [], []

    except Exception as e:
        print(f"number {number_persion} skipped: {e}")
