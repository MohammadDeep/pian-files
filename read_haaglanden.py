from pathlib import Path



root = Path("/home/asr/mohammadBalaghi/dataset_signal/newdatahaag")   # مسیر پوشه
#root  = Path('/media/mohammad/NewVolume/signalDataset/haaglanden-medisch-centrum-sleep-staging-database-1.1/recordings')


dst_dir = Path('/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1')

n_file = 26


import numpy as np
import pandas as pd



import os, glob, numpy as np

def save_shard(dst_dir, shard_id, Xs, Ys):
    Xs = np.ascontiguousarray(Xs, dtype=np.float32)
    Ys = np.ascontiguousarray(Ys, dtype=np.int32)
    
    np.save(os.path.join(dst_dir, f"X_{shard_id:03d}.npy"), Xs)
    np.save(os.path.join(dst_dir, f"y_{shard_id:03d}.npy"), Ys)
def read_data_haaglanden(
     root:str,
     number_persion : int = 1,
     print_data_analez : bool = False,
     stage_map = {
                    "Sleep stage W": 0, "W": 0,
                    "Sleep stage N1": 1, "Sleep stage 1": 1, "N1": 1,
                    "Sleep stage N2": 2, "Sleep stage 2": 2, "N2": 2,
                    "Sleep stage N3": 3, "Sleep stage 3": 3, "Sleep stage 4": 3, "N3": 3,
                    "Sleep stage R": 4, "R": 4, "REM": 4,
                    },
     win:int = 30
     ):

    psg_file = root / f"SN{number_persion:03d}.edf" # type: ignore
    print('dir',psg_file )
    raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None, verbose=False) # type: ignore

    data_x = raw.get_data()
    channel_name = raw.ch_names



    scoring_edf = root / f"SN{number_persion:03d}_sleepscoring.edf" # type: ignore

    # خواندن annotation ها
    ann = mne.read_annotations(scoring_edf)

    # تبدیل به DataFrame
    df = pd.DataFrame({
    "start_sec": ann.onset,
    "duration_sec": ann.duration,
    "label": ann.description
    })



    valid_labels = set(stage_map.keys())
    df_clean = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    df_clean["label"] = df_clean["label"].map(stage_map)

    data_y = df_clean.astype(int).to_numpy()

    # save dataset 

    idxs, Xs, ys,n_p = [], [], [],[]
    list_start = list(df_clean['start_sec'])
    list_label = list(df_clean['label'])
    for i in range(len(list(df_clean['start_sec']))):
        start = list_start[i]
        label = list_label[i]
        end = start + win

        Xs.append(data_x[:, int(start):int(end)])               # type: ignore # یک پنجره به طول win
        # مثالِ ساده برای لیبل: میانگین/مود برچسب‌ها داخل پنجره
        y = label
        ys.append(y)
        idxs.append(start)                         # شروع پنجره (برای متادیتا)
        n_p.append(number_persion)
        X = np.stack(Xs, 0)                            # شکل: [N, C, win]
        y = np.array(ys) 
    if print_data_analez:
        print('-' * 50)
        print('x analize : ')
        print('shape x: ',data_x.shape) # type: ignore
        print('channel signal :',channel_name)
        print('-' * 50)

        
        print('y analize :')
        print('number vlaue :' ,df["label"].value_counts())
        print('number vlaue (clean label) :' ,df_clean["label"].value_counts())
        print('shape label:', data_y.shape)
        print('-' * 50)

     
    return data_x, data_y, X, y, np.array(idxs, dtype=np.int32),np.array(n_p, dtype=np.int32) # type: ignore


fs = 256

import numpy as np
import mne

persion = 154
X_data, Y_data = [] , []
# X: [N, C, T]   y: [N]   meta: هرچه لازم داری (مثلاً subject_id, start_idx)
for i in range(persion):
    print('-'*50)
    number_persion = i +1
    print(number_persion)
    try :
        _,_, X, y, start_idx,subject_id = read_data_haaglanden(
            root = root, # type: ignore
            number_persion = number_persion,
            print_data_analez  = False,
            stage_map = {
                            "Sleep stage W": 0, "W": 0,
                            "Sleep stage N1": 1, "Sleep stage 1": 1, "N1": 1,
                            "Sleep stage N2": 2, "Sleep stage 2": 2, "N2": 2,
                            "Sleep stage N3": 3, "Sleep stage 3": 3, "Sleep stage 4": 3, "N3": 3,
                            "Sleep stage R": 4, "R": 4, "REM": 4,
                            },
            win = 32 * fs
            )
        np.savez_compressed(
            f"/home/asr/mohammadBalaghi/dataset_signal/newdatahaag/s{number_persion}.npz",
                X=X.astype("float32"),         # کم‌حجم‌تر از float64
                y=y.astype("int16"),
                subject_id=subject_id.astype("int16"),
                start_idx=start_idx.astype("int32")
            )
        X_data.append(X)
        Y_data.append(y)

        
        if number_persion % n_file == 0  or  number_persion == persion : 
            
            save_shard(dst_dir, number_persion, X_data, Y_data)
            X_data, Y_data = [],[]
            
        
    except:
        print(f'number {number_persion} is not define.')

