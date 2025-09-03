folder = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag"

import os
import numpy as np
files = [f for f in os.listdir(folder) if f.endswith(".npz")]
print(files)
print(len(files))
files =[f'{folder}/{f}' for f in files]

import glob
from tqdm import tqdm
X_all, y_all, subj_all, start_all = [], [], [], []
for i in tqdm(range(84,(84 + len(files)))):
    f = files[i]
    d = np.load(f)
    X_all.append(d["X"].astype(np.float32))
    y_all.append(d["y"].astype(np.int64))
    #subj_all.append(d["subject_id"])
    #start_all.append(d["start_idx"])

    if (i+1) % 21 == 0 or i+1 == len(files):
        print('save file')
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        #subj_all = np.concatenate(subj_all, axis=0)
        #start_all = np.concatenate(start_all, axis=0)

        np.save(f"/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/X_all_end{i+1}.npy", X_all)
        np.save(f"/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/y_all_end{i+1}.npy", y_all)
        #np.save(f"/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/subject_all_end{i+1}.npy", subj_all)
        #np.save(f"/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/startidx_all_end{i+1}.npy", start_all)
        X_all, y_all, subj_all, start_all = [], [], [], []