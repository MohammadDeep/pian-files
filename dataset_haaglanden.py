'''
====================================================================
                            haper pramatres
====================================================================                            
'''
# folder dataset .npy
folder = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag"





'''
====================================================================
                            Create dataset
====================================================================                            
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiNpzDataset(Dataset):
    def __init__(self, files, mmap=True, dtype=torch.float32):
        self.files = files
        self.mmap = mmap
        self.dtype = dtype

        # لیست طول هر فایل (برای mapping ایندکس کلی → فایل محلی)
        self.file_lengths = []
        for f in self.files:
            with np.load(f, mmap_mode="r" if mmap else None) as d:
                self.file_lengths.append(d["y"].shape[0])

        self.cum_lengths = np.cumsum(self.file_lengths)
        self.N = self.cum_lengths[-1]

        # کش برای فایل جاری
        self._z = None
        self._current_file = None

    def __len__(self):
        return self.N

    def _open_file(self, idx_file):
        if self._current_file != idx_file:
            self._z = np.load(self.files[idx_file], mmap_mode="r" if self.mmap else None)
            self._current_file = idx_file

    def __getitem__(self, idx):
        # پیدا کردن فایل و ایندکس محلی
        idx_file = np.searchsorted(self.cum_lengths, idx, side="right")
        start = 0 if idx_file == 0 else self.cum_lengths[idx_file-1]
        local_idx = idx - start

        self._open_file(idx_file)

        x = self._z["X"][local_idx] # type: ignore
        y = int(self._z["y"][local_idx]) # type: ignore

        x = torch.from_numpy(x).to(self.dtype)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


from torch.utils.data import DataLoader
files = [f for f in os.listdir(folder) if f.endswith(".npz")]
print(files)
print(len(files))
files =[f'{folder}/{f}' for f in files]
print('start create dataset')
ds = MultiNpzDataset(files)
x1, y1 = ds[0]
print('data shape : ', x1.shape)
print('y shape', y1)
print('create dataloader')
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)

for xb, yb in dl:
    print(xb.shape, yb.shape)
    break
