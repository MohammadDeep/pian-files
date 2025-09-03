'''
====================================================================
                            haper pramatres
====================================================================                            
'''
# folder dataset .npy
folder = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag"
EPOCHES = 100

dir_history_model = '/home/asr/mohammadBalaghi/pian-files/__HISTORY_MODEL'


'''
====================================================================
                            Create dataset
====================================================================                            
'''


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
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

        #x = torch.from_numpy(x).to(self.dtype)
        #y = torch.tensor(y, dtype=torch.long)
        x = torch.from_numpy(x).to(self.dtype)
        y = torch.tensor(int(self._z["y"][local_idx]), dtype=torch.long) # type: ignore
        
        return x, y


from torch.utils.data import DataLoader
files = [f for f in os.listdir(folder) if f.endswith(".npz")]
print(files)
print(len(files))
files =[f'{folder}/{f}' for f in files]
file_val = files[:20]
file_train = files[20:]
print('start create dataset')
dataset_train = MultiNpzDataset(file_train)
dataset_val = MultiNpzDataset(file_val)
x1, y1 = dataset_train[0]
print('data shape : ', x1.shape)
print('y shape', y1)
print('create dataloader')
dataloader_train = DataLoader(
    dataset_train, batch_size=32, shuffle=True,
    num_workers=2,                 # کم نگه دار
    pin_memory=True,              # اگر GPU داری بعداً True کن
    prefetch_factor=1,             # پیش‌واکشی کم
    persistent_workers=False       # ورکرها را دائمی نکن
)
dataloader_val = DataLoader(
    dataset_val, batch_size=32, shuffle=False,
    num_workers=2,                 # کم نگه دار
    pin_memory=True,              # اگر GPU داری بعداً True کن
    prefetch_factor=1,             # پیش‌واکشی کم
    persistent_workers=False       # ورکرها را دائمی نکن
)

for xb, yb in dataset_train:
    print(xb.shape, yb.shape)
    break




'''
====================================================================
                            Create modeles
====================================================================                            
'''

from pre_modeles.pre_modeles.t_model import SimpleResNet,CNN_LSTM_Model,BasicBlock,CNN_LSTM_Model,LSTM_Model
'''
start lne  = 2 * 256
step  = 2 * 256
'''
net1 = SimpleResNet(BasicBlock, layers=[1,1,1,2,2],list_step = [2,2,2,1,1], in_ch=8, base_planes=16)
lstm1 = LSTM_Model(input_size = 256,
                    hidden_size = 128,
                    num_layers = 2, 
                    num_classes = 5)
model = CNN_LSTM_Model(net1, lstm1)


N = 30  * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, 8, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model(dummy_input)
print(output.size())

from torchinfo import summary
summary(model, input_size=(32, 8, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from tqdm import tqdm
import torch

for epoch in tqdm(range(EPOCHES)):
    # ------------------------------
    # Train
    # ------------------------------
    model.train()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    
    for xb, yb in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHES} [train]"):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)                     # [B, n_classes]
        loss = loss_function(logits, yb)       # CrossEntropy

        loss.backward()
        optimizer.step()

        # آمار
        total_loss += loss.item() * xb.size(0)     # sum loss (وزن‌دار به اندازه batch)
        preds = logits.argmax(dim=1)
        labels = yb  # one-hot → index
        correct += (preds == labels).sum().item()

        total += yb.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # ------------------------------
    # Validation
    # ------------------------------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{EPOCHES} [val]"):
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = loss_function(logits, yb)

            val_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            labels = yb   # one-hot → index
            val_correct += (preds == labels).sum().item()
            val_total += yb.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    # ------------------------------
    # Print summary
    # ------------------------------
    print(f"Epoch {epoch+1}/{EPOCHES} "
          f"| train loss: {train_loss:.4f}, train acc: {train_acc:.3f} "
          f"| val loss: {val_loss:.4f}, val acc: {val_acc:.3f}")
