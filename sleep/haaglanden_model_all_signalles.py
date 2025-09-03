'''
====================================================================
                            haper pramatres
====================================================================                            
'''
# folder dataset .npy
folder = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag"
EPOCHES = 100

dir_history_model = '/home/asr/mohammadBalaghi/pian-files/__HISTORY_MODEL'
N_CLASSES = 5
IN_CH = 8

NH_LISTM = 128 
SEC = 32

BATCH_SIZE = 128  *4
NUM_WORKERS = 20

dir_save_modeles = '/home/asr/mohammadBalaghi/pian-files/__HISTOY_MODEL/haag_all_signall'

signal_names = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG']

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
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset



from pathlib import Path
save_dir = Path(dir_save_modeles)
save_dir.mkdir(parents=True, exist_ok=True)

class ShardedNPYDataset(Dataset):
    """
    شاردها را مثل X_000.npy / y_000.npy می‌خواند (X: [N, C, T], y: [N]).
    خواندن با mmap انجام می‌شود تا رم زیاد مصرف نشود.
    """
    def __init__(self, root_dir,
                 x_pattern="X_*.npy",
                 y_pattern="y_*.npy",
                 transform=None,
                 normalize=None,     # None یا dict مثل {"mean": [..], "std": [..]}
                 dtype_x=torch.float32,
                 dtype_y=torch.long):
        self.root_dir = str(root_dir)
        self.transform = transform
        self.normalize = normalize
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y

        # پیدا کردن شاردهای جفت
        xs = sorted(glob.glob(os.path.join(self.root_dir, x_pattern)))
        ys = sorted(glob.glob(os.path.join(self.root_dir, y_pattern)))
        assert len(xs) == len(ys) and len(xs) > 0, "تعداد X و y برابر/غیرصفر نیست"

        # mmap برای هر شارد
        self.shards = []
        self.cum_lens = [0]
        for x_path, y_path in zip(xs, ys):
            X = np.load(x_path, mmap_mode='r')  # شکل: (N, C, T)
            y = np.load(y_path, mmap_mode='r')  # شکل: (N,)
            assert X.shape[0] == y.shape[0], f"عدم تطابق N در {x_path}"
            self.shards.append((X, y, x_path, y_path))
            self.cum_lens.append(self.cum_lens[-1] + X.shape[0])

        self.total_len = self.cum_lens[-1]

        # اگر normalize داده شد، به آرایهٔ کانالی تبدیلش می‌کنیم (C, 1) برای broadcast
        if self.normalize is not None:
            m = np.asarray(self.normalize["mean"], dtype=np.float32)
            s = np.asarray(self.normalize["std"],  dtype=np.float32)
            self.norm_mean = torch.from_numpy(m)[:, None]  # (C,1)
            self.norm_std  = torch.from_numpy(s)[:, None]  # (C,1)
        else:
            self.norm_mean, self.norm_std = None, None

    def __len__(self):
        return self.total_len

    def _locate(self, idx):
        # idx جهانی → (idx شارد، offset داخل شارد)
        # باینری‌سرچ سریع‌تر است، ولی همین خطی هم معمولاً کافی است.
        lo, hi = 0, len(self.cum_lens)-1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cum_lens[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard_idx = lo - 1
        offset = idx - self.cum_lens[shard_idx]
        return shard_idx, offset

    def __getitem__(self, idx):
        shard_idx, k = self._locate(idx)
        X, y, _, _ = self.shards[shard_idx]

        x_np = X[k]   # (C, T)  از نوع numpy.memmap
        y_np = y[k]   # ()      numpy.int

        x = torch.from_numpy(np.array(x_np, copy=False)).to(self.dtype_x)  # (C,T)
        if self.norm_mean is not None:
            # نرمال‌سازی per-channel: (C,T) ← (C,1) broadcast
            x = (x - self.norm_mean) / (self.norm_std + 1e-6)

        if self.transform is not None:
            x = self.transform(x)  # اگر augment داری

        y_t = torch.tensor(int(y_np), dtype=self.dtype_y)
        return x, y_t

def compute_channel_stats(root_dir, x_pattern="X_*.npy", eps=1e-12):
    """
    میانگین و std کانالی (روی محور نمونه‌ها و زمان) را به‌صورت استریم محاسبه می‌کند.
    خروجی: dict {'mean': [C], 'std': [C]}
    """
    xs = sorted(glob.glob(os.path.join(root_dir, x_pattern)))
    assert len(xs) > 0

    # اول شکل C را پیدا کنیم
    X0 = np.load(xs[0], mmap_mode='r')
    _, C, T = X0.shape

    count = 0
    sum_c = np.zeros(C, dtype=np.float64)
    sumsq_c = np.zeros(C, dtype=np.float64)

    for x_path in xs:
        X = np.load(x_path, mmap_mode='r')   # (N, C, T)
        N = X.shape[0]
        # به‌جای لود کامل، روی محور (N,T) میانگین و مربع‌میانگین را می‌گیریم:
        # mean over axes (0,2): خروجی (C,)
        m = X.mean(axis=(0,2))        # (C,)
        v = ((X**2).mean(axis=(0,2))) # (C,)

        # تبدیل به جمع کلی با وزن N*T
        weight = N * T
        sum_c   += m * weight
        sumsq_c += v * weight
        count   += weight

    mean = (sum_c / count).astype(np.float32)
    var  = (sumsq_c / count - mean**2).clip(min=eps).astype(np.float32)
    std  = np.sqrt(var)
    return {"mean": mean.tolist(), "std": std.tolist()}

from torch.utils.data import random_split

root_dir = "/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1"

# (اختیاری) یک بار محاسبه و ذخیره کن، بعداً همان را استفاده کن:
stats = compute_channel_stats(root_dir)
print("stats:", stats)  # {'mean': [...], 'std': [...]}

full_ds = ShardedNPYDataset(root_dir, normalize=stats)

# اسپلیتِ ساده (۸۰/۲۰)
n_total = len(full_ds)
n_train = int(0.8 * n_total)
n_val   = n_total - n_train
train_ds, val_ds = full_ds[:n_train], full_ds[n_train:]

# (اختیاری) وزن کلاس‌ها و WeightedRandomSampler برای دیتاست train
# ابتدا شمارش کلاس‌ها:
def count_classes(ds, max_scan=200000):
    from collections import Counter
    cnt = Counter()
    L = min(len(ds), max_scan)
    for i in range(L):
        _, y = ds[i]
        cnt[int(y.item())] += 1
    return cnt

cls_counts = count_classes(train_ds)
print("train class counts:", cls_counts)

num_classes = max(cls_counts.keys()) + 1
counts = np.zeros(num_classes, dtype=np.float64)
for k,v in cls_counts.items():
    counts[k] = v
class_weights = 1.0 / np.clip(counts, 1, None)
sample_weights = []
for _, y in train_ds:
    sample_weights.append(class_weights[int(y)])
sample_weights = torch.tensor(sample_weights, dtype=torch.float)

sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

# DataLoader ها

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=NUM_WORKERS,
                          pin_memory=True, drop_last=True)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True)














'''
====================================================================
                            Create modeles
====================================================================                            
'''

from pre_modeles.pre_modeles.t_model import SimpleResNet,CNN_LSTM_Model,BasicBlock,CNN_LSTM_Model,LSTM_Model,CNN_LSTM_Model1,SimpleResNet1,CNN_LSTM_Model2,SimpleResNet2


'''
====================================================================
                            modele1
====================================================================                            
'''
print('-' * 50)
print('model1 : ')

'''
start lne  = 2 * 256
step  = 2 * 256
'''
net1 = SimpleResNet(BasicBlock, layers=[1,1,1,2,2],list_step = [2,2,2,1,1], in_ch=IN_CH, base_planes=16)
lstm1 = LSTM_Model(input_size = 256,
                    hidden_size = NH_LISTM,
                    num_layers = 2, 
                    num_classes = N_CLASSES)
model1 = CNN_LSTM_Model(net1, lstm1)


N = SEC  * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, IN_CH, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model1(dummy_input)
print(output.size())

from torchinfo import summary
summary(model1, input_size=(32, IN_CH, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model1.to(device)



'''
====================================================================
                            modele2
====================================================================                            
'''
print('-' * 50)
print('model2 : ')

'''
start lne  = 1 * 256
step  = 1 * 256
'''
net2 = SimpleResNet(BasicBlock, layers=[1,1,1,2,2],list_step = [2,2,1,1,1], in_ch=IN_CH, base_planes=16)
lstm2 = LSTM_Model(input_size = 256,
                hidden_size = NH_LISTM,
                    num_layers = 2, 
                    num_classes = N_CLASSES)
model2 = CNN_LSTM_Model(net2, lstm2)


N = SEC  * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, IN_CH, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model2(dummy_input)
print(output.size())

from torchinfo import summary
summary(model2, input_size=(32, IN_CH, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-4)
model2.to(device)



'''
====================================================================
                            modele3
====================================================================                            
'''
print('-' * 50)
print('model3 : ')

'''
start lne  = .5 * 256
step  = .5 * 256
'''
net3 = SimpleResNet(BasicBlock, layers=[1,1,1,2,2],list_step = [2,1,1,1,1], in_ch=IN_CH, base_planes=16)
lstm3 = LSTM_Model(input_size = 256,
                hidden_size = NH_LISTM,
                    num_layers = 2, 
                    num_classes = N_CLASSES)
model3 = CNN_LSTM_Model(net3, lstm3)


N = SEC  * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, IN_CH, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model3(dummy_input)
print(output.size())

from torchinfo import summary
summary(model3, input_size=(32, IN_CH, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function3 = nn.CrossEntropyLoss()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3, weight_decay=1e-4)
model3.to(device)


'''
====================================================================
                            modele4
====================================================================                            
'''
print('-' * 50)
print('model4 : ')

'''
start lne  = 1 * 256
step  = 1 * 256
'''

net4 = SimpleResNet1(BasicBlock, layers=[1,1,1,2,2,2,2],list_step = [2,1,1,1,1,1,1], in_ch=IN_CH, base_planes=4)
lstm4 = LSTM_Model(input_size = 256,
                hidden_size = NH_LISTM,
                    num_layers = 2, 
                    num_classes = N_CLASSES)
model4 = CNN_LSTM_Model1(net4, lstm4)


N = SEC  * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, IN_CH, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model4(dummy_input)
print(output.size())

from torchinfo import summary
summary(model4, input_size=(32, IN_CH, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function4 = nn.CrossEntropyLoss()
optimizer4 = torch.optim.Adam(model4.parameters(), lr=1e-3, weight_decay=1e-4)
model4.to(device)


'''
====================================================================
                            modele5
====================================================================                            
'''
print('-' * 50)
print('model5 : ')


'''
start lne  = 4 * 256
step  = 4 * 256
'''
net5 = SimpleResNet1(BasicBlock, layers=[2,2,2,2,2,2,2],list_step = [2,2,1,1,1,1,1], in_ch=IN_CH, base_planes=4)
lstm5 = LSTM_Model(input_size = 256,
                hidden_size = NH_LISTM,
                    num_layers = 2, 
                    num_classes = N_CLASSES)
model5 = CNN_LSTM_Model1(net5, lstm5)


N = SEC * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, IN_CH, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model5(dummy_input)
print(output.size())

from torchinfo import summary
summary(model5, input_size=(32, IN_CH, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function5 = nn.CrossEntropyLoss()
optimizer5 = torch.optim.Adam(model5.parameters(), lr=1e-3, weight_decay=1e-4)
model5.to(device)




'''
====================================================================
                            modele6
====================================================================                            
'''
print('-' * 50)
print('model5 : ')

'''
start lne  = 2* 256
step  = 2 * 256
'''
net6 = SimpleResNet1(BasicBlock, layers=[2,2,2,2,2,2,2],list_step = [2,1,1,1,1,1,1], in_ch=IN_CH, base_planes=4)
lstm6 = LSTM_Model(input_size = 256,
                hidden_size = NH_LISTM,
                    num_layers = 2, 
                    num_classes = N_CLASSES)
model6 = CNN_LSTM_Model1(net6, lstm6)


N = SEC  * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, IN_CH, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model6(dummy_input)
print(output.size())

from torchinfo import summary
summary(model6, input_size=(32, IN_CH, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function6 = nn.CrossEntropyLoss()
optimizer6 = torch.optim.Adam(model6.parameters(), lr=1e-3, weight_decay=1e-4)
model6.to(device)





'''
====================================================================
                            modele7
====================================================================                            
'''
print('-' * 50)
print('model7 : ')
'''
start lne  =   256//32
step  = 256//32
'''
net7 = SimpleResNet2(BasicBlock, layers=[1,1,1,1,1,1,1,1,1,1,1,1],list_step = [2,2,1,1,1,1,1,1,1,1,1], in_ch=IN_CH, base_planes=4) 
lstm7 = LSTM_Model(input_size = 64,
                hidden_size = NH_LISTM,
                    num_layers = 2, 
                    num_classes = N_CLASSES)
model7 = CNN_LSTM_Model2(net7, lstm7)



N = SEC * 256
# ورودی به مدل باید یک تنسور باشد
dummy_input = torch.randn(32, IN_CH, N)  # [batch_size, channels, sequence_length]
with torch.no_grad():
    output = model7(dummy_input)
print(output.size())

from torchinfo import summary
summary(model7, input_size=(32, IN_CH, N)) 

 
from vision.train_val_functiones.train_val_functiones import train
import torch.nn as nn
loss_function7 = nn.CrossEntropyLoss()
optimizer7 = torch.optim.Adam(model7.parameters(), lr=1e-3, weight_decay=1e-4)
model7.to(device)



from tqdm import tqdm
import torch
list_modeles = [model1, model2, model3, model4, model5, model6, model7]
list_optimzers = [optimizer1,
                   optimizer2,
                   optimizer3,
                   optimizer4,
                   optimizer5,
                   optimizer6,
                   optimizer7
                   ]
list_loss_funciton = [loss_function1,
                      loss_function2,
                      loss_function3,
                      loss_function4,
                      loss_function5,
                      loss_function6,
                      loss_function7]


history = []
for epoch in tqdm(range(EPOCHES)):
    # ------------------------------
    # Train
    # ------------------------------
   
    
    total_loss_list, correct_list, total_list = [0.0 for i in range(len(list_modeles))], [0.0 for i in range(len(list_modeles))], [0.0 for i in range(len(list_modeles))]
    
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHES} [train]"):
        xb, yb = xb.to(device), yb.to(device)
        for i1 in range(len(list_modeles)):
            model = list_modeles[i1]
            optimizer = list_optimzers[i1]
            loss_function = list_loss_funciton[i1]
            total_loss = total_loss_list[i1]
            correct = correct_list[i1]
            total = total_list[i1]
            model.train()
            optimizer.zero_grad()
            logits = model(xb)                     # [B, n_classes]
            loss = loss_function(logits, yb)       # CrossEntropy

            loss.backward()
            optimizer.step()

            # آمار
            total_loss_list[i1] += loss.item() * xb.size(0)     # sum loss (وزن‌دار به اندازه batch)
            preds = logits.argmax(dim=1)
            labels = yb  # one-hot → index
            correct_list[i1] += (preds == labels).sum().item()
            
            total_list[i1] +=  yb.size(0)
    train_loss, train_acc = [] ,[]
  
    for i5 in range(len(total_loss_list)):
        total_loss = total_loss_list[i5]
        total = total_list[i5]
        correct = correct_list[i5]
        train_loss.append(total_loss / total)
        train_acc.append(correct / total)

    # ------------------------------
    # Validation
    # ------------------------------
    
    val_loss_list, val_correct_list, val_total_list = [0.0 for i in range(len(list_modeles))], [0.0 for i in range(len(list_modeles))], [0.0 for i in range(len(list_modeles))]
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHES} [val]"):
            xb, yb = xb.to(device), yb.to(device)
            for i3 in range(len(list_modeles)):
                model = list_modeles[i3]
                loss_function = list_loss_funciton[i3]

                model.eval()
                logits = model(xb)
                loss = loss_function(logits, yb)

                val_loss_list[i3] += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                labels = yb   # one-hot → index
                val_correct_list[i3] += (preds == labels).sum().item()
                val_total_list[i3] += yb.size(0)

 

    val_loss, val_acc = [] ,[]
    for i7 in range(len(total_loss_list)):
        total_loss = val_loss_list[i7]
        total = val_total_list[i7]
        correct = val_correct_list[i7]
        val_loss.append(total_loss / total)
        val_acc.append(correct / total)
    # ------------------------------
    # Print summary
    # ------------------------------
    for i4 in range(len(val_acc)):
        print('-' * 50)
        print(f'model : {i4 + 1}')
        print(f"Epoch {epoch+1}/{EPOCHES} "
            f"| train loss: {train_loss[i4]:.4f}, train acc: {train_acc[i4]:.3f} "
            f"| val loss: {val_loss[i4]:.4f}, val acc: {val_acc[i4]:.3f}")
        history.append((train_loss[i4],train_acc[i4],val_loss[i4],val_acc[i4]))
    s = 0
    for model in list_modeles:
        # save
        s += 1
        torch.save(model.state_dict(), f"{dir_save_modeles}/model_{s}_weights_epoch{epoch}.pth")

import pickle
# سیو
with open(f"{dir_save_modeles}/my_dict.pkl", "wb") as f:
    pickle.dump(history, f)