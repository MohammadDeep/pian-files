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
    def __init__(self, files,files_y, mmap=True, dtype=torch.float32):
        self.files = files
        self.files_y = files_y
        self.mmap = mmap
        self.dtype = dtype

        '''# لیست طول هر فایل (برای mapping ایندکس کلی → فایل محلی)
        self.file_lengths = []
        for f in self.files_y:
            d = np.load(f)                      # mmap_mode نگذار؛ روی npz فشرده اثری ندارد
            self.file_lengths.append(d.shape[0])
            if hasattr(d, "close"):
                d.close()
        '''
        '''
        self.cum_lengths = np.cumsum(self.file_lengths)
        self.N = self.cum_lengths[-1]
        '''
        # کش برای فایل جاری
        self._z = None
        self._current_file = None

    def __len__(self):
        return self.N

    def _open_file(self, idx_file):
        if self._current_file != idx_file:
            self._x = np.load(self.files[idx_file], mmap_mode="r" if self.mmap else None)
            self._y = np.load(self.files_y[idx_file], mmap_mode="r" if self.mmap else None)
            self._current_file = idx_file

    def __getitem__(self, idx):
        # پیدا کردن فایل و ایندکس محلی
        idx_file = np.searchsorted(self.cum_lengths, idx, side="right")
        start = 0 if idx_file == 0 else self.cum_lengths[idx_file-1]
        local_idx = idx - start

        self._open_file(idx_file)

        x = self._x["X"][local_idx] # type: ignore
        y = int(self._y["y"][local_idx]) # type: ignore

        #x = torch.from_numpy(x).to(self.dtype)
        #y = torch.tensor(y, dtype=torch.long)
        x = torch.from_numpy(x).to(self.dtype)
        y = torch.tensor(int(self._z["y"][local_idx]), dtype=torch.long) # type: ignore
        
        return x, y

'''
from torch.utils.data import DataLoader
files = [f for f in os.listdir(folder) if f.endswith(".npz")]
print(files)
print(len(files))

files =[f"/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/X_all_end{i}.npy" for i in [21, 42 , 63, 84]]
files_y =[f"/home/asr/mohammadBalaghi/dataset_signal/newdatahaag1/y_all_end{i}.npy" for i in [21, 42 , 63, 84]]
file_val = files[:1]
file_val_y = files_y[:1]
file_train = files[1:]
file_train_y = files_y[1:]
print('start create dataset')
dataset_train = MultiNpzDataset(file_train, file_train_y)
dataset_val = MultiNpzDataset(file_val,file_val_y)
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
'''
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
'''