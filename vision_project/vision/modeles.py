import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
import torch
from torchvision import models




class HumanPresenceSqueezeNet(nn.Module):
    def __init__(self,
                 pretrained=True,
                 weights=models.SqueezeNet1_1_Weights.DEFAULT,
                 freeze_features_up_to=10):
        super().__init__()
        # 1) بارگذاری backbone با API جدید weights
        weights = models.SqueezeNet1_1_Weights.DEFAULT
        base = models.squeezenet1_1(weights=weights)

        # 2) ساخت classifier جدید شامل Dropout → Conv2d → Conv2d → Flatten
        #    خروجی نهایی پس از Flatten می‌شود (B, 1)
        self.classifier = nn.Sequential(
            base.classifier[0],                             # Dropout
            nn.Conv2d(512, 1, kernel_size=1, stride=1),     # جایگزینی لایه
            nn.Conv2d(1, 1, kernel_size=13, stride=1),      # همانی که قبل داشتید
            nn.Flatten()                                     # <<< این‌جا flatten!
        )

        # 3) فریز کردن لایه‌های ابتدایی features
        self.features = base.features
        for p in list(self.features.parameters())[:freeze_features_up_to]:
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)      # (B,512,13,13)
        x = self.classifier(x)    # (B,1) پس از Flatten
        return x






class InceptionModule_save_D(nn.Module):
    def __init__(self, in_channels
                 , out_1_3,
                 out_1_5 ,
                 out_1_7
                 , out_2_3 
                 , out_2_5 
                 ,out_2_7
                 ,p_dropout = 0.5
                 ):
        super(InceptionModule_save_D, self).__init__()
        out = (out_2_3+ out_2_5 +out_2_7)
        # شاخه اول: کانولوشن 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_3, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_3),
            nn.ReLU6(inplace= True),
           # nn.MaxPool2d(2),
            nn.Conv2d(out_1_3 ,out_2_3, kernel_size = 3, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_3),

        )

        # شاخه دوم: کانولوشن 1x1 و سپس 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_5, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_5),
            nn.ReLU6(inplace= True),
            #nn.MaxPool2d(2),
            nn.Conv2d(out_1_5 ,out_2_5, kernel_size = 5, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_5),

        )

        # شاخه سوم: کانولوشن 1x1 و سپس 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_7, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_7),
            nn.ReLU6(inplace= True),
            #nn.MaxPool2d(2),
            nn.Conv2d(out_1_7 ,out_2_7, kernel_size = 7, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_7),

        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out, kernel_size=1,stride=1, bias= False)
        )

        self.relu = nn.ReLU6(inplace= True)
        self.dropout = nn.Dropout2d(p = p_dropout)
    def forward(self, x):
        # محاسبه خروجی هر شاخه
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out4], 1) + torch.cat([out1, out2, out3], 1)
        out = self.relu(out)
        out = self.dropout(out)
        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return out


class InceptionModule(nn.Module):
    def __init__(self, in_channels
                 , out_1_3,
                 out_1_5 ,
                 out_1_7
                 , out_2_3 
                 , out_2_5 
                 ,out_2_7
                 ,p_dropout = 0.5
                 ):
        super(InceptionModule, self).__init__()
        out = (out_2_3+ out_2_5 +out_2_7)
        # شاخه اول: کانولوشن 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_3, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_3),
            nn.ReLU6(inplace= True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_1_3 ,out_2_3, kernel_size = 3, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_3),

        )

        # شاخه دوم: کانولوشن 1x1 و سپس 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_5, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_5),
            nn.ReLU6(inplace= True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_1_5 ,out_2_5, kernel_size = 5, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_5),

        )

        # شاخه سوم: کانولوشن 1x1 و سپس 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_1_7, kernel_size=1,padding='same', bias= False),
            nn.BatchNorm2d(num_features = out_1_7),
            nn.ReLU6(inplace= True),
            nn.MaxPool2d(2),
            nn.Conv2d(out_1_7 ,out_2_7, kernel_size = 7, padding='same',bias = False),
            nn.BatchNorm2d(num_features = out_2_7),

        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out, kernel_size=1,stride=2, bias= False)
        )

        self.relu = nn.ReLU6(inplace= True)
        self.dropout = nn.Dropout2d(p = p_dropout)
    def forward(self, x):
        # محاسبه خروجی هر شاخه
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out4], 1) + torch.cat([out1, out2, out3], 1)
        out = self.relu(out)
        out = self.dropout(out)
        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return out

class Model_4(nn.Module):
    def __init__(self,classes_number = 1,  in_channels = 3):
        super(Model_4, self).__init__()

        self.layer_1 = InceptionModule(
            in_channels = in_channels
            ,out_1_3 = 2
            ,out_2_3 = 16
            ,out_1_5 = 2
            ,out_2_5 = 16
            ,out_1_7 = 2
            ,out_2_7 = 16
            ,p_dropout = 0
        )
        self.layer_2 = InceptionModule(
            in_channels = 16 * 3
            ,out_1_3 = 16
            ,out_2_3 = 32
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_3 = InceptionModule(
            in_channels = 32 * 3
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 32
            ,out_2_5 = 64
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )

        self.layer_4 = InceptionModule(
            in_channels = 32  + 64 * 2
            ,out_1_3 = 32
            ,out_2_3 = 128
            ,out_1_5 = 32
            ,out_2_5 = 128
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )

        self.layer_5 = InceptionModule(
            in_channels = 32  + 128 * 2
            ,out_1_3 = 64
            ,out_2_3 = 256
            ,out_1_5 = 64
            ,out_2_5 = 256
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = .10
        )
        self.layer_6 = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(256* 2 + 32, classes_number)
        )


    def forward(self, x):
        # محاسبه خروجی هر شاخه
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return x



class Model_6(nn.Module):
    def __init__(self,classes_number = 1,  in_channels = 3):
        super(Model_6, self).__init__()

        self.layer_1 = InceptionModule(
            in_channels = in_channels
            ,out_1_3 = 2
            ,out_2_3 = 16
            ,out_1_5 = 2
            ,out_2_5 = 16
            ,out_1_7 = 2
            ,out_2_7 = 16
            ,p_dropout = 0
        )
        self.layer_2 = InceptionModule(
            in_channels = 16 * 3
            ,out_1_3 = 16
            ,out_2_3 = 32
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 16
            ,out_2_7 = 16,
            p_dropout = 0
        )
        self.layer_3 = InceptionModule(
            in_channels = 32 * 2 + 16
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 32
            ,out_2_5 = 64
            ,out_1_7 = 16
            ,out_2_7 = 16,
            p_dropout = 0
        )

        self.layer_4 = InceptionModule(
            in_channels = 16  + 64 * 2
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 32
            ,out_2_5 = 64
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )

        self.layer_5 = InceptionModule(
            in_channels = 32  + 64 * 2
            ,out_1_3 = 64
            ,out_2_3 = 64
            ,out_1_5 = 64
            ,out_2_5 = 64
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_6 = InceptionModule(
            in_channels =64* 2 + 32
            ,out_1_3 = 64
            ,out_2_3 = 80
            ,out_1_5 = 64
            ,out_2_5 = 80
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_7 = InceptionModule(
            in_channels =80 * 2 + 32
            ,out_1_3 = 64
            ,out_2_3 = 128
            ,out_1_5 = 64
            ,out_2_5 = 128
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )

        self.layer_8 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(128* 2 + 32, classes_number)
        )


    def forward(self, x):
        # محاسبه خروجی هر شاخه
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)

        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return x



class Model_7(nn.Module):
    def __init__(self,classes_number = 1,  in_channels = 3):
        super(Model_7, self).__init__()

        self.layer_1 = InceptionModule(
            in_channels = in_channels
            ,out_1_3 = 3
            ,out_2_3 = 16
            ,out_1_5 = 3
            ,out_2_5 = 16
            ,out_1_7 = 3
            ,out_2_7 = 16
            ,p_dropout = 0
        )
        self.layer_2 = InceptionModule(
            in_channels = 16 * 3
            ,out_1_3 = 16
            ,out_2_3 = 32
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 8
            ,out_2_7 = 16,
            p_dropout = 0
        )
        self.layer_3 = InceptionModule(
            in_channels = 32 * 2 + 16
            ,out_1_3 = 20
            ,out_2_3 = 40
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 8
            ,out_2_7 = 16,
            p_dropout = 0
        )

        self.layer_4 = InceptionModule(
            in_channels = 40 + 32 + 16
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 16
            ,out_2_7 = 16,
            p_dropout = 0
        )

        self.layer_5 = InceptionModule(
            in_channels = 64 + 32 + 16
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 20
            ,out_2_5 = 40
            ,out_1_7 = 8
            ,out_2_7 = 16,
            p_dropout = 0
        )
        self.layer_6 = InceptionModule_save_D(
            in_channels =64 + 40 + 16
            ,out_1_3 = 32
            ,out_2_3 = 80
            ,out_1_5 = 32
            ,out_2_5 = 60
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_7 = InceptionModule(
            in_channels =80 + 60 + 32
            ,out_1_3 = 32
            ,out_2_3 = 100
            ,out_1_5 = 32
            ,out_2_5 = 100
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_8 = InceptionModule_save_D(
            in_channels =100 * 2 + 32
            ,out_1_3 = 32
            ,out_2_3 = 128
            ,out_1_5 = 32
            ,out_2_5 = 128
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_9 = InceptionModule(
            in_channels =128 * 2 + 32
            ,out_1_3 = 32
            ,out_2_3 = 200
            ,out_1_5 = 32
            ,out_2_5 = 200
            ,out_1_7 = 16
            ,out_2_7 = 40,
            p_dropout = 0
        )

        self.layer_10 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(200* 2 + 40, classes_number)
        )


    def forward(self, x):
        # محاسبه خروجی هر شاخه
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        x = self.layer_10(x)

        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return x





class Model_8(nn.Module):
    def __init__(self,classes_number = 1,  in_channels = 3):
        super(Model_8, self).__init__()

        self.layer_1 = InceptionModule(
            in_channels = in_channels
            ,out_1_3 = 3
            ,out_2_3 = 16
            ,out_1_5 = 3
            ,out_2_5 = 16
            ,out_1_7 = 3
            ,out_2_7 = 16
            ,p_dropout = 0.2
        )
        self.layer_2 = InceptionModule(
            in_channels = 16 * 3
            ,out_1_3 = 16
            ,out_2_3 = 32
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 8
            ,out_2_7 = 16,
            p_dropout = 0.2
        )
        self.layer_3 = InceptionModule(
            in_channels = 32 * 2 + 16
            ,out_1_3 = 20
            ,out_2_3 = 40
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 8
            ,out_2_7 = 16,
            p_dropout = 0.2
        )

        self.layer_4 = InceptionModule(
            in_channels = 40 + 32 + 16
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 16
            ,out_2_5 = 32
            ,out_1_7 = 16
            ,out_2_7 = 16,
            p_dropout = 0
        )

        self.layer_5 = InceptionModule(
            in_channels = 64 + 32 + 16
            ,out_1_3 = 32
            ,out_2_3 = 64
            ,out_1_5 = 20
            ,out_2_5 = 40
            ,out_1_7 = 8
            ,out_2_7 = 16,
            p_dropout = 0
        )
        self.layer_6 = InceptionModule_save_D(
            in_channels =64 + 40 + 16
            ,out_1_3 = 32
            ,out_2_3 = 128
            ,out_1_5 = 32
            ,out_2_5 = 128
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_7 = InceptionModule(
            in_channels =128 * 2+ 32
            ,out_1_3 = 32
            ,out_2_3 = 128
            ,out_1_5 = 32
            ,out_2_5 = 128
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_8 = InceptionModule_save_D(
            in_channels =128 * 2 + 32
            ,out_1_3 = 64
            ,out_2_3 = 256
            ,out_1_5 = 64
            ,out_2_5 = 256
            ,out_1_7 = 16
            ,out_2_7 = 32,
            p_dropout = 0
        )
        self.layer_9 = InceptionModule(
            in_channels =256 * 2 + 32
            ,out_1_3 = 64
            ,out_2_3 = 256
            ,out_1_5 = 64
            ,out_2_5 = 256
            ,out_1_7 = 16
            ,out_2_7 = 64,
            p_dropout = 0
        )

        self.layer_10 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(256* 2 + 64, classes_number)
        )


    def forward(self, x):
        # محاسبه خروجی هر شاخه
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        x = self.layer_10(x)

        # اتصال خروجی‌ها در امتداد بعد کانال (dim=1)
        return x


dic_model_2 = {
    'model_name' : 'model_2',
    'model_stucher' : HumanPresenceSqueezeNet,
    'input_shape' : (224, 224),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
    'use_sigmoid' : True # in test_step and train step for accurcy 
}



dic_model_3 = {
    'model_name' : 'model_2',
    'model_stucher' : HumanPresenceSqueezeNet,
    'input_shape' : (224, 224),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
    'use_sigmoid' : True 
}


dic_model_4 = {
    'model_name' : 'Model_4',
    'model_stucher' : Model_4,
    'input_shape' : (224, 224),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
    'use_sigmoid' : True 
}



dic_model_6 = {
    'model_name': 'Model_6',
    'model_stucher': Model_6,
    'input_shape' : (256, 256),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
    'use_sigmoid' : True 
}


dic_model_7 = {
    'model_name': 'Model_7',
    'model_stucher': Model_7,
    'input_shape' : (256, 256),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
    'use_sigmoid' : True 
}

dic_model_8 = {
    'model_name': 'Model_8',
    'model_stucher': Model_8,
    'input_shape' : (256, 256),
    'mean' : [0.485, 0.456, 0.406],
    'std' :[0.229, 0.224, 0.225],
    'use_sigmoid' : True 
}