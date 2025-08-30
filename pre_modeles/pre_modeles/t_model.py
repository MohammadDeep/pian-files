'''
for all modelse :
    input signal 
    labeles 

for each signall:
    fs base
    window size
    output size
    step for all model


'''

import torch
import torch.nn as nn 
from typing import Optional, Callable
def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, groups=groups, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """
    یک بلوک استاندارد ResNet (نوع BasicBlock مثل ResNet-18/34)
    """
    expansion = 1  # برای BasicBlock خروجی هم‌بعد با planes است

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        hava_maxpool:bool = True
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock فقط با groups=1 و base_width=64 پشتیبانی می‌شود.")

        # لایه‌های مسیر اصلی
        self.conv1 = conv3x3(in_channels, planes, stride, groups, dilation)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, groups, dilation)
        self.hava_maxpool = hava_maxpool
        if hava_maxpool:
            self.maxpool = nn.MaxPool1d(2)
        self.bn2   = norm_layer(planes)

        # مسیر میان‌بُر (skip). اگر ابعاد/استراید فرق کند، ۱×۱ می‌گذاریم
        if downsample is None and (stride != 1 or in_channels != planes * self.expansion or hava_maxpool):
            if hava_maxpool:
                n = 2
            else:
                n =1
            downsample = nn.Sequential(
                conv1x1(in_channels,  planes * self.expansion, n * stride),
                norm_layer(planes * self.expansion),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.hava_maxpool:
         out = self.maxpool(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


import torch
import torch.nn as nn
from typing import Type, List

# فرض: BasicBlock شما همین فایل است و امضایش با کدی که فرستادید یکی است.

class SimpleResNet(nn.Module):
    def __init__(self,
                 block: Type[BasicBlock],
                 layers: List[int],          # تعداد بلوک‌ها در هر stage، مثل [1,1,1] یا [2,2,2]
                 list_step:List[int],
                 in_ch: int = 3,
                 base_planes: int = 64):
        super().__init__()

        # 1) Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_planes, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(base_planes),
            nn.ReLU(inplace=True),
        )
        self.inplanes = base_planes  # تعداد کانال فعلی بعد از stem

        # 2) Stages (فقط بلوک اول هر stage می‌تواند stride>1 داشته باشد)
        self.layer1 = self._make_layer(block, planes=base_planes,   blocks=layers[0], stride=list_step[0])
        self.layer2 = self._make_layer(block, planes=base_planes*2, blocks=layers[1], stride=list_step[1])
        self.layer3 = self._make_layer(block, planes=base_planes*4, blocks=layers[2], stride=list_step[2])
        self.layer4 = self._make_layer(block, planes=base_planes*8, blocks=layers[3], stride=list_step[3])
        self.layer5 = self._make_layer(block, planes=base_planes*16, blocks=layers[4], stride=list_step[4])
      

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int, poolMax:bool = True) -> nn.Sequential:
        layers = []
        # بلوک اول stage
        layers.append(
            block(in_channels=self.inplanes,
                  planes=planes,
                  stride=stride,
                  norm_layer=nn.BatchNorm1d,
                  hava_maxpool=poolMax)       # برای پرهیز از mismatch، خاموش نگه‌دار
        )
        self.inplanes = planes * block.expansion

        # بلوک‌های بعدی همان stage
        for _ in range(1, blocks):
            layers.append(
                block(in_channels=self.inplanes,
                      planes=planes,
                      stride=1,
                      norm_layer=nn.BatchNorm1d,
                      hava_maxpool=False)
            )
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # [B, base_planes, H, W]
        x = self.layer1(x)   # stage1: رزولوشن ثابت
        x = self.layer2(x)   # stage2: H,W نصف
        x = self.layer3(x)   # stage3: دوباره نصف
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class SimpleResNet1(nn.Module):
    def __init__(self,
                 block: Type[BasicBlock],
                 layers: List[int], 
                 list_step :List[int],  
                 in_ch: int = 3,
                 
                 base_planes: int = 64):
        super().__init__()

        # 1) Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_planes, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(base_planes),
            nn.ReLU(inplace=True),
        )
        self.inplanes = base_planes  # تعداد کانال فعلی بعد از stem

        # 2) Stages (فقط بلوک اول هر stage می‌تواند stride>1 داشته باشد)
        self.layer1 = self._make_layer(block, planes=base_planes,   blocks=layers[0], stride=list_step[0])
        self.layer2 = self._make_layer(block, planes=base_planes*2, blocks=layers[1], stride=list_step[1])
        self.layer3 = self._make_layer(block, planes=base_planes*4, blocks=layers[2], stride=list_step[2])
        self.layer4 = self._make_layer(block, planes=base_planes*8, blocks=layers[3], stride=list_step[3])
        self.layer5 = self._make_layer(block, planes=base_planes*16, blocks=layers[4], stride=list_step[4])
        self.layer6 = self._make_layer(block, planes=base_planes*32, blocks=layers[5], stride=list_step[5])
        self.layer7 = self._make_layer(block, planes=base_planes*64, blocks=layers[6], stride=list_step[6])
      

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int, poolMax:bool = True) -> nn.Sequential:
        layers = []
        # بلوک اول stage
        layers.append(
            block(in_channels=self.inplanes,
                  planes=planes,
                  stride=stride,
                  norm_layer=nn.BatchNorm1d,
                  hava_maxpool=poolMax)       # برای پرهیز از mismatch، خاموش نگه‌دار
        )
        self.inplanes = planes * block.expansion

        # بلوک‌های بعدی همان stage
        for _ in range(1, blocks):
            layers.append(
                block(in_channels=self.inplanes,
                      planes=planes,
                      stride=1,
                      norm_layer=nn.BatchNorm1d,
                      hava_maxpool=False)
            )
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # [B, base_planes, H, W]
        x = self.layer1(x)   # stage1: رزولوشن ثابت
        x = self.layer2(x)   # stage2: H,W نصف
        x = self.layer3(x)   # stage3: دوباره نصف
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x
    
class SimpleResNet2(nn.Module):
    def __init__(self,
                 block: Type[BasicBlock],
                 layers: List[int],          # تعداد بلوک‌ها در هر stage، مثل [1,1,1] یا [2,2,2]
                 list_step:List[int],
                 in_ch: int = 3,
                 base_planes: int = 64):
        super().__init__()

        # 1) Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_planes, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(base_planes),
            nn.ReLU(inplace=True),
        )
        self.inplanes = base_planes  # تعداد کانال فعلی بعد از stem

        # 2) Stages (فقط بلوک اول هر stage می‌تواند stride>1 داشته باشد)
        self.layer1 = self._make_layer(block, planes=base_planes,   blocks=layers[0], stride=list_step[0],poolMax = False)
        self.layer2 = self._make_layer(block, planes=base_planes*2, blocks=layers[1], stride=list_step[1],poolMax = False)
        self.layer3 = self._make_layer(block, planes=base_planes*4, blocks=layers[2], stride=list_step[2],poolMax = False)
        self.layer4 = self._make_layer(block, planes=base_planes*8, blocks=layers[3], stride=list_step[3],poolMax = False)
        self.layer5 = self._make_layer(block, planes=base_planes*16, blocks=layers[4], stride=list_step[4],poolMax = False)
        self.layer6 = self._make_layer(block, planes=base_planes*32, blocks=layers[5], stride=list_step[5],poolMax = False)
        self.layer7 = self._make_layer(block, planes=base_planes*32, blocks=layers[6], stride=list_step[6],poolMax = False)
        self.layer8 = self._make_layer(block, planes=base_planes*32, blocks=layers[7], stride=list_step[7],poolMax = False)
        self.layer9 = self._make_layer(block, planes=base_planes*64, blocks=layers[8], stride=list_step[8],poolMax = False)
        self.layer10= self._make_layer(block, planes=base_planes*64, blocks=layers[9], stride=list_step[9],poolMax = False)
        self.layer11= self._make_layer(block, planes=base_planes*64, blocks=layers[10],stride=list_step[10],poolMax = False)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int, poolMax:bool = True) -> nn.Sequential:
        layers = []
        # بلوک اول stage
        layers.append(
            block(in_channels=self.inplanes,
                  planes=planes,
                  stride=stride,
                  norm_layer=nn.BatchNorm1d,
                  hava_maxpool=poolMax)       # برای پرهیز از mismatch، خاموش نگه‌دار
        )
        self.inplanes = planes * block.expansion

        # بلوک‌های بعدی همان stage
        for _ in range(1, blocks):
            layers.append(
                block(in_channels=self.inplanes,
                      planes=planes,
                      stride=1,
                      norm_layer=nn.BatchNorm1d,
                      hava_maxpool=False)
            )
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # [B, base_planes, H, W]
        x = self.layer1(x)   # stage1: رزولوشن ثابت
        x = self.layer2(x)   # stage2: H,W نصف
        x = self.layer3(x)   # stage3: دوباره نصف
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class LSTM_Model(nn.Module):
    def __init__(self, 
                 input_size: int,
                   hidden_size: int,
                     num_layers: int, 
                     num_classes: int):
        super(LSTM_Model, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hn, cn) = self.lstm(x)
        
        # استفاده از آخرین hidden state برای پیش‌بینی
        out = self.fc(hn[-1])  # hn[-1] آخرین hidden state از آخرین لایه LSTM
        return out


class CNN_LSTM_Model(nn.Module):
    def __init__(self, cnn_model: SimpleResNet, lstm_model: LSTM_Model):
        super(CNN_LSTM_Model, self).__init__()
        self.cnn = cnn_model
        self.lstm = lstm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. استخراج ویژگی با CNN
        # ورودی x: [batch_size, in_channels, sequence_length]
        features = self.cnn(x)
        # خروجی features: [batch_size, cnn_output_channels, cnn_output_length]
        
        # 2. آماده‌سازی برای LSTM
        # نیاز به تغییر ابعاد از [B, C, L] به [B, L, C]
        features = features.permute(0, 2, 1)
        
        # 3. طبقه‌بندی با LSTM
        output = self.lstm(features)
        # خروجی output: [batch_size, num_classes]
        
        return output
    


class CNN_LSTM_Model1(nn.Module):
    def __init__(self, cnn_model: SimpleResNet1, lstm_model: LSTM_Model):
        super(CNN_LSTM_Model1, self).__init__()
        self.cnn = cnn_model
        self.lstm = lstm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. استخراج ویژگی با CNN
        # ورودی x: [batch_size, in_channels, sequence_length]
        features = self.cnn(x)
        # خروجی features: [batch_size, cnn_output_channels, cnn_output_length]
        
        # 2. آماده‌سازی برای LSTM
        # نیاز به تغییر ابعاد از [B, C, L] به [B, L, C]
        features = features.permute(0, 2, 1)
        
        # 3. طبقه‌بندی با LSTM
        output = self.lstm(features)
        # خروجی output: [batch_size, num_classes]
        
        return output
# --- نمونهٔ استفاده ---
if __name__ == "__main__":
    N  = 256*8 * 1+ 0* 256//32
    '''
    start lne  = 2 * 256
    step  = 2 * 256
    '''
    net1 = SimpleResNet(BasicBlock, layers=[1,1,1,2,2],list_step = [2,2,2,1,1], in_ch=3, base_planes=16)
    lstm1 = LSTM_Model(input_size = 256,
                   hidden_size = 2,
                     num_layers = 2, 
                     num_classes = 3)
    model1 = CNN_LSTM_Model(net1, lstm1)
    
    '''
    start lne  = 1 * 256
    step  = 1 * 256
    '''
    net2 = SimpleResNet(BasicBlock, layers=[1,1,1,2,2],list_step = [2,2,1,1,1], in_ch=3, base_planes=16)
    lstm2 = LSTM_Model(input_size = 256,
                   hidden_size = 2,
                     num_layers = 2, 
                     num_classes = 3)
    model2 = CNN_LSTM_Model(net2, lstm2)
  
    '''
    start lne  = .5 * 256
    step  = .5 * 256
    '''
    net3 = SimpleResNet(BasicBlock, layers=[1,1,1,2,2],list_step = [2,1,1,1,1], in_ch=3, base_planes=16)
    lstm3 = LSTM_Model(input_size = 256,
                   hidden_size = 2,
                     num_layers = 2, 
                     num_classes = 3)
    model3 = CNN_LSTM_Model(net3, lstm3)

    '''
    start lne  = 1 * 256
    step  = 1 * 256
    '''

    net4 = SimpleResNet1(BasicBlock, layers=[1,1,1,2,2,2,2],list_step = [2,1,1,1,1,1,1], in_ch=3, base_planes=4)
    lstm4 = LSTM_Model(input_size = 256,
                   hidden_size = 2,
                     num_layers = 2, 
                     num_classes = 3)
    model4 = CNN_LSTM_Model(net4, lstm4)

    '''
    start lne  = 4 * 256
    step  = 4 * 256
    '''
    net5 = SimpleResNet1(BasicBlock, layers=[2,2,2,2,2,2,2],list_step = [2,2,1,1,1,1,1], in_ch=3, base_planes=4)
    lstm5 = LSTM_Model(input_size = 256,
                   hidden_size = 2,
                     num_layers = 2, 
                     num_classes = 3)
    model5 = CNN_LSTM_Model(net5, lstm5)
  
    '''
    start lne  = 2* 256
    step  = 2 * 256
    '''
    net6 = SimpleResNet1(BasicBlock, layers=[2,2,2,2,2,2,2],list_step = [2,1,1,1,1,1,1], in_ch=3, base_planes=4)
    lstm6 = LSTM_Model(input_size = 256,
                   hidden_size = 2,
                     num_layers = 2, 
                     num_classes = 3)
    model6 = CNN_LSTM_Model(net6, lstm6)

    '''
    start lne  =   256//32
    step  = 256//32
    '''
    net7 = SimpleResNet2(BasicBlock, layers=[1,1,1,1,1,1,1,1,1,1,1,1],list_step = [2,2,1,1,1,1,1,1,1,1,1], in_ch=3, base_planes=4) 
    lstm7 = LSTM_Model(input_size = 64,
                   hidden_size = 2,
                     num_layers = 2, 
                     num_classes = 3)
    model7 = CNN_LSTM_Model(net7, lstm7)





    # ورودی به مدل باید یک تنسور باشد
    dummy_input = torch.randn(32, 3, N)  # [batch_size, channels, sequence_length]
    with torch.no_grad():
        output = model1(dummy_input)
    print(output.size())

    from torchinfo import summary
    summary(net1, input_size=(32, 3, N)) 