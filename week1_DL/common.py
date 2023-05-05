import torch
import torch.nn as nn

'''
이 소스는 실제 많이 이용되는 모듈 중 일부를 다룸.

이 모듈을 이용해서 네트워크를 구성하는 다양한 실험을 해보기.
우수한 성능을 만드는 네트워크 찾아보기.

'''

# padding 모듈
def autopad(k, p=None, d=1):  # kernel, padding, dilation   # dilation은 element 간격을 의미함
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation: SiLU 함수
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# Residual style block
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1) # kernel size 수정 (ResNet18)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2    # shortcut=True 이고 입출력 채널 크기 같으면 add=True

    def forward(self, x):
        # add==True면 element-wise add
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# conv 3개 쓰는 방식
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

#
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h) # batch_size, output_channel, w, h

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

# yolov5에서 제안되어 엄청난 성능 향상
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)

        return x
    
# ------------------------------------------------------------------------
# VGG16
class VGG16(nn.Module):
    def __init__(self, base_dim=64, num_classes=10):
        super(VGG16, self).__init__()
        
        self.feature = nn.Sequential(
            Conv(1, base_dim, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim, base_dim, k=3, s=1, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, padding=0),
            
            Conv(base_dim, base_dim*2, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*2, base_dim*2, k=3, s=1, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, padding=0),
            
            Conv(base_dim*2, base_dim*4, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*4, base_dim*4, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*4, base_dim*4, k=3, s=1, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, padding=0),
            
            Conv(base_dim*4, base_dim*8, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*8, base_dim*8, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*8, base_dim*8, k=3, s=1, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, padding=0),
            
            Conv(base_dim*8, base_dim*8, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*8, base_dim*8, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*8, base_dim*8, k=3, s=1, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, padding=0),
        )
        
        # FC Layer 3개층
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=base_dim*8*1*1, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(in_features=4096, out_features=num_classes, bias=True),
        )
        
    def forward(self, x):
        x = self.feature(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)         # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 80
        #print(x.shape)
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)

        return x

# ResNet: BottleNeck 블럭 활용
class ResNet(nn.Module):
    def __init__(self, base_dim=64, num_classes=10):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            Conv(1, base_dim, k=7, s=2, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = Bottleneck(base_dim, base_dim*2)
        self.conv3 = Bottleneck(base_dim*2, base_dim*4)
        self.conv4 = Bottleneck(base_dim*4, base_dim*8)
        self.conv5 = Bottleneck(base_dim*8, base_dim*16)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        # FC Layer
        self.fc_layer = nn.Linear(in_features=base_dim*16, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)         # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 80
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)

        return x



# practice 1 (VGG modifying)
class miniVGG(nn.Module):
    def __init__(self, base_dim=8, num_classes=10):
        super(miniVGG, self).__init__()
        
        self.base_dim = base_dim
        self.feature = nn.Sequential(
            Conv(1, base_dim, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim, base_dim, k=3, s=1, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, padding=0),
            
            Conv(base_dim, base_dim*2, k=3, s=1, p=1, act=nn.ReLU()),
            Conv(base_dim*2, base_dim*2, k=3, s=1, p=1, act=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, padding=0),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=base_dim*2*7*7, out_features=num_classes, bias=True)
        )
        
    def forward(self, x):
        x = self.feature(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)         # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 base_dim*2
        #print(x.shape)
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)
        
        return x

# practice 2
class miniResNet(nn.Module):
    def __init__(self, base_dim=8, num_classes=10):
        super(miniResNet, self).__init__()
        
        self.conv1 = Bottleneck(1, base_dim)
        self.conv2 = Bottleneck(base_dim, base_dim*2)
        self.avg_pool = nn.AvgPool2d(2)
        
        self.fc_layer = nn.Linear(in_features=base_dim*2*14*14, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)         # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 80
        
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)

        return x

# practice 3
class miniViT(nn.Module):
    def __init__(self, base_dim=8, num_classes=10):
        super(miniViT, self).__init__()
        self.conv1 = TransformerBlock(c1=1, c2=base_dim, num_heads=1, num_layers=2)
        self.pooling = nn.Linear(in_features=base_dim*28*28, out_features=num_classes, bias=True)
        self.fc_layer = nn.Linear(in_features=base_dim*28*28, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)  # return shape: (batch_size, output_channel, w, h)
        #print(x.shape)
        x = x.reshape(x.size(0), -1)     # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 80
        #print(x.shape)
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)

        return x


# practice 4
class miniBottleneckCSP(nn.Module):
    def __init__(self, base_dim=8, num_classes=10):
        super(miniBottleneckCSP, self).__init__()
        
        self.conv1 = BottleneckCSP(1, base_dim)
        self.conv2 = BottleneckCSP(base_dim, base_dim*2)
        self.avg_pool = nn.AvgPool2d(2)
        
        self.fc_layer = nn.Linear(in_features=base_dim*2*14*14, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)         # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 80
        
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)

        return x
    
# practice 4
class miniBottleneckCSP(nn.Module):
    def __init__(self, base_dim=8, num_classes=10):
        super(miniBottleneckCSP, self).__init__()
        
        self.conv1 = BottleneckCSP(1, base_dim)
        self.conv2 = BottleneckCSP(base_dim, base_dim*2)
        self.max_pool = nn.MaxPool2d(2)
        
        self.fc_layer = nn.Linear(in_features=base_dim*2*14*14, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)         # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 80
        
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)

        return x
    
    
# practice 5
class miniSPPF(nn.Module):
    def __init__(self, base_dim=8, num_classes=10):
        super(miniSPPF, self).__init__()
        
        self.conv1 = Conv(1, base_dim)
        self.conv2 = SPPF(base_dim, base_dim*2)
        self.conv3 = SPPF(base_dim*2, base_dim*4)
        self.max_pool = nn.MaxPool2d(2)
        self.fc_layer = nn.Linear(in_features=base_dim*4*14*14, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)         # 차원 변경: reshape(-1, 320)과 같음, x.size(0): 행 크기, 즉 80
        #print(x.shape)
        
        x = self.fc_layer(x)
        x = nn.Softmax(dim=1)(x)

        return x