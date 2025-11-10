import torch
import torch.nn as nn

def basic_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return block

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x    

class Recurrent_block(nn.Module):
    def __init__(self, in_channels, out_channels, t = 2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, t = 2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(in_channels, out_channels,t = t),
            Recurrent_block(in_channels, out_channels,t = t)
        )
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x*psi

class R2AttU_Net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, t = 5):
        super(R2AttU_Net,self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(in_channels = in_channels, out_channels = 64, t = t)
        self.RRCNN2 = RRCNN_block(in_channels = 64, out_channels = 128, t = t)
        self.RRCNN3 = RRCNN_block(in_channels = 128, out_channels = 256, t = t)
        self.RRCNN4 = RRCNN_block(in_channels = 256, out_channels = 512, t = t)
        self.RRCNN5 = RRCNN_block(in_channels = 512, out_channels = 1024, t = t)
        
            
        self.up5 = UpConv(in_channels = 1024, out_channels = 512)
        self.att5 = AttentionGate(F_g = 512, F_l = 512, F_int = 256)
        self.up_RRCNN5 = RRCNN_block(in_channels = 1024, out_channels = 512, t = t)
    
        self.up4 = UpConv(in_channels = 512, out_channels = 256)
        self.att4 = AttentionGate(F_g = 256, F_l = 256, F_int = 128)
        self.up_RRCNN4 = RRCNN_block(in_channels = 512, out_channels = 256, t = t)

        self.up3 = UpConv(in_channels = 256, out_channels = 128)
        self.att3 = AttentionGate(F_g = 128, F_l = 128, F_int = 64)
        self.up_RRCNN3 = RRCNN_block(in_channels = 256, out_channels = 128, t = t)

        self.up2 = UpConv(in_channels = 128, out_channels = 64)
        self.att2 = AttentionGate(F_g = 64, F_l = 64, F_int = 32)
        self.up_RRCNN2 = RRCNN_block(in_channels = 128, out_channels = 64, t = t)
    
        self.conv_1x1 = nn.Conv2d(64, out_channels, kernel_size = 1, stride = 1, padding=0)
        
    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.max_pool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.max_pool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.max_pool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.max_pool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        x4 = self.att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.up_RRCNN5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_RRCNN4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_RRCNN3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_RRCNN2(d2)

        d1 = self.conv_1x1(d2)

        return d1