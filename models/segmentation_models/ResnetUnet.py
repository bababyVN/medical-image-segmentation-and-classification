import torch
import torch.nn as nn
import torchvision.models as models

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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.basic_block = basic_block(in_channels, out_channels)
        self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, 2, 2)
    
    def forward(self, down, skip):
        x = self.up_sample(down)
        x = torch.cat([x, skip], dim=1)
        x = self.basic_block(x)
        return x
    
class ResNetUnet(nn.Module):
    def __init__(self, n_classes=1, freeze=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        self.maxpool = backbone.maxpool
        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4
        
        if freeze:
            self._freeze_backbone()
            
        self.decoder5 = DecoderBlock(2048 + 1024, 1024) 
        self.decoder4 = DecoderBlock(1024 + 512, 512)    
        self.decoder3 = DecoderBlock(512 + 256, 256)     
        self.decoder2 = DecoderBlock(256 + 64, 64)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def _freeze_backbone(self):
        layers = [self.encoder1, self.encoder2, self.encoder3, 
                         self.encoder4, self.encoder5]
        
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.maxpool(e1)
        e2 = self.encoder2(p1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        d5 = self.decoder5(e5, e4)  
        d4 = self.decoder4(d5, e3)  
        d3 = self.decoder3(d4, e2)  
        d2 = self.decoder2(d3, e1) 
        d1 = self.decoder1(d2)
        out = self.out(d1)
        
        return out
