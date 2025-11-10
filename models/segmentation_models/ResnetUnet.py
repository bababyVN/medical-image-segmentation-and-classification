import torch
import torch.nn as nn
import torchvision.models as models

def basic_block(in_channels, out_channels):
    # The first conv layer takes in_channels, and the output of the block is out_channels
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
    def __init__(self, in_channels_down, in_channels_skip, out_channels):
        super().__init__()
        
        # Calculate the channels after concatenation
        # This must match the size expected by the checkpoint: 2048 + 1024 = 3072 for d5
        concatenated_channels = in_channels_down + in_channels_skip
        
        # 1. Convolution Block: Takes the concatenated channels as input. (This logic is correct for the checkpoint.)
        self.basic_block = basic_block(concatenated_channels, out_channels) 
        
        # 2. Up-sampling: ConvTranspose2d takes the 'down' input channels
        # *** CRITICAL FIX APPLIED HERE: Up_sample output channels MUST match its input channels (in_channels_down)
        # to match the weight shape expected by the checkpoint, e.g., 2048 -> 2048. ***
        self.up_sample = nn.ConvTranspose2d(in_channels_down, in_channels_down, kernel_size=2, stride=2)


    def forward(self, down, skip):
        # 1. Upsample 'down'
        x = self.up_sample(down)
        
        # 2. Concatenate 'x' (upsampled) with 'skip' (encoder features)
        # x's channels (e.g., 2048) + skip's channels (e.g., 1024) = 3072. 
        # This matches the input size of the basic_block.
        x = torch.cat([x, skip], dim=1)
        
        # 3. Pass through the dual convolution block
        x = self.basic_block(x)
        return x
    
class ResNetUnet(nn.Module):
    def __init__(self, n_classes=1, freeze=True):
        super().__init__()
        # FIX: Using pretrained=True for backward compatibility
        backbone = models.resnet50(pretrained=True) 
            
        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        self.maxpool = backbone.maxpool
        self.encoder2 = backbone.layer1 # 256 channels
        self.encoder3 = backbone.layer2 # 512 channels
        self.encoder4 = backbone.layer3 # 1024 channels
        self.encoder5 = backbone.layer4 # 2048 channels
        
        if freeze:
            self._freeze_backbone()
            
        # Decoder structure based on ResNet50 channels:
        # Args: DecoderBlock(in_channels_down, in_channels_skip, out_channels_block)
        
        # d5: (e5, e4) -> (2048, 1024) -> Output: 1024. Total Cat Input: 3072
        self.decoder5 = DecoderBlock(in_channels_down=2048, in_channels_skip=1024, out_channels=1024)
        
        # d4: (d5, e3) -> (1024, 512) -> Output: 512. Total Cat Input: 1536
        self.decoder4 = DecoderBlock(in_channels_down=1024, in_channels_skip=512, out_channels=512)
        
        # d3: (d4, e2) -> (512, 256) -> Output: 256. Total Cat Input: 768
        self.decoder3 = DecoderBlock(in_channels_down=512, in_channels_skip=256, out_channels=256)
        
        # d2: (d3, e1) -> (256, 64) -> Output: 64. Total Cat Input: 320
        self.decoder2 = DecoderBlock(in_channels_down=256, in_channels_skip=64, out_channels=64)
        
        # Decoder 1: Upsamples D2 output (64 channels) to final size
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
        # Encoder
        e1 = self.encoder1(x) 
        p1 = self.maxpool(e1) 
        e2 = self.encoder2(p1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 
        e5 = self.encoder5(e4) 
        
        # Decoder 
        d5 = self.decoder5(e5, e4)  
        d4 = self.decoder4(d5, e3)  
        d3 = self.decoder3(d4, e2)  
        d2 = self.decoder2(d3, e1) 
        
        d1 = self.decoder1(d2)
        out = self.out(d1)
        
        return out