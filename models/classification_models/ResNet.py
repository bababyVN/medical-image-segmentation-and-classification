import torch.nn as nn 

# Resnet18 use basicblock
# Resnet50 use bottleneck block as basicblock
# Resnet 18 layer

class BasicBlock(nn.Module):
    
    """
        Basic block in Resnet 18 
    """
    
    def __init__(self, input_channel, output_channel, stride = 1, padding=1  ) :
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.conv1 = nn.Conv2d(input_channel,output_channel, kernel_size=(3,3), stride=stride,
                     padding=1, bias=False )
        self.conv2 = nn.Conv2d(output_channel,output_channel, kernel_size=(3,3), stride=1,
                     padding=1, bias=False )
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        
        self.identity = nn.Sequential()
        if stride != 1 or self.input_channel != self.output_channel: # Identity have different dimension / channel 
            self.identity = nn.Sequential(
                nn.Conv2d(input_channel,output_channel, kernel_size=(1,1), stride=stride,
                   padding=0  , bias=False),
                nn.BatchNorm2d(output_channel)
            )
            
        
    def forward(self,x):
        identity = self.identity(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x += identity # Residual connection
        x = self.relu(x)
        return x 
     
class BottleNeckBlock(nn.Module):
    
    """
        Bottle neck block in Resnet 50 
    """
    
    def __init__(self, input_channel, output_channel, stride = 1, padding=1  ) :
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.conv1 = nn.Conv2d(input_channel,output_channel // 4, kernel_size=(1,1), stride=stride,
                     padding=0, bias=False )
        self.conv2 = nn.Conv2d(output_channel // 4,output_channel // 4, kernel_size=(3,3), stride=1,
                     padding=1, bias=False )
        self.conv3 =  nn.Conv2d(output_channel // 4, output_channel, kernel_size=(1,1), stride=1,
                                padding=0, bias= False)
        self.bn1 = nn.BatchNorm2d(output_channel // 4)
        self.bn2 = nn.BatchNorm2d(output_channel // 4)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        
        self.identity = nn.Sequential()
        if stride != 1 or self.input_channel != self.output_channel: # Identity have different dimension / channel 
            self.identity = nn.Sequential(
                nn.Conv2d(input_channel,output_channel, kernel_size=(1,1), stride=stride,
                   padding=0  , bias=False),
                nn.BatchNorm2d(output_channel)
            )
            
        
    def forward(self,x):
        identity = self.identity(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += identity # Residual connection
        x = self.relu(x)
        return x 
    
    
    
class ResNet18(nn.Module):
    
    def __init__(self, num_classes = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channel = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(stride=2,kernel_size= (3,3),padding= 1)
        
        self.layer1 = self.make_layer(BasicBlock,64,2,1)
        self.layer2 = self.make_layer(BasicBlock,128,2,2)
        self.layer3 = self.make_layer(BasicBlock,256,2,2)
        self.layer4 = self.make_layer(BasicBlock,512,2,2)
        
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512,self.num_classes)
        
    def make_layer(self, block, out_channel, numblocks, stride):
        layers = []
        strides = [stride] + [1] * (numblocks-1) # retain height width in the subsequent layers
        for stride in strides :
            layers.append(block(self.input_channel, out_channel, stride))
            self.input_channel = out_channel
        
        return nn.Sequential(*layers)
    
    def forward(self,x):

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
   
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x 


class ResNet50(nn.Module):
    """_summary_

        Resnet50 from the original paper
        
    """
    def __init__(self, num_classes = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channel = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(stride=2,kernel_size= (3,3),padding= 1)
        
        self.layer1 = self.make_layer(BottleNeckBlock,256,3,1)
        self.layer2 = self.make_layer(BottleNeckBlock,512,4,2)
        self.layer3 = self.make_layer(BottleNeckBlock,1024,6,2)
        self.layer4 = self.make_layer(BottleNeckBlock,2048,3,2)
        
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048,self.num_classes)
        
    def make_layer(self, block, out_channel, numblocks, stride):
        layers = []
        strides = [stride] + [1] * (numblocks-1) # retain height width in the subsequent layers
        for stride in strides :
            layers.append(block(self.input_channel, out_channel, stride))
            self.input_channel = out_channel
        
        return nn.Sequential(*layers)
    
    def forward(self,x):

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
   
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x 