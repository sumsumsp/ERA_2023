import torch.nn as nn
import torch.nn.functional as F


class Model_S10(nn.Module):
    def __init__(self, dropout=0.10):
        super(Model_S10, self).__init__()
        ## Convolution Block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()  #32*32* 3 || 32*32*64
        )
        
        # Layer 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),  padding=1, bias=False), #32*32* 128
            nn.MaxPool2d(2, 2),  #16*16*128
            nn.BatchNorm2d(128), 
            nn.ReLU()
        )
        
        # Resnetblock
        self.R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128), #16*16*128
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128), #16*16*128
            nn.ReLU()
        )
        
        # Layer 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),# 8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Layer 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512), #8*8*512
            nn.ReLU()
        )
        
        # R2
        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512), #8*8*512
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(), #8*8*512
            nn.BatchNorm2d(512) 
        )
        
        # Maxpool
        self.pool3 = nn.MaxPool2d(4, 4) #2*2*
        
        # Fully connected
        self.fc = nn.Linear(2048, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x1 = self.conv2(x)
        x2 = self.R1(x1) * x1
        x = x1 + x2
        
        x = self.conv3(x)
        
        x3 = self.conv4(x)
        x4 = self.R2(x3) * x3
        x = x4 + x3
        
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)