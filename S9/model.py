import torch.nn as nn
import torch.nn.functional as F

class Model_S9(nn.Module):
    def __init__(self,dropout= 0.025):
        super(Model_S9, self).__init__()
                ## Convolution Block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias = False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, 3, groups=32, padding=1, bias = False), # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
            nn.Conv2d(64, 64, 1, padding=0, bias = False),   # Input: 16x16x32 | Output: 18x18x64 | RF: 13x13
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),
                    )
        
        ## Transition Block1
        self.trans1 = nn.Sequential(
            nn.Conv2d(64,16,3, padding=1, stride=2, dilation=2), # Input: 32x32x64 | Output: 16x16x16 | RF: 5x5
                    )

        ## Convolution Block2
        self.conv2 =  nn.Sequential(
            nn.Conv2d(16, 64, 3,  padding=1, bias = False), # Input: 16x16x32 | Output: 16x16x32 | RF: 9x9
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            ## Depthwise Seperable Convolution1
            nn.Conv2d(64,64, 3,  padding=1,groups=64 ,bias = False),  # Input: 16x16x32 | Output: 16x16x32 | RF: 13x13
            nn.Conv2d(64, 128, 1, padding=0, bias = False),   # Input: 16x16x32 | Output: 18x18x64 | RF: 13x13
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout),
        )
        
        #Transition Block2
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, stride=2, dilation=2), # Input: 18x18x32 | Output: 9x9x64 | RF: 13x13
            #nn.ReLU()
        )

        #Convolution Block3
        self.conv3 = nn.Sequential(
                        ## Dilation Block
            nn.Conv2d(32, 64, 3,  bias = False, padding=1), # Input: 9x9x64 | Output: 7x7x64 | RF: 29x29
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, 3,  bias = False, padding=1),  # Input: 7x7x64| Output: 7x7x64 | RF: 45x45
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout),
        )

        #Transition Block3
        self.trans3 = nn.Sequential(

            nn.Conv2d(128, 10, 1 ), # Input: 7x7x64| Output: 4x4x16 | RF: 61x61
            #nn.ReLU()
        )


        ## Output Block
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) 


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)

        x = self.conv2(x) 
        x = self.trans2(x) 

        x = self.conv3(x) 
        x = self.trans3(x)

        #x = self.conv4(x)
        x = self.gap(x)

        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)