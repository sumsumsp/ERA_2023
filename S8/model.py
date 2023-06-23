
##S6 models 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self): 
        super(Net, self).__init__()
        
        self.conv1= nn.Sequential(
             nn.Conv2d (1, 6 ,3),
             nn.ReLU(),
             nn.BatchNorm2d(6),
             nn.Dropout2d(0.1),
             
             nn.Conv2d (6, 12 ,3),
             nn.ReLU(),
             nn.BatchNorm2d(12),
             nn.Dropout2d(0.1),
             
             nn.Conv2d (12, 24 ,3),
             nn.ReLU(),
             nn.BatchNorm2d(24),
             nn.Dropout2d(0.1)
        
        )
        
        self.trans1 = nn.Sequential(
             nn.Conv2d (24,12,1),
             nn.ReLU(),
             nn.BatchNorm2d(12),
             nn.AvgPool2d(2,2),

        )
        
        self.conv2= nn.Sequential(
             nn.Conv2d (12, 16 ,3),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Dropout2d(0.1),
             
             nn.Conv2d (16, 32 ,3),
             nn.ReLU(),
             nn.BatchNorm2d(32),
             nn.Dropout2d(0.1)
        
        )
            
        self.trans2 = nn.Sequential(
             nn.Conv2d (32,16,1),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.AvgPool2d(2,2))    
        
        self.fc = nn.Linear(144,10)
        
    def forward(self,x):
        x= self.conv1(x)
        x= self.trans1(x)
        x= self.conv2(x)
        x= self.trans2(x)
        x= x.view(x.size(0), -1)
        x= self.fc(x)
        return F.log_softmax(x, dim=1)
        
        
        
 #S7 assignments MODELS        
        
dropout_value = 0.1
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Input Block
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
            
        ) # output_size = 26

      
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
            
        ) # output_size = 24
        

        # TRANSITION BLOCK 1
       
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
        )
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11 
        # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 9
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Dropout(dropout_value))
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Dropout(dropout_value)
           
        ) # output_size = 7
        
        self.gap = nn.Sequential (nn.AvgPool2d(kernel_size=4))
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x= self.gap(x)    
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
    
    
    

    
 #STEP 2

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # Input Block
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
            
        ) # output_size = 26

      
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
            
        ) # output_size = 24
        

        # TRANSITION BLOCK 1
       
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=6, kernel_size=(1, 1), padding=0, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
        )
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11 
        # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # OUTPUT BLOCK
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Dropout(dropout_value)) # Output Size =6
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Dropout(dropout_value) #Output size= 4
           
        ) # output_size = 7
        
        #self.gap = nn.Sequential (nn.AvgPool2d(kernel_size=4))
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=4, Output=1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        #x= self.gap(x)    
        x = self.convblock8(x)
        x= self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
    
 

 #STEP 3


class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        # Input Block
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
            
        ) # output_size = 26

      
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
            
        ) # output_size = 24
        

        # TRANSITION BLOCK 1
       
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=6, kernel_size=(1, 1), padding=0, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
        )
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11 
        # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 9
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Dropout(dropout_value))
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(16),
             nn.Dropout(dropout_value)
           
        ) # output_size = 7
        
        #self.gap = nn.Sequential (nn.AvgPool2d(kernel_size=4))
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        #x= self.gap(x)    
        x = self.convblock8(x)
        x= self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# S8 asignment 

#Model3 Wwith BN 
import torch.nn as nn
import torch.nn.functional as F
class Model_3BN(nn.Module):
    def __init__(self, drop=0.025):
        super(Model_3BN, self).__init__()

        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(3, 3), padding= 1, bias=False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),)
        
        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop),)
        
        self.convblock1c = nn.Sequential(    
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.transblock1 =  nn.Sequential(
            #Stride 2 conv
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=(1, 1), bias=False),  # Input: 32x32x32 | Output: 16x16x32 | RF: 7x7
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            #nn.Dropout(drop),

        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock2a = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop))
        
        self.convblock2b  = nn.Sequential( 
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding= 1, bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop)) 
            
        self.convblock2c  = nn.Sequential(  
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),  
    
        ) 

        self.transblock2 =  nn.Sequential(
            #Stride 2 conv
            #DepthwiseSeparable(64,64,2), # Input: 16x16x64 | Output: 8x8x64 | RF: 15x15
            nn.Conv2d(in_channels=32, out_channels=20, kernel_size=(1, 1), bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            ##nn.Dropout(drop),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock3a = nn.Sequential(          
            nn.Conv2d(in_channels=20, out_channels=42, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(42),
            nn.ReLU(),
            nn.Dropout(drop))
        
        self.convblock3b=nn.Sequential(
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(3, 3),padding =1, bias=False), # Input: 8x8x128 | Output: 8x8x32 | RF: 23x23
            nn.BatchNorm2d(42),
            nn.ReLU(),
            nn.Dropout(drop) 

        ) 
        self.transblock3 =  nn.Sequential(
            nn.Conv2d(in_channels=42, out_channels=10, kernel_size=(1, 1), bias=False), # Input: 8x8x128 | Output: 6x6x32 | RF: 39x39
            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # Input: 6x6x32 | Output: 4x4x32 | RF: 55x55
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            #nn.Dropout(drop),
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x =  self.convblock1a(x)
        x =  self.convblock1b(x)
        x =  self.convblock1c(x)+x
        x =  self.transblock1(x)
        x =  self.pool1(x)
        x =  self.convblock2a(x)
        x =  self.convblock2b(x)
        x =  self.convblock2c(x)
        x =  self.transblock2(x)
        x =  self.pool2(x)
        x =  self.convblock3a(x)
        x =  self.convblock3b(x)+x
        x =  self.transblock3(x)
        x= self.global_avgpool(x)     
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

#S8 Group normalization added 

class Model_3GN(nn.Module):
    def __init__(self, drop=0.025):
        super(Model_3GN, self).__init__()

        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(3, 3), padding= 1, bias=False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),)
        
        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop),)
        
        self.convblock1c = nn.Sequential(    
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.transblock1 =  nn.Sequential(
            #Stride 2 conv
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=(1, 1), bias=False),  # Input: 32x32x32 | Output: 16x16x32 | RF: 7x7
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            #nn.Dropout(drop),

        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock2a = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop))
        
        self.convblock2b  = nn.Sequential( 
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding= 1, bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.GroupNorm(num_groups=3, num_channels=24),
            nn.ReLU(),
            nn.Dropout(drop)) 
            
        self.convblock2c  = nn.Sequential(  
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),  
    
        ) 

        self.transblock2 =  nn.Sequential(
            #Stride 2 conv
            #DepthwiseSeparable(64,64,2), # Input: 16x16x64 | Output: 8x8x64 | RF: 15x15
            nn.Conv2d(in_channels=32, out_channels=20, kernel_size=(1, 1), bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            ##nn.Dropout(drop),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock3a = nn.Sequential(          
            nn.Conv2d(in_channels=20, out_channels=42, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(42),
            nn.ReLU(),
            nn.Dropout(drop))
        
        self.convblock3b=nn.Sequential(
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(3, 3),padding =1, bias=False), # Input: 8x8x128 | Output: 8x8x32 | RF: 23x23
            nn.BatchNorm2d(42),
            
            nn.ReLU(),
            nn.Dropout(drop) 

        ) 
        self.transblock3 =  nn.Sequential(
            nn.Conv2d(in_channels=42, out_channels=10, kernel_size=(1, 1), bias=False), # Input: 8x8x128 | Output: 6x6x32 | RF: 39x39
            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # Input: 6x6x32 | Output: 4x4x32 | RF: 55x55
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            #nn.Dropout(drop),
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x =  self.convblock1a(x)
        x =  self.convblock1b(x)
        x =  self.convblock1c(x)+x
        x =  self.transblock1(x)
        x =  self.pool1(x)
        x =  self.convblock2a(x)
        x =  self.convblock2b(x)
        x =  self.convblock2c(x)
        x =  self.transblock2(x)
        x =  self.pool2(x)
        x =  self.convblock3a(x)
        x =  self.convblock3b(x)+x
        x =  self.transblock3(x)
        x= self.global_avgpool(x)     
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
#Layer normailizatio
import torch.nn as nn
import torch.nn.functional as F
class Model_3(nn.Module):
    def __init__(self, drop=0.025):
        super(Model_3, self).__init__()

        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(3, 3), padding= 1, bias=False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),)
        
        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop),)
        
        self.convblock1c = nn.Sequential(    
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3),padding= 1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.transblock1 =  nn.Sequential(
            #Stride 2 conv
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=(1, 1), bias=False),  # Input: 32x32x32 | Output: 16x16x32 | RF: 7x7
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            #nn.Dropout(drop),

        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock2a = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3),padding= 1, bias=False, ),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            #nn.BatchNorm2d(12),
            nn.LayerNorm([12,16,16]),
            nn.ReLU(),
            nn.Dropout(drop))
        
        self.convblock2b  = nn.Sequential( 
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), padding= 1, bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(24), #nn.LayerNorm([24,5,5])
            nn.ReLU(),
            nn.Dropout(drop)) 
            
        self.convblock2c  = nn.Sequential(  
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),  
    
        ) 

        self.transblock2 =  nn.Sequential(
            #Stride 2 conv
            #DepthwiseSeparable(64,64,2), # Input: 16x16x64 | Output: 8x8x64 | RF: 15x15
            nn.Conv2d(in_channels=32, out_channels=20, kernel_size=(1, 1), bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            ##nn.Dropout(drop),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock3a = nn.Sequential(          
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(3, 3), stride =2, bias=False),
            #nn.BatchNorm2d(24),
            nn.LayerNorm([32,3,3]),
            nn.ReLU(),
            nn.Dropout(drop))
        
        self.convblock3b=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),padding =1, bias=False), # Input: 8x8x128 | Output: 8x8x32 | RF: 23x23
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop) 

        ) 
        self.transblock3 =  nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), bias=False), # Input: 8x8x128 | Output: 6x6x32 | RF: 39x39
            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # Input: 6x6x32 | Output: 4x4x32 | RF: 55x55
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            #nn.Dropout(drop),
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x =  self.convblock1a(x)
        x =  self.convblock1b(x)
        x =  self.convblock1c(x)+x
        x =  self.transblock1(x)
        x =  self.pool1(x)
        x =  self.convblock2a(x)
        x =  self.convblock2b(x)
        x =  self.convblock2c(x)
        x =  self.transblock2(x)
        x =  self.pool2(x)
        x =  self.convblock3a(x)
        x =  self.convblock3b(x)+x
        x =  self.transblock3(x)
        x= self.global_avgpool(x)     
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
