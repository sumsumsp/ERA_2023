# ERA_2023
## Assignment 5 
### Assignment was to correct the code, label the codes (markdown file) and split the code in different files 
#### The whole code was split into 3 Files 
- model.py: contains mainly the model architecture 
- utils.py: contains data loader, transform method, train and test loop 
- S5.ipynb: This is a jupyter notebook run file which connect and load all files. 
-  This ReadME explains the S5 file according to heading Code blocks and then connecting to other two files 

##### Code Block 1 (S5)
_Imported the necessary modules for deep learning using PyTorch and torchvision:_    
  torch: providing data structures and operations for tensors and neural networks. 
  torch.nn: classes for building neural network architectures. 
  torch.nn.functional (imported as F):  It provides various activation functions, loss functions, and other operations commonly used in deep learning. 
  torch.optim: optimization algorithms such as stochastic gradient descent (SGD), Adam, and others. 
  torchvision.datasets: T datasets commonly used in computer vision, such as MNIST, CIFAR-10, etc. 
  torchvision.transforms: for preprocessing and augmenting images. 
  
##### Code Block 2 (S5)
   Check the GPU availability and connecting the device to gpu 
 
 #### Code Block 3 ( utils) 
   _The code you provided defines two sets of data transformations: train_transforms and test_transforms._    
   The **train_transforms** variable represents the transformations to be applied to the training data.        
         - transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1): This transformation randomly applies a center crop of size 22x22 to the input image with a probability of
            0.1. This helps introduce some variation in the training data by occasionally cropping the center of the image.  
         - transforms.Resize((28, 28)): This transformation resizes the image to a fixed size of 28x28 pixels.  
         - transforms.RandomRotation((-15., 15.), fill=0): This transformation randomly rotates the image within the range of -15 to 15 degrees. The fill=0 argument specifies that
           any pixels introduced by rotation should be filled with zeros.      
         - transforms.ToTensor(): This transformation converts the image from a PIL Image object to a PyTorch tensor.    
         - transforms.Normalize((0.1307,), (0.3081,)): Normalization helps bring the pixel values into a standard range, which can aid in model training.  
    The **test_transforms** variable represents the transformations to be applied to the test data

#### Code Block 4 (utils)
   The code you provided initializes the training and test datasets using the MNIST dataset from torchvision. The MNIST dataset consists of grayscale images of handwritten digits
   from 0 to 9.  
   ![Image Data set] (https://en.wikipedia.org/wiki/MNIST_database#/media/File:MnistExamples.png)
   
#### Code Block 5 (S5)
   _Sets up the data loaders for the training and test datasets._ 
   
   **batch_size = 512** : This line sets the batch size to 512.  
   **kwargs = {'batch_size': batch_size, 'num_workers': 12, 'pin_memory': True}** . The batch_size argument is set to the previously defined batch_size value. The num_workers argument specifies the number of subprocesses to use for data loading. The pin_memory argument is set to True, which enables faster data transfer to the GPU if you are using one.
   **train_loader = torch.utils.data.DataLoader(train_data.....)** : This line creates the data loader for the training dataset. The train_data argument is the training dataset object (datasets.MNIST) created earlier. The shuffle=True argument shuffles the training data before each epoch to introduce randomness and prevent the model from overfitting. **test_loader = torch.utils.data.DataLoader(test_data,......** This line creates the data loader for the test dataset. 

#### Code Block 6 (S5) 
  _Visualize the data iteration from train loader_ 
  
 #### Code Block 7 (S5) 
 **Buid the model**    
_model summary_     
 
 
       | Layer (type)       |Output Shape       |  Param       |
       
       |    Conv2d-1        |  [-1, 32, 26, 26] |          288 |  
            Conv2d-2           [-1, 64, 24, 24]          18,432
            Conv2d-3          [-1, 128, 10, 10]          73,728
            Conv2d-4            [-1, 256, 8, 8]         294,912
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510

**Total params: 592,720  
Trainable params: 592,720  
Non-trainable params: 0  
Estimated Total Size (MB): 2.93** 

