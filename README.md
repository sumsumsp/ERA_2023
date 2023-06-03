# ERA_2023
## Assignment 5 
### Assignment was to correct the code, label the codes (markdown file) and split the code in different files 
#### The whole code was split into 3 Files 
- model.py: contains mainly the model architecture 
- utils.py: contains data loader, transform method, train and test loop 
- S5.ipynb: This is a jupyter notebook run file which connect and load all files. 
-  This ReadME explains the S5 file according to heading Code blocks and then connecting to other two files 

##### Code Block 1 (S5)
Imported the necessary modules for deep learning using PyTorch and torchvision:  
  torch: providing data structures and operations for tensors and neural networks. 
  torch.nn: classes for building neural network architectures. 
  torch.nn.functional (imported as F):  It provides various activation functions, loss functions, and other operations commonly used in deep learning. 
  torch.optim: optimization algorithms such as stochastic gradient descent (SGD), Adam, and others. 
  torchvision.datasets: T datasets commonly used in computer vision, such as MNIST, CIFAR-10, etc. 
  torchvision.transforms: for preprocessing and augmenting images. 
  
##### Code Block 2 (S5)
   Check the GPU availability and connecting the device to gpu 
 
 #### Code Block 3 (importing from utils) 
   The code you provided defines two sets of data transformations: train_transforms and test_transforms. 
   The train_transforms variable represents the transformations to be applied to the training data. Here's a breakdown of the transformations in train_transforms:
         - transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1): This transformation randomly applies a center crop of size 22x22 to the input image with a probability of
            0.1. This helps introduce some variation in the training data by occasionally cropping the center of the image.
         - transforms.Resize((28, 28)): This transformation resizes the image to a fixed size of 28x28 pixels.
         - transforms.RandomRotation((-15., 15.), fill=0): This transformation randomly rotates the image within the range of -15 to 15 degrees. The fill=0 argument specifies that
           any pixels introduced by rotation should be filled with zeros.

transforms.ToTensor(): This transformation converts the image from a PIL Image object to a PyTorch tensor. PyTorch tensors are the primary data structure used in deep learning with PyTorch.

transforms.Normalize((0.1307,), (0.3081,)): This transformation normalizes the image tensor by subtracting the mean (0.1307) and dividing by the standard deviation (0.3081). Normalization helps bring the pixel values into a standard range, which can aid in model training.

The test_transforms variable represents the transformations to be applied to the test data. Here's a breakdown of the transformations in test_transforms:

transforms.ToTensor(): This transformation converts the image from a PIL Image object to a PyTorch tensor, similar to the ToTensor() transformation in train_transforms.

transforms.Normalize((0.1307,), (0.3081,)): This transformation performs the same normalization as in train_transforms, subtracting the mean and dividing by the standard deviation.
