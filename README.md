# ERA_2023
## Assignment 5 
### Assignment was to correct the code, label the codes (markdown file) and split the code in different files 
#### The whole code was split into 3 Files 
- model.py: contains mainly the model architecture 
- utils.py: contains data loader, transform method, train and test loop 
- S5.ipynb: This is a jupyter notebook run file which connect and load all files. 
-  This ReadME explains the S5 file according to heading Code blocks and then connecting to other two files 

##### Code Block 1 (S5)
Imported the necessary modules for deep learning using PyTorch and torchvision. 
  torch: providing data structures and operations for tensors and neural networks.
  torch.nn: classes for building neural network architectures.
  torch.nn.functional (imported as F):  It provides various activation functions, loss functions, and other operations commonly used in deep learning.
  torch.optim: optimization algorithms such as stochastic gradient descent (SGD), Adam, and others.
  torchvision.datasets: T datasets commonly used in computer vision, such as MNIST, CIFAR-10, etc.
  torchvision.transforms: for preprocessing and augmenting images 
  
##### Code Block 2
