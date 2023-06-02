
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the layers of the neural network
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias= False )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias= False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias= False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3,bias= False)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x), 2))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)