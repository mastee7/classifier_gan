import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        # initializes the parent class(nn.module)
        super().__init__()
        # Convolution layer that takes an input with 3 channels and produces 6 feature maps using 5*5 filters
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # Max-pooling layer with 2*2 window and stride of 2 for down-sample
        # Which will reduce the spatial dimensions of the feature maps
        self.pool = nn.MaxPool2d(2, 2)
        
        # Another layer taking 6 feature maps from previous layers as input and produces 16 features using 5*5 filter
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # These are Fully connected layers 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # forward pass goes through 2 convolution layers followed by ReLU activation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch to pass through FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # The weights are initizlied from normal dist with mean 0 and std 0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
