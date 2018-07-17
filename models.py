## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Start: 224 x 224 x 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        # After Conv1: 220 x 220 x 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # After Pool1: 110 x 110 x 32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # After Conv2: 108 x 108 x 64
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # After Pool2: 54 x 54 x 64
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        # After Conv3: 54 x 54 x 128
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # After Pool3: 27 x 27 x 128
        
        
        self.linear1 = nn.Linear(27*27*128, 512)
        
        self.dropout = nn.Dropout2d()
        
        self.linear2 = nn.Linear(512, 136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.pool3(F.relu(self.conv3(x)))
                
        # flatten layers
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.linear1(x))
        
        x = F.dropout(x, training=self.training)
        
        x = self.linear2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
