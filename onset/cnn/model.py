from torch import nn
import torch
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv2_dimensions, conv_out_dimensions, linear_out):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, conv2_dimensions, kernel_size=conv1_kernel_size)
        self.pool1 = nn.MaxPool2d((3, 1))
        self.conv2 = nn.Conv2d(conv2_dimensions, conv_out_dimensions, kernel_size=conv2_kernel_size)
        self.pool2 = nn.MaxPool2d((3, 1))
        self.fc1 = nn.LazyLinear(linear_out) 
        self.fc2 = nn.Linear(linear_out, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Sigmoid activation for output
        return x