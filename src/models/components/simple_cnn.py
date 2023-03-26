import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_shape=[68, 2]):
        super(CNN, self).__init__()
        self.output_shape = output_shape

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_shape[0] * output_shape[1])

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.relu2(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.relu3(x)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = x.reshape(x.size(0), self.output_shape[0], self.output_shape[1])
        return x
