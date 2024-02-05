import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class CNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x