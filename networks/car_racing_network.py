import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, s_channels, a_size, kernel_size, stride, zero_padding):
        super().__init__():
        
        self.conv1 = nn.Conv2d(s_channels, 16, kernel_size=3, stride=1, zero_padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, zero_padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, zero_padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, zero_padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(6*6*128, 256)
        self.fc2 = nn.Linear(256, a_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
class Critic(nn.Module):
    ...

if __name__ == '__main__':
    ...
