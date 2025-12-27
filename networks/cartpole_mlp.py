import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, s_size, h_size, a_size):
        super().__init__()

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        return self.fc2(x)
